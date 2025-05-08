# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

"""Utilities for quantising `nn.Module`."""

import dataclasses
from typing import Any

import torch
from torch import nn, Tensor

from . import fit as F
from . import quantisation as Q

FmtSpec = Q.TensorFormat | F.Scaled


def quantise_parameter_(
    param: nn.Parameter, fmt_spec: FmtSpec, error_weight: Tensor | None = None
) -> None:
    """Quantise a parameter in-place.

    Attaches a dictionary containing quantisation stats under `param._quantised`.
    """
    if hasattr(param, "_quantised"):
        raise ValueError(f"Param of shape {tuple(param.shape)} was already quantised")
    with torch.no_grad():
        if isinstance(fmt_spec, Q.TensorFormat):
            if error_weight is not None:
                raise ValueError(
                    "Cannot use `quantise_parameter_` with a Q.TensorFormat"
                    " and error_weight, which requires a F.Scaled."
                )
            fmt = fmt_spec
        elif isinstance(fmt_spec, F.Scaled):
            fmt = fmt_spec.fit(param, error_weight)
        new_value = fmt.quantise(param)
        param._quantised = dict(
            fmt=fmt,
            bits=fmt.count_bits_tensor(param),
            rmse=(new_value - param).float().pow(2).mean().sqrt().item(),
        )
        param[...] = new_value


DEFAULT_IGNORE = ("vision_model", "multi_modal_projector")


def _named_parameters_to_quantise(
    model: nn.Module, ignore: tuple[str]
) -> list[tuple[str, nn.Parameter]]:
    """Find all named_parameters in a model that should be quantised."""
    params = []
    for name, param in model.named_parameters():
        if param.ndim == 2 and not any(p in ignore for p in name.split(".")):
            params.append((name, param))
    return params


def _quantisation_log(
    model: nn.Module, verbose: bool, ignore: tuple[str]
) -> dict[str, Any]:
    log = {}
    total_bits, total_nelement = 0, 0
    for name, param in model.named_parameters():
        if any(p in ignore for p in name.split(".")):
            pass
        elif hasattr(param, "_quantised"):
            log[name] = param._quantised.copy()
            if not verbose:
                log[name].pop("fmt")
            total_bits += param._quantised["bits"]
        else:
            total_bits += Q.TorchFormat(param.dtype).count_bits_tensor(param)
        total_nelement += param.nelement()
    return dict(bits_per_param=total_bits / total_nelement, params=log)


def _quantise_named_parameter(name: str, *args: Any, **kwargs: Any) -> None:
    try:
        quantise_parameter_(*args, **kwargs)
    except Exception as e:
        raise ValueError(f"Failed to quantise {name!r}") from e


def quantise_2d_fixed(
    model: nn.Module,
    fmt_spec: FmtSpec,
    error_weight: dict[str, Tensor] | None = None,
    ignore: tuple[str, ...] = DEFAULT_IGNORE,
    verbose_log: bool = False,
) -> dict[str, Any]:
    """Quantise a model using a 'fixed' scheme.

    Only quantise 2D parameters and ignore anything under "vision_model" (default).

    Returns a dictionary describing the quantisation result.
    """
    for name, param in _named_parameters_to_quantise(model, ignore):
        _quantise_named_parameter(
            name, param, fmt_spec, error_weight[name] if error_weight else None
        )
    return _quantisation_log(model, verbose=verbose_log, ignore=ignore)


def quantise_2d_variable(
    model: nn.Module,
    fmt_spec: F.Scaled,
    fisher_sum: dict[str, float],
    error_weight: dict[str, Tensor] | None = None,
    min_element_bits: float | None = None,
    ignore: tuple[str, ...] = DEFAULT_IGNORE,
    verbose_log: bool = False,
) -> dict[str, Any]:
    """Quantise a model using a variable scheme based on Fisher sensitivity.

    Note that `fmt_spec.element_bits` is interpreted an approximate target for the
    model-wide average bits per parameter.

    `min_element_bits` -- default depends on `fmt_spec.element_family`,
        "fp" -- 3 bits
        otherwise -- 2 bits
    """
    if min_element_bits is None:
        min_element_bits = 3 if fmt_spec.element_family == "fp" else 2

    params_to_quantise = _named_parameters_to_quantise(model, ignore)
    nelement = torch.tensor([p.nelement() for _, p in params_to_quantise])
    rms = torch.tensor(
        [
            p.square().mean(dtype=torch.float32).sqrt().to(p.dtype)
            for _, p in params_to_quantise
        ]
    )
    fisher_mean = torch.tensor(
        [fisher_sum[n] / p.nelement() for n, p in params_to_quantise]
    )
    bit_delta = rms.log2() + 0.5 * fisher_mean.log2()
    bit_offset = fmt_spec.element_bits - (bit_delta * nelement).sum() / nelement.sum()
    for i, (name, param) in enumerate(params_to_quantise):
        bit_width = float(bit_offset + bit_delta[i])
        if fmt_spec.compressor is None:
            # Perhaps consider a tighter "global" method
            bit_width = int(round(bit_width))
        bit_width = max(bit_width, min_element_bits)
        _quantise_named_parameter(
            name,
            param,
            dataclasses.replace(fmt_spec, element_bits=bit_width),
            error_weight[name] if error_weight else None,
        )
    return _quantisation_log(model, verbose=verbose_log, ignore=ignore)


def quantise_2d_heuristic(
    model: nn.Module,
    fmt_spec: F.Scaled,
    highp_add_bits: float,
    highp_names: tuple[str, ...],
    highp_first_layers: int,
    highp_last_layers: int,
    error_weight: dict[str, Tensor] | None = None,
    ignore: tuple[str, ...] = DEFAULT_IGNORE,
    verbose_log: bool = False,
) -> dict[str, Any]:
    """Quantise a model, using higher precision for some layers.

    Only quantise 2D parameters and ignore anything under "vision_model" (default).

    Returns a dictionary describing the quantisation result.
    """
    try:
        layers = model.model.layers
    except AttributeError:
        layers = model.language_model.model.layers
    highp_params = set([])
    for layer in (
        layers[:highp_first_layers] + layers[len(layers) - highp_last_layers :]
    ):
        for param in layer.parameters():
            highp_params.add(param)

    for name, param in _named_parameters_to_quantise(model, ignore):
        fmt_spec_i = fmt_spec
        if any(p in highp_names for p in name.split(".")) or param in highp_params:
            fmt_spec_i = dataclasses.replace(
                fmt_spec_i, element_bits=fmt_spec.element_bits + highp_add_bits
            )
        _quantise_named_parameter(
            name, param, fmt_spec_i, error_weight[name] if error_weight else None
        )
    return _quantisation_log(model, verbose=verbose_log, ignore=ignore)
