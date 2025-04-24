# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

"""Utilities for quantising `nn.Module`."""

import dataclasses
from typing import Any

import torch
from torch import nn

from . import fit as F
from . import quantisation as Q

FmtSpec = Q.TensorFormat | F.Scaled


def quantise_parameter_(param: nn.Parameter, fmt_spec: FmtSpec) -> None:
    """Quantise a parameter in-place.

    Attaches a dictionary containing quantisation stats under `param._quantised`.
    """
    if hasattr(param, "_quantised"):
        raise ValueError(f"Param of shape {tuple(param.shape)} was already quantised")
    with torch.no_grad():
        if isinstance(fmt_spec, Q.TensorFormat):
            fmt = fmt_spec
        elif isinstance(fmt_spec, F.Scaled):
            fmt = fmt_spec.fit(param)
        new_value = fmt.quantise(param)
        param._quantised = dict(
            bits=fmt.count_bits_tensor(param),
            rmse=(new_value - param).float().pow(2).mean().sqrt().item(),
            rms=param.float().pow(2).mean().sqrt().item(),
        )
        param[...] = new_value


DEFAULT_IGNORE = ("vision_model",)


def _named_parameters_to_quantise(
    model: nn.Module, ignore: tuple[str]
) -> list[tuple[str, nn.Parameter]]:
    """Find all named_parameters in a model that should be quantised."""
    params = []
    for name, param in model.named_parameters():
        if param.ndim == 2 and not any(p in ignore for p in name.split(".")):
            params.append((name, param))
    return params


def _quantisation_log(model: nn.Module) -> dict[str, Any]:
    log = {
        name: (
            dict(
                nelement=param.nelement(),
                **param._quantised,
            )
            if hasattr(param, "_quantised")
            else dict(
                nelement=param.nelement(),
                bits=Q.TorchFormat(param.dtype).count_bits_tensor(param),
            )
        )
        for name, param in model.named_parameters()
    }
    bits_per_param = sum(p["bits"] for p in log.values()) / sum(
        p["nelement"] for p in log.values()
    )
    return dict(bits_per_param=bits_per_param, params=log)


def _quantise_named_parameter_(
    name: str, param: nn.Parameter, fmt_spec: FmtSpec
) -> None:
    try:
        quantise_parameter_(param, fmt_spec)
    except Exception as e:
        raise ValueError(f"Failed to quantise {name!r}") from e


def quantise_2d_fixed_(
    model: nn.Module, fmt_spec: FmtSpec, ignore: tuple[str] = DEFAULT_IGNORE
) -> dict[str, Any]:
    """Quantise a model using a 'fixed' scheme.

    Only quantise 2D parameters and ignore anything under "vision_model" (default).

    Returns a dictionary describing the quantisation result.
    """
    for name, param in _named_parameters_to_quantise(model, ignore):
        _quantise_named_parameter_(name, param, fmt_spec)
    return _quantisation_log(model)


def quantise_2d_variable(
    model: nn.Module,
    fmt_spec: F.Scaled,
    fisher_sum: dict[str, float],
    ignore: tuple[str] = DEFAULT_IGNORE,
) -> dict[str, Any]:
    """Quantise a model using a variable scheme based on Fisher sensitivity."""
    params_to_quantise = _named_parameters_to_quantise(model, ignore)
    nelement = torch.tensor([p.nelement() for _, p in params_to_quantise])
    fisher_mean = torch.tensor(
        [fisher_sum[n] / p.nelement() for n, p in params_to_quantise]
    )
    bit_offset = (
        fmt_spec.element_bits
        - 0.5 * fisher_mean.log().mul(nelement).sum() / nelement.sum()
    )
    for (name, param), fisher_mean_i in zip(params_to_quantise, fisher_mean):
        bit_width = float(bit_offset + 0.5 * fisher_mean_i.log())
        if fmt_spec.compressor is None:
            # Perhaps consider a tighter "global" method
            # We also need a minimum bit width (e.g. 3 for FP)
            bit_width = int(round(bit_width))
        _quantise_named_parameter_(
            name, param, dataclasses.replace(fmt_spec, element_bits=bit_width)
        )
    return _quantisation_log(model)
