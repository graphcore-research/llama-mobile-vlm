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


def quantise_2d_fixed_(model: nn.Module, fmt_spec: FmtSpec) -> dict[str, Any]:
    """Quantise a model using a 'fixed' scheme.

    Returns a dictionary describing the quantisation result.
    """
    param_log = {}
    for name, param in model.named_parameters():
        if param.ndim == 2:
            quantise_parameter_(param, fmt_spec)
            param_log[name] = dict(
                nelement=param.nelement(),
                **param._quantised,
            )
        else:
            param_log[name] = dict(
                nelement=param.nelement(),
                bits=Q.TorchFormat(param.dtype).count_bits_tensor(param),
            )
    return dict(
        bits_per_param=sum(p["bits"] for p in param_log.values())
        / sum(p["nelement"] for p in param_log.values()),
        params=param_log,
    )
