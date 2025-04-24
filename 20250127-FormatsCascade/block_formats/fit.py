# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

"""A wrapper of `quantisation` to automatically fit quantisers to data."""

from math import log2
from typing import Any, Literal
from dataclasses import dataclass
import dataclasses

import scipy.optimize
import torch
from torch import Tensor

from . import quantisation as Q


@dataclass
class Scaled:
    """A scaled tensor format, with "unbound" parameters that can be fit to data.

    args -- valid values depend on element_family
        int -- see `Q.IntFormat` (none)
        fp -- see `Q.FPFormat` (if `exponent_bits` is specified, disable as search axis)
        normal -- see `Q.crd_normal` and `Q.crd_block_normal`
        laplace -- see `Q.crd_laplace` and `Q.crd_block_laplace`
        t -- see `Q.crd_t` and `Q.crd_block_t` (if `df` is specified, disable as search axis)
        lloyd_max -- see `Q.lut_lloyd_max`, e.g. "init", "threshold"
    """

    element_bits: float
    element_family: Literal["int", "fp", "normal", "laplace", "t", "lloyd_max"]
    scale_format: Q.TensorFormat
    block_shape: Q.BlockShape
    scaling: Q.Scaling
    compressor: Q.Compressor | None
    args: dict[str, Any] = dataclasses.field(default_factory=lambda: {})

    _type: str = "fit_scaled"

    def __str__(self) -> str:
        block = ",".join("*" if g is None else str(g) for g in self.block_shape)
        compress = f"+Z{self.compressor}" if self.compressor else ""
        args = (
            "(" + ",".join(f"{k}={v}" for k, v in self.args.items()) + ")"
            if self.args
            else ""
        )
        return f"{self.element_bits}b-{self.element_family}{args}{compress}{{{block}:{self.scale_format}:{self.scaling}}}"

    def fit(self, tensor: Tensor, weight: Tensor | None = None) -> Q.TensorFormat:
        return _scaled_quantiser(
            tensor,
            weight=weight,
            element_bits=self.element_bits,
            element_family=self.element_family,
            scale_format=self.scale_format,
            block_shape=self.block_shape,
            scaling=self.scaling,
            compressor=self.compressor,
            args=self.args,
        )


def _fit_scale(
    tensor: Tensor,
    format: Q.TensorFormat,
    weight: Tensor | None,
    bounds: tuple[float, float],
) -> Q.ScaledFormat:
    """Wrap `format` in a `ScaledFormat` that is tuned to optimise RMSE."""

    fmt = lambda log_s: Q.ScaledFormat(format, 2**log_s)
    opt = scipy.optimize.minimize_scalar(
        lambda log_s: Q.qrmse_norm(fmt(log_s), tensor, weight=weight).item(),
        bounds=(log2(bounds[0]), log2(bounds[1])),
        options=dict(xatol=0.1),
    )
    return fmt(opt.x)


def _scaled_quantiser(
    tensor: Tensor,
    weight: Tensor | None,
    element_bits: float,
    element_family: Literal["int", "fp", "normal", "laplace", "t", "lloyd_max"],
    scale_format: Q.TensorFormat,
    block_shape: Q.BlockShape,
    scaling: Q.Scaling,
    compressor: Q.Compressor | None,
    args: dict[str, Any],
) -> Q.TensorFormat:
    """Fit a scaled quantiser to the given tensor."""

    if compressor is not None:
        assert (
            element_family == "int"
        ), 'scaled_quantiser only supports compression with element_family="int"'

        # Search to find the grid resolution matching the target `element_bits`
        tensor, _ = Q.block_normalise(
            tensor,
            block_shape,
            scaling,
            element_range=(-1, 1),
            scale_format=scale_format,
        )
        # Note: don't use train_grid, since it rounds up the element range > absmax,
        # which causes problems with block-absmax scaling
        amax = tensor.abs().max() if scaling == "rms" else 1
        fmt = lambda b: Q.CompressedLUTFormat.train(
            Q.LUTFormat.create(
                torch.linspace(-amax, amax, round(2**b)), f"GRID{{n={round(2**b):.0f}}}"
            ),
            tensor,
            compressor=compressor,
        )
        opt = scipy.optimize.minimize_scalar(
            lambda b: abs(
                fmt(b).count_bits_tensor(tensor) / tensor.nelement() - element_bits
            ),
            # Note: +16 is for heavy-tailed distributions (fits Student-t, df >= 2)
            bounds=(element_bits, element_bits + 16),
            options=dict(xatol=0.01),
        )
        return Q.LinearScalingCompressionFormat(
            fmt(opt.x), scale_format, block_shape, scaling
        )

    tensor, _ = Q.block_normalise(
        tensor,
        block_shape,
        scaling,
        element_range=(
            Q.IntFormat(element_bits).range if element_family == "int" else (-1, 1)
        ),
        scale_format=scale_format,
    )
    if element_family == "lloyd_max":
        args = args.copy()
        args.setdefault("init", "kmeans++" if scaling == "rms" else "uniform_minmax")
        args.setdefault("threshold", 1e-4)
        element_format = Q.lut_lloyd_max(tensor, element_bits, weight=weight, **args)
    else:
        if scaling == "rms":
            base_scale = (
                3**0.5 / Q.IntFormat(element_bits).range[1]
                if element_family == "int"
                else 1.0
            )
            fit_scale = lambda fmt: _fit_scale(
                tensor, fmt, weight, (base_scale / 16, base_scale * 16)
            )
        else:
            fit_scale = lambda fmt: fmt

        if element_family in ("int", "normal", "laplace"):
            # No hyperparameters, except scale
            element_format = fit_scale(
                dict(int=Q.IntFormat, normal=Q.crd_normal, laplace=Q.crd_laplace)[
                    element_family
                ](element_bits, **args)
            )

        elif element_family == "fp":
            args = args.copy()
            args.setdefault("rounding", "nearest")
            if "exponent_bits" in args:
                args.setdefault(
                    "mantissa_bits", element_bits - args["exponent_bits"] - 1
                )
                element_format = fit_scale(Q.FPFormat(**args))
            else:
                # Exhaustive search over exponents
                fmts = [
                    fit_scale(Q.FPFormat(e, element_bits - e - 1, **args))
                    for e in range(2, element_bits)
                ]
                element_format = min(
                    fmts,
                    key=lambda fmt: Q.qrmse_norm(fmt, tensor, weight=weight).item(),
                )

        elif element_family == "t":
            if "df" in args:
                element_format = fit_scale(Q.crd_t(element_bits, **args))
            else:
                # 1D search over df
                fmt = lambda log2df: fit_scale(Q.crd_t(element_bits, 2**log2df, **args))
                opt = scipy.optimize.minimize_scalar(
                    lambda log2df: Q.qrmse_norm(
                        fmt(log2df), tensor, weight=weight
                    ).item(),
                    bounds=(log2(3), log2(100)),
                    options=dict(xatol=0.1),
                )
                element_format = fmt(opt.x)

        else:
            assert False, f"unexpected element_family {element_family!r}"

    return Q.LinearScalingFormat(element_format, scale_format, block_shape, scaling)
