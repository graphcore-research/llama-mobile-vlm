# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

"""A wrapper of `quantisation` to automatically fit quantisers to data."""

import dataclasses
from dataclasses import dataclass
from math import log2, prod
from typing import Any, Literal

import scipy.optimize
import torch
from torch import Tensor

from . import quantisation as Q

MAX_ROTATION_SIZE = 32768


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
    scaling_match: Literal["search", "moments"] = "search"
    sparse_format: Q.TensorFormat | None = None
    sparse_ratio: float = 0
    # If a tuple: (compressor, training_compressor)
    compressor: Q.Compressor | tuple[Q.Compressor, Q.Compressor] | None = None
    rotation: int | None = None
    args: dict[str, Any] = dataclasses.field(default_factory=lambda: {})

    _type: str = "fit_scaled"

    def __str__(self) -> str:
        block = ",".join("*" if g is None else str(g) for g in self.block_shape)
        compress = ""
        if self.compressor:
            if isinstance(self.compressor, tuple):
                out_compressor, train_compressor = self.compressor
                compress = f"+Z{out_compressor}|Z{train_compressor}"
            else:
                compress = f"+Z{self.compressor}"
        sparse = ""
        if self.sparse_ratio:
            sparse_ratio = format(
                self.sparse_ratio, ".1%" if 1e-3 <= self.sparse_ratio else ".0e"
            )
            sparse = f"+S[{sparse_ratio}:{self.sparse_format}]"
        rotation = ""
        if self.rotation is not None:
            rotation = "+R"
        args = (
            "(" + ",".join(f"{k}={v}" for k, v in self.args.items()) + ")"
            if self.args
            else ""
        )
        return (
            f"{self.element_bits:.3g}b-{self.element_family}{args}{compress}"
            f"{{{block}:{self.scale_format}:{self.scaling}:{self.scaling_match}}}{rotation}{sparse}"
        )

    @property
    def supports_error_weight(self) -> bool:
        """We only support `error_weight is not None` when there is an optimisation to perform.

        E.g. element_family=lloyd_max optimises quantisation bins,
             element_family=fp/t optimise exponent_bits/df respectively (unless specified),
             scaling=rms optimises format scale.
        """
        if self.compressor is not None:
            return False
        if self.element_family == "lloyd_max":
            return True
        if self.element_family == "fp" and "exponent_bits" not in self.args:
            return True
        if self.element_family == "t" and "df" not in self.args:
            return True
        return self.scaling_match == "search"

    def fit(self, tensor: Tensor, error_weight: Tensor | None = None) -> Q.TensorFormat:
        if error_weight is not None and not self.supports_error_weight:
            raise ValueError(f"fit.Scaled({self}) doesn't support `error_weight`")

        if self.sparse_ratio:
            tensor, _, _ = Q.SparseFormat.split(tensor, self.sparse_ratio)
        if self.rotation:
            # Avoid rotating the vocabulary dimension, too large
            rotation_dims = tuple(
                d for d, s in enumerate(tensor.shape) if s <= MAX_ROTATION_SIZE
            )
            tensor, _ = Q.RandomRotationFormat.rotate(
                tensor, rotation_dims, self.rotation
            )
        format = Q.LinearScalingFormat(
            _scaled_element_format(
                tensor,
                error_weight=error_weight,
                element_bits=self.element_bits,
                element_family=self.element_family,
                scale_format=self.scale_format,
                block_shape=self.block_shape,
                scaling=self.scaling,
                scaling_match=self.scaling_match,
                compressor=self.compressor,
                args=self.args,
            ),
            self.scale_format,
            self.block_shape,
            self.scaling,
        )
        if self.rotation:
            format = Q.RandomRotationFormat(format, rotation_dims, self.rotation)
        if self.sparse_ratio:
            format = Q.SparseFormat(format, self.sparse_format, self.sparse_ratio)
        return format


def _find_compressed_grid_quantiser(
    tensor: Tensor,
    amax: Tensor,
    compressor: Q.Compressor | tuple[Q.Compressor, Q.Compressor],
    args: dict[str, Any],
    target_bits: float,
    max_n: int = 2**24,
) -> Q.CompressedLUTFormat:
    if isinstance(compressor, tuple):
        out_compressor, train_compressor = compressor
    else:
        out_compressor = train_compressor = compressor

    def fmt(half_n: float) -> Q.CompressedLUTFormat:
        # Use an odd number of grid datapoints to represent zero (can be critical)
        # Don't use train_grid, since it rounds up the element range > absmax,
        # which causes problems with block-absmax scaling
        n = round(half_n) * 2 + 1
        return Q.CompressedLUTFormat.train(
            Q.LUTFormat.create(torch.linspace(-amax, amax, n), f"GRID{{n={n:.0f}}}"),
            tensor,
            compressor=train_compressor,
            **args,
        )

    # Line search for a half_n that is too large
    # Start at a lower bound on half_n, based on a uniform distribution
    half_n_max = 2 ** (target_bits - 1)
    while fmt(half_n_max).count_bits_tensor(tensor) / tensor.nelement() < target_bits:
        half_n_max *= 2
        if half_n_max >= max_n // 2:
            return fmt(half_n_max)  # degenerate case - give up

    half_n = scipy.optimize.bisect(
        lambda hn: fmt(hn).count_bits_tensor(tensor) / tensor.nelement() - target_bits,
        half_n_max / 2,
        half_n_max,
        xtol=1,
    )
    return dataclasses.replace(fmt(half_n), compressor=out_compressor)


def _scaled_element_format(
    tensor: Tensor,
    error_weight: Tensor | None,
    element_bits: float,
    element_family: Literal["int", "fp", "normal", "laplace", "t", "lloyd_max"],
    scale_format: Q.TensorFormat,
    block_shape: Q.BlockShape,
    scaling: Q.Scaling,
    scaling_match: Literal["search", "moments"],
    compressor: Q.Compressor | tuple[Q.Compressor, Q.Compressor] | None,
    args: dict[str, Any],
) -> Q.TensorFormat:
    """Fit a scaled element format to the given tensor."""

    def normalised(element_range: tuple[float, float]) -> tuple[Tensor, Tensor]:
        return Q.block_normalise(
            tensor,
            block_shape=block_shape,
            scaling=scaling,
            scale_format=scale_format,
            element_range=element_range,
        )

    if compressor is not None:
        # Find a grid resolution to hit the target `element_bits`
        if element_family != "int":
            raise ValueError(
                'fit.Scaled with compression only supports element_family="int"'
            )
        # Note: element_range=(-1, 1) is safe for absmax|signmax, since
        # _find_compressed_grid_quantiser returned format has range (-amax, amax)
        tensor, _ = normalised((-1, 1))
        return _find_compressed_grid_quantiser(
            tensor,
            tensor.abs().max() if scaling == "rms" else 1,
            compressor,
            args,
            target_bits=element_bits,
        )

    if element_family == "lloyd_max":
        # Train a Lloyd-Max quantiser on the normalised tensor
        args = args.copy()
        args.setdefault("init", "kmeans++" if scaling == "rms" else "uniform_minmax")
        args.setdefault("threshold", 1e-4)
        # Note: report the range consistently with training, not based on the actual absmax
        tensor, _ = normalised((-1, 1))
        return Q.lut_lloyd_max(
            tensor, element_bits, weight=error_weight, range=(-1, 1), **args
        )

    FIND_SCALED_FORMAT_STEPS = 17

    def find_scaled_format(
        format: Q.ScalarFormat, steps: int = FIND_SCALED_FORMAT_STEPS
    ) -> tuple[Q.ScaledFormat, float]:
        """Search for the best (RMSE) scaled element format, allowing clipping."""

        norm_tensor, norm_scale = normalised(format.range)

        def _eval(log_s: float) -> tuple[Q.ScaledFormat, float]:
            scaled = Q.ScaledFormat.create(
                format, 2**log_s, range=None if scaling == "rms" else format.range
            )
            rmse_norm = Q.rmse_norm(
                tensor,
                scaled.quantise(norm_tensor) * norm_scale,
                weight=error_weight,
            ).item()
            return scaled, rmse_norm

        base_scale = 1
        if scaling == "rms" and format._type == "int":
            # A base scale of 1 for integer is very bad. Use is the appropriate RMS
            # scale if the data were Uniform(-1, 1)
            base_scale = 3**0.5 / format.range[1]

        if scaling_match == "moments":
            return _eval(log2(base_scale))
        if scaling_match == "search":
            # A brute-force search, because the space can be multimodal & this needs to
            # be robust (could also use scipy.optimize.basinhopping, but it's less
            # predictable)
            scaled_and_rmse = [
                _eval(log_s)
                for log_s in torch.linspace(
                    log2(base_scale / 4), log2(base_scale * 4), steps
                ).tolist()
            ]
            return min(scaled_and_rmse, key=lambda x: x[1])
        assert False, f"unexpected scaling_match={scaling_match!r}"

    if element_family == "int":
        return find_scaled_format(Q.IntFormat(element_bits, **args))[0]

    if element_family == "fp":
        # Search over exponent_bits
        args = args.copy()
        args.setdefault("rounding", "nearest")
        assert (
            "mantissa_bits" not in args
        ), 'cannot specify args["mantissa_bits"] to F.Scaled(element_type="fp")'
        assert (
            round(element_bits) == element_bits
        ), 'fractional `element_bits` are unsupported for F.Scaled(element_type="fp")'
        element_bits = int(element_bits)

        candidate_exponent_bits = (
            [args.pop("exponent_bits")]
            if "exponent_bits" in args
            else list(range(2, element_bits))
        )
        format_with_rmse = [
            find_scaled_format(Q.FPFormat(e, element_bits - e - 1, **args))
            for e in candidate_exponent_bits
        ]
        return min(format_with_rmse, key=lambda x: x[1])[0]

    if element_family == "normal":
        return find_scaled_format(
            Q.crd_normal(element_bits, **args)
            if scaling == "rms" or any(b is None for b in block_shape)
            else Q.crd_block_normal(
                element_bits, prod(block_shape), scaling=scaling, **args
            )
        )[0]

    if element_family == "laplace":
        return find_scaled_format(
            Q.crd_laplace(element_bits, **args)
            if scaling == "rms" or any(b is None for b in block_shape)
            else Q.crd_block_laplace(
                element_bits, prod(block_shape), scaling=scaling, **args
            )
        )[0]

    if element_family == "t":
        # Search over df
        args = args.copy()
        if "df" in args:
            candidate_df = [args.pop("df")]
            inner_steps = FIND_SCALED_FORMAT_STEPS
        else:
            candidate_df = (2 ** torch.linspace(log2(3), log2(100), 12)).tolist()
            inner_steps = min(9, FIND_SCALED_FORMAT_STEPS)
        format_with_rmse = [
            find_scaled_format(
                (
                    Q.crd_t(element_bits, df, **args)
                    if scaling == "rms" or any(b is None for b in block_shape)
                    else Q.crd_block_t(
                        element_bits, prod(block_shape), df, scaling=scaling, **args
                    )
                ),
                steps=inner_steps,
            )
            for df in candidate_df
        ]
        return min(format_with_rmse, key=lambda x: x[1])[0]

    assert False, f"unexpected element_family {element_family!r}"
