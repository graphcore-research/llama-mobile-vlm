# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Utilities for "fake quantisation"."""

import bz2
import itertools as it
import math
import re
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Optional,
    Tuple,
    TypeAlias,
    Union,
    cast,
)

import scipy.stats
import torch
import tqdm
from torch import Tensor

Shape = Tuple[int, ...]

# Utilities


def shuffle(t: Tensor) -> Tensor:
    """Shuffle the flattened tensor, then reassemble."""
    y = torch.empty_like(t.flatten())
    y[torch.randperm(t.nelement(), device=t.device, dtype=torch.int32)] = t.flatten()
    return y.view(t.shape)


def rmse_norm(x: Tensor, qx: Tensor, weight: Tensor | None = None) -> Tensor:
    """RMS error of quantisation, normalised by original tensor RMS."""
    x = x.float()
    d2 = qx.to(torch.float32, copy=True).sub_(x).square_()
    x2 = x.square()
    if weight is not None:
        d2.mul_(weight)
        x2.mul_(weight)
    return (d2.sum() / x2.sum()).sqrt()


def qrmse_norm(
    fmt: "TensorFormat", tensor: Tensor, weight: Tensor | None = None
) -> Tensor:
    """RMS error of quantisation, normalised by original tensor RMS."""
    return rmse_norm(tensor, fmt.quantise(tensor), weight=weight)


def snr(x: Tensor, qx: Tensor) -> Tensor:
    """Signal-to-noise ratio."""
    x = x.float()
    qx = qx.float()
    return x.pow(2).sum() / (qx - x).pow(2).sum()


# Tensor formats


class TensorFormat:
    """Quantisation formats for tensors."""

    def quantise(self, tensor: Tensor) -> Tensor:
        raise NotImplementedError

    def count_bits(self, shape: Shape) -> int:
        raise NotImplementedError

    def count_bits_tensor(self, tensor: Tensor) -> float:
        return self.count_bits(tensor.shape)


# Scalar formats


@dataclass
class ScalarFormat(TensorFormat):
    """Elementwise scalar formats (abstract base class).

    Subclasses define: `_type`, `__str__`, `bits`, `range`, `quantise`
    """

    def __str__(self) -> str:
        raise NotImplementedError

    @property
    def bits(self) -> float:
        raise NotImplementedError

    @property
    def range(self) -> tuple[float, float]:
        raise NotImplementedError

    def count_bits(self, shape: Shape) -> int:
        return int(math.ceil(self.bits * math.prod(shape)))


@dataclass
class FPFormat(ScalarFormat):
    """Note that this format does not reserve an exponent code for specials.

    For exponent e : [0, 2^E - 1], mantissa m : [0, 2^M - 1], the represented value is:

        2^(e - 2^(E-1))) * (1 + m / 2^M)   if e != 0  (normal)
        2^(1 - 2^(E-1))) * (m / 2^M)       if e == 0  (subnormal)
    """

    exponent_bits: int
    mantissa_bits: int
    rounding: Literal["nearest", "to_inf", "to_zero"]
    _type: str = "fp"

    def __post_init__(self) -> None:
        if self.exponent_bits < 2 or self.mantissa_bits < 0:
            raise ValueError(
                f"FPFormat(exponent_bits={self.exponent_bits},"
                f" mantissa_bits={self.mantissa_bits}) is invalid"
                ", requiring exponent_bits >= 2, mantissa_bits >= 0"
            )

    def __str__(self) -> str:
        return f"E{self.exponent_bits}M{self.mantissa_bits}"

    @property
    def bits(self) -> float:
        return 1 + self.exponent_bits + self.mantissa_bits

    @property
    def range(self) -> tuple[float, float]:
        max_exponent = 2 ** (self.exponent_bits - 1) - 1
        absmax = cast(float, 2**max_exponent * (2 - 2**-self.mantissa_bits))
        return (-absmax, absmax)

    @property
    def min_absolute_normal(self) -> float:
        min_exponent = 1 - 2 ** (self.exponent_bits - 1)
        return cast(float, 2**min_exponent)

    @property
    def min_absolute_subnormal(self) -> float:
        return self.min_absolute_normal * 2.0**-self.mantissa_bits

    def quantise(self, x: Tensor) -> Tensor:
        assert x.dtype in [
            torch.float32,
            torch.bfloat16,
        ], "Quantising is only supported from bfloat16 and float32"

        downscale = 2.0 ** (127 - 2 ** (self.exponent_bits - 1))
        m_bits_before = {torch.float32: 23, torch.bfloat16: 7}[x.dtype]
        int_dtype = {torch.float32: torch.int32, torch.bfloat16: torch.int16}[x.dtype]
        mask = (
            2 ** (m_bits_before - self.mantissa_bits) - 1
            if m_bits_before > self.mantissa_bits
            else 0
        )
        if self.rounding == "nearest":
            offset = mask >> 1
        elif self.rounding == "to_inf":
            offset = mask
        elif self.rounding == "to_zero":
            offset = 0

        q = torch.clip(x, *self.range)
        q /= downscale
        q = ((q.view(int_dtype) + offset) & ~mask).view(x.dtype)
        q *= downscale

        return q.to(x.dtype)


@dataclass
class FPFormat_NoSign(FPFormat):
    _type: str = "fp_nosign"

    def __str__(self) -> str:
        return f"{super().__str__()}+"

    @property
    def bits(self) -> float:
        return self.exponent_bits + self.mantissa_bits

    def quantise(self, x: Tensor) -> Tensor:
        return super().quantise(x.clamp_min(0))


@dataclass
class TorchFormat(ScalarFormat):
    dtype: str
    _type: str = "torch"

    def __init__(self, dtype: Union[torch.dtype, str], _type: str = "torch"):
        assert _type == "torch"
        self.dtype = str(dtype).replace("torch.", "")

    @property
    def torch_dtype(self) -> torch.dtype:
        return cast(torch.dtype, getattr(torch, self.dtype))

    def quantise(self, x: Tensor) -> Tensor:
        return torch.clip(x, *self.range).to(self.torch_dtype).to(x.dtype)

    def __str__(self) -> str:
        return str(self.torch_dtype).replace("torch.", "").upper()

    @property
    def bits(self) -> float:
        torch_dtype = self.torch_dtype
        return (
            torch.finfo(torch_dtype).bits
            if torch_dtype.is_floating_point
            else torch.iinfo(torch_dtype).bits
        )

    @property
    def range(self) -> tuple[float, float]:
        torch_dtype = self.torch_dtype
        info = (
            torch.finfo(torch_dtype)
            if torch_dtype.is_floating_point
            else torch.iinfo(torch_dtype)
        )
        return (info.min, info.max)


@dataclass
class IntFormat(ScalarFormat):
    bits_: float
    mode: Literal["symmetric", "asymmetric"] = "asymmetric"
    _type: str = "int"

    def __post_init__(self) -> None:
        assert self.mode in (
            "symmetric",
            "asymmetric",
        ), f"unexpected IntFormat(mode={self.mode!r})"
        self.bits_ = math.log2(round(2.0**self.bits_))

    def __str__(self) -> str:
        suffix = "-S" if self.mode == "symmetric" else ""
        if int(self.bits_) == self.bits_:
            return f"E0M{self.bits_ - 1:.0f}{suffix}"
        return f"E0M{{{self.bits_ - 1:.2f}}}{suffix}"

    @property
    def bits(self) -> float:
        return self.bits_

    @property
    def range(self) -> tuple[float, float]:
        n_values = int(round(2.0**self.bits_))
        half_range = (
            (n_values - 1) // 2 if self.mode == "asymmetric" else (n_values - 1) / 2
        )
        return (-half_range - (2 * half_range + 1 < n_values), half_range)

    def quantise(self, x: Tensor) -> Tensor:
        n_values = int(round(2.0**self.bits_))
        offset = 0.5 if n_values % 2 == 0 and self.mode == "symmetric" else 0
        return torch.clip(torch.round(x + offset) - offset, *self.range)


@dataclass
class ExpCeilFormat(ScalarFormat):
    """An exponent-only format for positive numbers, with no zero, always rounding up."""

    bits_: int
    _type: str = "exp"

    def __str__(self) -> str:
        return f"EXP{self.bits_}"

    @property
    def bits(self) -> float:
        return self.bits_

    @property
    def range(self) -> tuple[float, float]:
        return (
            cast(float, 2 ** (-self.exponent_bias)),
            cast(float, 2 ** (2**self.bits_ - 1 - self.exponent_bias)),
        )

    @property
    def exponent_bias(self) -> float:
        return 2.0 ** (self.bits_ - 1) - 1

    def quantise(self, x: Tensor) -> Tensor:
        y: Tensor = 2 ** torch.clip(
            torch.ceil(torch.log2(x)),
            -self.exponent_bias,
            2**self.bits_ - 1 - self.exponent_bias,
        )
        return y


@dataclass
class LUTFormat(ScalarFormat):
    values: tuple[float, ...]
    name: str
    _range: tuple[float, float]
    _type: str = "lut"

    @classmethod
    def create(
        cls,
        values: tuple[float, ...] | Tensor,
        name: str,
        range: tuple[float, float] | None = None,
    ) -> "LUTFormat":
        values = tuple(values.tolist() if isinstance(values, Tensor) else values)
        return cls(values=values, name=name, _range=range or (min(values), max(values)))

    def __post_init__(self) -> None:
        self.values = tuple(self.values)

    def __str__(self) -> str:
        return f"LUT{int(math.ceil(self.bits))}[{self.name}]"

    @property
    def range(self) -> tuple[float, float]:
        return self._range

    @property
    def bits(self) -> float:
        return math.log2(len(self.values))

    def to_idx(self, x: Tensor) -> Tensor:
        # This has slightly worse accuracy if computed in x.dtype, so use float32
        values = torch.tensor(self.values, device=x.device)
        boundaries = (values[1:] + values[:-1]).div(2)
        return torch.bucketize(x, boundaries, out_int32=True)

    def quantise(self, x: Tensor) -> Tensor:
        values = torch.tensor(self.values, device=x.device, dtype=x.dtype)
        return values[self.to_idx(x)]


def parse(value: str) -> ScalarFormat:
    if value == "FP32":
        return FP32
    if value == "FP16":
        return FP16
    if value == "BFLOAT16":
        return BFLOAT16
    m = re.match(r"^E(\d+)M(\d+)(-(RN|RZ|RI))?$", value)
    if m:
        exponent_bits = int(m.group(1))
        mantissa_bits = int(m.group(2))
        if exponent_bits == 0:
            assert not m.group(3)
            return IntFormat(1 + mantissa_bits)
        if exponent_bits >= 2:
            rounding = cast(
                Literal["nearest", "to_inf", "to_zero"],
                {
                    None: "nearest",
                    "-RN": "nearest",
                    "-RZ": "to_zero",
                    "-RI": "to_inf",
                }[m.group(3)],
            )
            return FPFormat(exponent_bits, mantissa_bits, rounding)
        raise ValueError(f"No format {value!r} available (note: E1M6 == E0M7)")
    m = re.match(r"EXP(\d+)", value)
    if m:
        return ExpCeilFormat(int(m.group(1)))
    raise ValueError(f"Couldn't parse {value!r}")


def lut_function(fn: Callable[[Tensor], Tensor], bits: int, name: str) -> LUTFormat:
    """A lookup table quantiser based on mapping [-1, 1] via a function"""
    return LUTFormat.create(fn(torch.linspace(-1, 1, steps=2**bits)), name)


def lut_grid(resolution: float, max: float) -> LUTFormat:
    """A fixed-resolution grid that spans (-max, max)."""
    half_n = torch.tensor(max).div(resolution).ceil().long().item()
    values = torch.arange(-half_n, half_n + 1).mul(resolution)
    return LUTFormat.create(values, f"GRID{{{resolution}}}")


# Wrappers


@dataclass
class ScaledFormat(ScalarFormat):
    format: ScalarFormat
    scale: float
    _range: tuple[float, float]
    _type: str = "scaled"

    @classmethod
    def create(
        cls,
        format: ScalarFormat,
        scale: float,
        range: tuple[float, float] | None = None,
    ) -> "ScaledFormat":
        if range is None:
            min_, max_ = format.range
            range = (min_ * scale, max_ * scale)
        return cls(format=format, scale=scale, _range=range)

    def __str__(self) -> str:
        return f"{self.format}{{*{self.scale:.3g}}}"

    @property
    def bits(self) -> float:
        return self.format.bits

    @property
    def range(self) -> tuple[float, float]:
        return self._range

    def quantise(self, tensor: Tensor) -> Tensor:
        return self.format.quantise(tensor / self.scale) * self.scale

    def count_bits_tensor(self, tensor: Tensor) -> float:
        return self.format.count_bits_tensor(tensor / self.scale)


@dataclass
class RandomRotationFormat(TensorFormat):
    format: TensorFormat
    dims: tuple[int, ...]
    seed: int
    _type: str = "random_rotation"

    def __str__(self) -> str:
        return f"{self.format}{{rot{list(self.dims)}}}"

    @staticmethod
    def rotate(
        tensor: Tensor, dims: tuple[int, ...], seed: int
    ) -> tuple[Tensor, list[Tensor]]:
        """Returns (rotated, [rotations, ...])."""
        generator = torch.Generator(tensor.device).manual_seed(seed)
        rotations = [
            torch.nn.init.orthogonal_(
                torch.empty(tensor.shape[dim], tensor.shape[dim], device=tensor.device),
                generator=generator,
            ).to(tensor.dtype)
            for dim in dims
        ]
        for dim, rotation in zip(dims, rotations):
            tensor = (tensor.movedim(dim, -1) @ rotation).movedim(-1, dim)
        return tensor, rotations

    @staticmethod
    def unrotate(
        tensor: Tensor, dims: tuple[int, ...], rotations: list[Tensor]
    ) -> Tensor:
        for dim, rotation in zip(dims, rotations):
            tensor = (tensor.movedim(dim, -1) @ rotation.T).movedim(-1, dim)
        return tensor

    def quantise(self, tensor: Tensor) -> Tensor:
        tensor, rotations = self.rotate(tensor, self.dims, self.seed)
        return self.unrotate(self.format.quantise(tensor), self.dims, rotations)

    def count_bits(self, shape: Shape) -> int:
        return self.format.count_bits(shape)

    def count_bits_tensor(self, tensor: Tensor) -> float:
        tensor, _ = self.rotate(tensor, self.dims, self.seed)
        return self.format.count_bits_tensor(tensor)


@dataclass
class SparseFormat(TensorFormat):
    """A format wrapper that first removes a fixed percentage of absmax "outliers"."""

    format: TensorFormat
    sparse_format: ScalarFormat
    sparse_ratio: float
    _type: str = "outlier"

    @staticmethod
    def n_sparse(shape: Shape, sparse_ratio: float) -> int:
        return int(sparse_ratio * math.prod(shape))

    @classmethod
    def split(
        cls, tensor: Tensor, sparse_ratio: float
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Split-out and zero sparse (absmax) values.

        returns -- `(dense, sparse_idx, sparse_values)`
        """
        n_sparse = cls.n_sparse(tensor.shape, sparse_ratio)
        sparse_idx = torch.topk(tensor.abs().flatten(), n_sparse, sorted=False).indices
        dense = tensor.clone()
        dense.flatten()[sparse_idx] = 0
        return dense, sparse_idx, tensor.flatten()[sparse_idx]

    def __str__(self) -> str:
        sparse_ratio = format(
            self.sparse_ratio, ".1%" if 1e-3 <= self.sparse_ratio else ".0e"
        )
        return f"{self.format}+S[{sparse_ratio}:{self.sparse_format}]"

    def quantise(self, tensor: Tensor) -> Tensor:
        tensor, sparse_idx, sparse_values = self.split(tensor, self.sparse_ratio)
        tensor = self.format.quantise(tensor)
        tensor.flatten()[sparse_idx] = self.sparse_format.quantise(sparse_values)
        return tensor

    def count_sparse_bits(self, shape: Shape) -> int:
        n_sparse = self.n_sparse(shape, self.sparse_ratio)
        sparse_value_bits = self.sparse_format.count_bits((n_sparse,))
        sparse_mask_bits = 32 * n_sparse  # flat-COO format
        return sparse_value_bits + sparse_mask_bits

    def count_bits(self, shape: Shape) -> int:
        return self.format.count_bits(shape) + self.count_sparse_bits(shape)

    def count_bits_tensor(self, tensor: Tensor) -> float:
        tensor, _, _ = self.split(tensor, self.sparse_ratio)
        return self.format.count_bits_tensor(tensor) + self.count_sparse_bits(
            tensor.shape
        )


# Lloyd-Max


LloydMaxInit: TypeAlias = Union[
    Tensor,
    tuple[Literal["uniform_rms"], float],
    Literal["uniform_minmax", "kmeans++", "cuberoot"],
]


def _lloyd_max_init(
    init: LloydMaxInit, tensor: Tensor, weight: Tensor | None, codepoints: int
) -> Tensor:
    if isinstance(init, Tensor):
        assert init.shape == (codepoints,)
        return init.to(tensor.dtype, copy=True)
    if isinstance(init, tuple) and len(init) == 2 and init[0] == "uniform_rms":
        mean, std = tensor.mean(), tensor.std()
        return torch.linspace(
            mean - init[1] * std,
            mean + init[1] * std,
            codepoints,
            device=tensor.device,
            dtype=tensor.dtype,
        )
    if init == "uniform_minmax":
        return torch.linspace(
            tensor.min(),
            tensor.max(),
            codepoints,
            device=tensor.device,
            dtype=tensor.dtype,
        )
    if init == "kmeans++":
        s = tensor[: int(2**20)]
        midpoints = torch.empty(codepoints, device=s.device, dtype=s.dtype)
        p = torch.ones_like(s)
        for i in range(codepoints):
            midpoints[i] = s[torch.multinomial(p / p.sum(), 1)]
            midpoints[: i + 1] = midpoints[: i + 1].sort().values
            closest = torch.bucketize(s, (midpoints[:i] + midpoints[1 : i + 1]) / 2)
            p = (s - midpoints[closest]) ** 2
            if weight is not None:
                p *= weight[: len(s)].pow(2)
        return midpoints
    if init == "cuberoot":
        # Note: doesn't respect `weight`
        s = tensor[: int(2**20)].sort().values
        delta = (s[1:] - s[:-1]) ** (2 / 3)
        # delta += delta.mean()
        delta_sum = delta.cumsum(0)
        loc = torch.linspace(
            0, delta_sum[-1], codepoints + 2, device=s.device, dtype=s.dtype
        )[1:-1]
        # Note - it would be better to interpolate here, rather than round-to-nearest
        return s[torch.bucketize(loc, delta_sum)]
    raise ValueError(f"Unexpected init scheme {init}")


def lut_lloyd_max(
    tensor: Tensor,
    bits: float,
    threshold: float,
    *,
    weight: Tensor | None = None,
    range: tuple[float, float] | None = None,
    init: LloydMaxInit = "kmeans++",
    incremental: bool = True,
    max_samples: int | None = None,
    dtype: torch.dtype | None = None,
    progress: bool = False,
) -> LUTFormat:
    """Use Lloyd-Max (k-means) to find the RMS-optimal quantiser for the given tensor.

    threshold -- when the ratio of changed cluster assignments <= threshold, stop

    weight -- if provided, a positive tensor the same shape as `tensor`, to use as an
              importance weight for each sample

    incremental -- start with a subset of the data and scale up
    """

    # Preparation: shuffle, truncate, cast, get init
    idx = torch.randperm(tensor.nelement(), device=tensor.device, dtype=torch.int32)
    tensor = tensor.flatten()[idx]
    if weight is not None:
        weight = weight.flatten()[idx]

    if max_samples is not None:
        tensor = tensor[:max_samples]
    if dtype is None:
        # Very large tensors have stability problems due the float32
        # mantissa length, so default to float64
        dtype = torch.float32 if tensor.nelement() <= 2**26 else torch.float64
    tensor = tensor.to(dtype)
    midpoints = _lloyd_max_init(init, tensor, weight, int(round(2**bits)))
    if weight is not None:
        weight = weight.to(dtype)
        sum_weight = torch.empty_like(midpoints)

    # K-means iteration
    idx = torch.empty(tensor.shape, device=tensor.device, dtype=torch.int64)
    last_idx = torch.empty_like(idx)
    n = 2**20 if incremental else tensor.nelement()
    tqdm_ = tqdm.tqdm(it.count(), disable=not progress)
    for _ in tqdm_:
        last_idx[:n] = idx[:n]
        boundaries = (midpoints[1:] + midpoints[:-1]) / 2
        torch.bucketize(tensor[:n], boundaries, out=idx[:n])

        if weight is None:
            midpoints.scatter_reduce_(
                0, idx[:n], tensor[:n], "mean", include_self=False
            )
        else:
            # Weighted mean for each midpoint
            midpoints.scatter_reduce_(
                0, idx[:n], tensor[:n] * weight[:n], "sum", include_self=False
            )
            sum_weight.zero_().scatter_reduce_(
                0, idx[:n], weight[:n], "sum", include_self=False
            )
            midpoints.div_(sum_weight.clamp_min_(torch.finfo(dtype).smallest_normal))

        midpoints = torch.cummax(midpoints, 0).values
        idx_change = (last_idx[:n] != idx[:n]).float().mean().item()
        tqdm_.set_postfix_str(f"{idx_change:.1e}")
        if idx_change <= threshold:
            if tensor.nelement() <= n:
                break
            n *= 2
    assert (midpoints[:-1] <= midpoints[1:]).all().item()
    return LUTFormat.create(midpoints, "LM", range=range)


def nf_approx(bits: int) -> LUTFormat:
    return lut_function(
        lambda n: cast(Tensor, (n + n**3) / 2), bits=bits, name="NF-approx"
    )


FP32 = TorchFormat(torch.float32)
FP16 = TorchFormat(torch.float16)
BFLOAT16 = TorchFormat(torch.bfloat16)
# See: QLoRA [https://arxiv.org/abs/2305.14314]
NF4 = LUTFormat.create(
    (
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ),
    "NF",
)
SF4_DF5 = LUTFormat.create(
    (
        -1.000,
        -0.628,
        -0.455,
        -0.334,
        -0.237,
        -0.153,
        -0.075,
        0.000,
        0.066,
        0.133,
        0.205,
        0.284,
        0.376,
        0.491,
        0.657,
        1.000,
    ),
    "SF_DF5",
)


# Cube-root-density optimal formats


def crd_quantiser(
    n: int,
    scaling: Literal["rms", "absmax", "signmax"],
    mode: Literal["symmetric", "repeat_zero", "asymmetric"],
    name: str,
    icdf: Callable[[Tensor, float], Tensor],
    power: float = 1 / 3,
) -> LUTFormat:
    # For cdf in [0.0, 0.5] and [0.5, 1.0], should we include the endpoints?
    # 1 = yes, 0 = no.
    neg_min, neg_max, pos_min, pos_max = {
        ("symmetric", "rms"): (0, 0, 0, 0),
        ("symmetric", "absmax"): (1, 0, 0, 1),
        ("repeat_zero", "rms"): (0, 1, 1, 0),
        ("repeat_zero", "absmax"): (1, 1, 1, 1),
        ("asymmetric", "rms"): (0, 1, 0, 0),
        ("asymmetric", "absmax"): (1, 1, 0, 1),
        ("asymmetric", "signmax"): (0, 1, 0, 1),
    }[(mode, scaling)]
    if not (neg_max or pos_min):
        # Need to special-case this, otherwise we'd have a double-gap around zero
        p = torch.linspace(0, 1, n + 2 - neg_min - pos_max)[1 - neg_min :][:n]
    else:
        halfn = n // 2
        off = 1 - neg_min
        p_neg = torch.linspace(0, 0.5, halfn + 2 - neg_min - neg_max)[off : halfn + off]
        off = 1 - pos_min
        p_pos = torch.linspace(0.5, 1, halfn + 2 - pos_min - pos_max)[off : halfn + off]
        p = torch.cat([p_neg, p_pos])

    if power == 0:
        if scaling == "rms":
            raise ValueError(
                f"Cannot use power=0 with scaling='rms', as the pdf doesn't normalise"
            )
        table = 2 * p - 1
    else:
        table = icdf(p, power)

    scaling_name = dict(rms="R", absmax="A", signmax="S")[scaling]
    mode_name = dict(symmetric="S", repeat_zero="Z", asymmetric="A")[mode]
    if power == 1 / 3:
        power_name = ""
    elif 0 < power < 1:
        power_name = f"{{1/{1/power:.0f}}}"
    else:
        power_name = f"{{{power:.0f}}}"
    return LUTFormat.create(table, f"CRD{power_name}-{name}-{scaling_name}{mode_name}")


def crd_normal(
    bits: float,
    mode: Literal["symmetric", "repeat_zero", "asymmetric"] = "symmetric",
    **args: Any,
) -> LUTFormat:
    """Cube-root-pdf quantisation for Normal-distributed data, rms=1."""
    return crd_quantiser(
        int(round(2**bits)),
        scaling="rms",
        mode=mode,
        name="N",
        icdf=lambda p, power: scipy.stats.norm.ppf(p, scale=power**-0.5),
        **args,
    )


def crd_laplace(
    bits: float,
    mode: Literal["symmetric", "repeat_zero", "asymmetric"] = "symmetric",
    **args: Any,
) -> LUTFormat:
    """Cube-root-pdf quantisation for Laplace-distributed data, rms=1."""
    return crd_quantiser(
        int(round(2**bits)),
        scaling="rms",
        mode=mode,
        name="L",
        icdf=lambda p, power: scipy.stats.laplace.ppf(p, scale=1 / (power * 2**0.5)),
        **args,
    )


def crd_t(
    bits: float,
    df: float,
    mode: Literal["symmetric", "repeat_zero", "asymmetric"] = "symmetric",
    **args: Any,
) -> LUTFormat:
    """Cube-root-pdf quantisation for Student-T-distributed data, rms=1."""

    def icdf(p: Tensor, power: float) -> Tensor:
        cdof = (df + 1 - 1 / power) * power
        cscale = ((df - 2) / cdof) ** 0.5
        return scipy.stats.t.ppf(p, cdof, scale=cscale)

    name_df = f"{df:.0f}" if int(df) == df else f"{df:.1f}"
    return crd_quantiser(
        int(round(2**bits)),
        scaling="rms",
        mode=mode,
        name=f"T[{name_df}]",
        icdf=icdf,
        **args,
    )


def crd_block_normal(
    bits: float,
    block_size: int,
    scaling: Literal["absmax", "signmax"] = "absmax",
    mode: Literal["symmetric", "repeat_zero", "asymmetric"] = "symmetric",
    **args: Any,
) -> LUTFormat:
    """Cube-root-pdf quantisation for (absmax|signmax)-normalised Normal data."""

    def icdf(p: Tensor, power: float) -> Tensor:
        s = power**-0.5 / torch.tensor(block_size).div(torch.pi).log().mul(2).sqrt()
        return scipy.stats.truncnorm.ppf(p, -1 / s, 1 / s, scale=s)

    return crd_quantiser(
        int(round(2**bits)), scaling=scaling, mode=mode, name="N", icdf=icdf, **args
    )


def _trunclaplace_ppf(q: Tensor, a: float, scale: float = 1) -> Tensor:
    e_a = torch.tensor(a).neg().exp()
    return torch.where(
        q < 0.5,
        scale * torch.log(2 * q * (1 - e_a) + e_a),
        -scale * torch.log(2 - e_a - 2 * q * (1 - e_a)),
    )


def crd_block_laplace(
    bits: float,
    block_size: int,
    scaling: Literal["absmax", "signmax"] = "absmax",
    mode: Literal["symmetric", "repeat_zero", "asymmetric"] = "symmetric",
    **args: Any,
) -> LUTFormat:
    """Cube-root-pdf quantisation for (absmax|signmax)-normalised Laplace data."""

    def icdf(p: Tensor, power: float) -> Tensor:
        scale = power**-1 / (0.57721566 + torch.tensor(block_size).log())
        return _trunclaplace_ppf(p, float(1 / scale), scale=scale)

    return crd_quantiser(
        int(round(2**bits)), scaling=scaling, mode=mode, name="L", icdf=icdf, **args
    )


def crd_block_t(
    bits: float,
    block_size: int,
    df: float,
    scaling: Literal["absmax", "signmax"] = "absmax",
    mode: Literal["symmetric", "repeat_zero", "asymmetric"] = "symmetric",
    **args: Any,
) -> LUTFormat:
    """Cube-root-pdf quantisation for (absmax|signmax)-normalised Student-T data."""

    def icdf(p: Tensor, power: float) -> Tensor:
        expected_max = (
            torch.tensor(block_size, dtype=torch.float64)  # .pow(df) is to blame
            .div(torch.pi)
            .log()
            .mul(2)
            .pow((df - 3) / 2)
            .mul(block_size)
            .pow(1 / df)
            .mul((df / (df - 2)) ** 0.5)
        ).item()
        cdof = (df + 1 - 1 / power) * power
        cscale = (df / cdof) ** 0.5
        a0, a1 = scipy.stats.t.cdf([-expected_max, expected_max], cdof, scale=cscale)
        return scipy.stats.t.ppf(a0 + p * (a1 - a0), cdof, scale=cscale) / expected_max

    name_df = f"{df:.0f}" if int(df) == df else f"{df:.1f}"
    return crd_quantiser(
        int(round(2**bits)),
        scaling=scaling,
        mode=mode,
        name=f"T[{name_df}]",
        icdf=icdf,
        **args,
    )


# Tensor formats

BlockShape = Tuple[Optional[int], ...]
Scaling = Literal["absmax", "signmax", "rms"]


def block_normalise(
    tensor: Tensor,
    block_shape: BlockShape,
    scaling: Scaling,
    element_range: tuple[float, float],
    scale_format: TensorFormat,
) -> tuple[Tensor, Tensor]:
    """Normalise the tensor, returning the normalised tensor & scale."""

    if tensor.ndim != len(block_shape):
        raise ValueError(
            f"block_normalise tensor shape {tuple(tensor.shape)}"
            f" must be the same rank as block_shape {block_shape}"
        )

    def _get_scale(block_tensor: Tensor) -> Tensor:
        """Reduce over odd dimensions (1, 3, ...) to get the scale."""
        block_dims = tuple(range(1, block_tensor.ndim, 2))
        eps = torch.finfo(block_tensor.dtype).smallest_normal
        if scaling == "absmax":
            element_absmax = min(-element_range[0], element_range[1])
            return (
                block_tensor.abs()
                .amax(dim=block_dims, keepdim=True)
                .div(element_absmax)
                .clamp_min_(eps)
            )
        if scaling == "signmax":
            element_signmax = (
                element_range[0]
                if -element_range[0] > element_range[1]
                else element_range[1]
            )
            bmin = block_tensor.amin(dim=block_dims, keepdim=True).clamp_max_(-eps)
            bmax = block_tensor.amax(dim=block_dims, keepdim=True).clamp_min_(eps)
            return torch.where(-bmin > bmax, bmin, bmax).div(element_signmax)
        if scaling == "rms":
            # Care is required here when everything in a block is small but non-zero,
            # so that the RMS underflows. We need to clamp_min_ before sqrt() to avoid
            # NaN or exploding values.
            return (
                block_tensor.pow(2)
                .mean(dim=block_dims, keepdim=True, dtype=torch.float32)
                .clamp_min_(torch.finfo(torch.float32).smallest_normal)
                .sqrt()
                .to(block_tensor.dtype)
            )
        assert False, f"unexpected scaling={scaling}"

    blocked_shape = tuple(
        s
        for si, bi in zip(tensor.shape, block_shape)
        for s in ((1, si) if bi is None else (si // bi, bi))
    )
    scale = scale_format.quantise(
        _get_scale(tensor.reshape(blocked_shape))
        .broadcast_to(blocked_shape)
        .reshape(tensor.shape)
    )
    scale = torch.where(scale == 0, 1, scale)  # protect against /0
    return tensor / scale, scale


@dataclass
class LinearScalingFormat(TensorFormat):
    """A block/channel/tensor scaling scheme for tensors.

    block_shape -- size of blocks in each dimension
                   e.g. (1, 8)       input-blocks of size 8
                        (2, 2)       square blocks of 2x2 (4 elements)
                        (1, None)    per-output-channel scaling
                        (None, None) per-tensor scaling

    scaling -- "absmax" - ensure the abs(max(x)) is within range of the `element_format`
               "signmax" - ensure the signed max-abs value is within range (for use with
                           formats with more range to one side of zero; `scale_format`
                           must be signed)
               "rms" - ensure that the RMS of elements is =1 (the user must ensure that
                       `element_format` has a sensible range to represent such values)
    """

    element_format: ScalarFormat
    scale_format: TensorFormat
    block_shape: BlockShape
    scaling: Scaling

    _type: str = "linear"

    def __str__(self) -> str:
        block = ",".join("*" if g is None else str(g) for g in self.block_shape)
        return f"{self.element_format}{{{block}:{self.scale_format}:{self.scaling}}}"

    def _count_scale_bits(self, shape: Shape) -> Shape:
        return self.scale_format.count_bits(
            tuple(
                1 if bi is None else si // bi for si, bi in zip(shape, self.block_shape)
            )
        )

    def count_bits(self, shape: Shape) -> int:
        return self.element_format.count_bits(shape) + self._count_scale_bits(shape)

    def count_bits_tensor(self, tensor: Tensor) -> float:
        scaled_tensor, _ = self.normalise(tensor)
        return self.element_format.count_bits_tensor(
            scaled_tensor
        ) + self._count_scale_bits(tensor.shape)

    def normalise(self, tensor: Tensor) -> tuple[Tensor, Tensor]:
        return block_normalise(
            tensor,
            block_shape=self.block_shape,
            scaling=self.scaling,
            element_range=self.element_format.range,
            scale_format=self.scale_format,
        )

    def quantise(self, tensor: Tensor) -> Tensor:
        scaled_tensor, scale = self.normalise(tensor)
        return self.element_format.quantise(scaled_tensor) * scale


# "Compression" formats


class CompressedTensorFormat(TensorFormat):
    def count_bits_tensor(self, tensor: Tensor) -> float:
        raise NotImplementedError()

    def count_bits(self, shape: Shape) -> int:
        raise NotImplementedError(
            "CompressedTensorFormat `count_bits` depends on the data - use `count_bits_tensor` instead"
        )


Compressor: TypeAlias = Literal["optimal", "bz2", "huffman", "arithmetic"]


@dataclass
class CompressedLUTFormat(CompressedTensorFormat):
    """A lookup table, followed by a lossless compressor."""

    lut: LUTFormat
    model_logp: Tensor
    compressor: Compressor

    def __post_init__(self):
        assert self.model_logp.shape == (len(self.lut.values),)
        assert self.model_logp.exp().sum().sub(1).abs().item() < 1e-4

    def __str__(self) -> str:
        return f"{self.lut}+Z{self.compressor}"

    @property
    def range(self) -> tuple[float, float]:
        return self.lut.range

    def quantise(self, tensor: Tensor) -> Tensor:
        return self.lut.quantise(tensor)

    def count_bits_tensor(self, tensor: Tensor) -> float:
        idx = self.lut.to_idx(tensor)

        if self.compressor == "optimal":
            log2 = torch.tensor(2, device=tensor.device, dtype=tensor.dtype).log()
            return -self.model_logp[idx].sum().div(log2).item()

        if self.compressor == "bz2":
            idx_bytes = idx.to(torch.uint32).cpu().numpy().tobytes()
            return len(bz2.compress(idx_bytes)) * 8

        if self.compressor == "huffman":
            # We don't count the bits to encode the table, since it's considered
            # fixed (derived from `model_logp` not `tensor`).
            import dahuffman

            # Note: use freq = p * large-const, since EOF is added with freq=1
            codec = dahuffman.HuffmanCodec.from_frequencies(
                {i: p.exp().item() * 2**20 for i, p in enumerate(self.model_logp)}
            )
            # Instead of actually encoding the data, which is very slow, encode test
            # sequences to work out the number of bits per item and index into that.
            # This works because huffman compression is stateless between symbols.
            n_samples = 1024
            nbits = torch.tensor(
                [
                    8 * len(codec.encode([i] * n_samples)) / n_samples
                    for i in range(len(self.model_logp))
                ],
                device=idx.device,
            )
            return nbits[idx].sum().item()

        if self.compressor == "arithmetic":
            import arithmetic_compressor

            # Clip the minimum probability to avoid numerical issues (empirical threshold)
            codec = arithmetic_compressor.AECompressor(
                arithmetic_compressor.models.StaticModel(
                    {
                        i: p.exp().clip(min=2e-4).item()
                        for i, p in enumerate(self.model_logp)
                    }
                )
            )
            # We don't count the bits to encode the table, since it's considered
            # fixed (derived from `model_logp` not `tensor`).
            return len(codec.compress(idx.cpu().numpy()))  # list of bits

        raise ValueError(f"Unknown compressor {self.compressor!r}")

    @classmethod
    def train(
        cls,
        lut: LUTFormat,
        data: Tensor,
        smoothing: float = 1.0,
        compressor: Compressor = "optimal",
    ) -> "CompressedLUTFormat":
        counts = (
            lut.to_idx(data.flatten())
            .bincount(minlength=len(lut.values))
            .to(torch.float32)
            .add_(smoothing)
        )
        return cls(
            lut, model_logp=counts.div_(counts.sum()).log_(), compressor=compressor
        )

    @classmethod
    def train_grid(
        cls, data: Tensor, resolution: float, **args: Any
    ) -> "CompressedLUTFormat":
        return cls.train(
            lut_grid(resolution, data.abs().amax().item()), data=data, **args
        )
