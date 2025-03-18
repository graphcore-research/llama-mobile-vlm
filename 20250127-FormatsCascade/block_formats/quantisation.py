# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Utilities for "fake quantisation"."""

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
    Sequence,
    Tuple,
    TypeAlias,
    Union,
    cast,
)

import scipy.stats
import torch
import tqdm
from torch import Tensor, nn

Shape = Tuple[int, ...]

# Utilities


def shuffle(t: Tensor) -> Tensor:
    """Shuffle the flattened tensor, then reassemble."""
    y = torch.empty_like(t.flatten())
    y[torch.randperm(t.nelement(), device=t.device, dtype=torch.int32)] = t.flatten()
    return y.view(t.shape)


def rmse_norm(x: Tensor, qx: Tensor) -> Tensor:
    x = x.float()
    qx = qx.float()
    return ((qx - x).pow(2).sum() / x.pow(2).sum()).sqrt()


def snr(x: Tensor, qx: Tensor) -> Tensor:
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
        assert self.exponent_bits >= 2, "FPFormat requires at least 2 exponent bits"

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
    _type: str = "int"

    def __str__(self) -> str:
        if int(self.bits_) == self.bits_:
            return f"E0M{self.bits_ - 1:.0f}"
        return f"E0M{{{self.bits_ - 1:.2f}}}"

    @property
    def bits(self) -> float:
        return self.bits_

    @property
    def range(self) -> tuple[float, float]:
        n_values = int(round(2.0**self.bits_))
        half_range = (n_values - 1) // 2
        return (-half_range - (2 * half_range + 1 < n_values), half_range)

    def quantise(self, x: Tensor) -> Tensor:
        return torch.clip(torch.round(x), *self.range)


@dataclass
class ExpCeilFormat(ScalarFormat):
    """An exponent-only format for positive numbers, with no zero."""

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
    values: Tuple[float, ...]
    name: str
    _type: str = "lut"

    @classmethod
    def create(cls, values: Tensor, name: str) -> "LUTFormat":
        return cls(values=tuple(values.tolist()), name=name)

    def __post_init__(self) -> None:
        self.values = tuple(self.values)

    def __str__(self) -> str:
        return f"LUT{int(math.ceil(self.bits))}[{self.name}]"

    @property
    def bits(self) -> float:
        return math.log2(len(self.values))

    @property
    def range(self) -> tuple[float, float]:
        return (min(self.values), max(self.values))

    def quantise(self, x: Tensor) -> Tensor:
        # This has slightly worse accuracy if computed in x.dtype, so use float32
        values = torch.tensor(self.values, device=x.device)
        boundaries = (values[1:] + values[:-1]).div(2)
        return values[torch.bucketize(x, boundaries)].to(x.dtype)


@dataclass
class ScaledFormat(ScalarFormat):
    format: ScalarFormat
    scale: float
    _type: str = "scaled"

    def __str__(self) -> str:
        return f"{self.format}{{*{self.scale}}}"

    @property
    def bits(self) -> float:
        return self.format.bits

    @property
    def range(self) -> tuple[float, float]:
        min_, max_ = self.format.range
        return (min_ * self.scale, max_ * self.scale)

    def quantise(self, tensor: Tensor) -> Tensor:
        return self.format.quantise(tensor / self.scale) * self.scale


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


LloydMaxInit: TypeAlias = Union[
    Tensor, tuple[Literal["uniform"], float], Literal["kmeans++"], Literal["cuberoot"]
]


def _lloyd_max_init(init: LloydMaxInit, tensor: Tensor, codepoints: int) -> Tensor:
    if isinstance(init, Tensor):
        assert init.shape == (codepoints,)
        return init.to(tensor.dtype, copy=True)
    if isinstance(init, tuple) and len(init) == 2 and init[0] == "uniform":
        mean, std = tensor.mean(), tensor.std()
        return torch.linspace(
            mean - init[1] * std,
            mean + init[1] * std,
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
        return midpoints
    if init == "cuberoot":
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
    init: LloydMaxInit = "kmeans++",
    incremental: bool = True,
    max_samples: int | None = None,
    dtype: torch.dtype | None = None,
    progress: bool = False,
) -> LUTFormat:
    """Use Lloyd-Max (k-means) to find the RMS-optimal quantiser for the given tensor.

    threshold -- when the ratio of changed cluster assignments <= threshold, stop

    incremental -- start with a subset of the data and scale up
    """
    # Preparation: shuffle, truncate, cast, get init
    tensor = shuffle(tensor.flatten())
    if max_samples is not None:
        tensor = tensor[:max_samples]
    if dtype is None:
        # Very large tensors have stability problems due the float32
        # mantissa length, so default to float64
        dtype = torch.float32 if tensor.nelement() <= 2**26 else torch.float64
    tensor = tensor.to(dtype)
    midpoints = _lloyd_max_init(init, tensor, int(2**bits))

    # K-means iteration
    idx = torch.empty(tensor.shape, device=tensor.device, dtype=torch.int64)
    last_idx = torch.empty_like(idx)
    n = 2**20 if incremental else tensor.nelement()
    tqdm_ = tqdm.tqdm(it.count(), disable=not progress)
    for _ in tqdm_:
        last_idx[:n] = idx[:n]
        boundaries = (midpoints[1:] + midpoints[:-1]) / 2
        torch.bucketize(tensor[:n], boundaries, out=idx[:n])
        midpoints.scatter_reduce_(0, idx[:n], tensor[:n], "mean", include_self=False)
        midpoints = torch.cummax(midpoints, 0).values
        idx_change = (last_idx[:n] != idx[:n]).float().mean().item()
        tqdm_.set_postfix_str(f"{idx_change:.1e}")
        if idx_change <= threshold:
            if tensor.nelement() <= n:
                break
            n *= 2
    assert (midpoints[:-1] <= midpoints[1:]).all().item()
    return LUTFormat.create(midpoints, "LM")


def nf_approx(bits: int) -> LUTFormat:
    return lut_function(
        lambda n: cast(Tensor, (n + n**3) / 2), bits=bits, name="NF-approx"
    )


FP32 = TorchFormat(torch.float32)
FP16 = TorchFormat(torch.float16)
BFLOAT16 = TorchFormat(torch.bfloat16)
# See: QLoRA [https://arxiv.org/abs/2305.14314]
NF4 = LUTFormat(
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

    table = tuple(icdf(p, power).tolist())
    scaling_name = dict(rms="R", absmax="A", signmax="S")[scaling]
    mode_name = dict(symmetric="S", repeat_zero="Z", asymmetric="A")[mode]
    if power == 1 / 3:
        power_name = ""
    elif power < 1:
        power_name = f"{{1/{1/power:.0f}}}"
    else:
        power_name = f"{{{power:.0f}}}"
    return LUTFormat(table, f"CRD-{name}-{scaling_name}{mode_name}{power_name}")


def crd_normal(
    bits: float,
    mode: Literal["symmetric", "repeat_zero", "asymmetric"] = "symmetric",
    **args: Any,
) -> LUTFormat:
    """Cube-root-pdf quantisation for Normal-distributed data, rms=1."""
    return crd_quantiser(
        int(2**bits),
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
        int(2**bits),
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

    return crd_quantiser(
        int(2**bits), scaling="rms", mode=mode, name="T", icdf=icdf, **args
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
        int(2**bits), scaling=scaling, mode=mode, name="N", icdf=icdf, **args
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
        int(2**bits), scaling=scaling, mode=mode, name="L", icdf=icdf, **args
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
            torch.tensor(block_size)
            .div(torch.pi)
            .log()
            .mul(2)
            .pow((df - 3) / 2)
            .mul(block_size)
            .pow(1 / df)
            .mul((df / (df - 2)) ** 0.5)
        )
        cdof = (df + 1 - 1 / power) * power
        cscale = (df / cdof) ** 0.5
        a0, a1 = scipy.stats.t.cdf([-expected_max, expected_max], cdof, scale=cscale)
        return scipy.stats.t.ppf(a0 + p * (a1 - a0), cdof, scale=cscale) / expected_max

    return crd_quantiser(
        int(2**bits), scaling=scaling, mode=mode, name="T", icdf=icdf, **args
    )


# Tensor formats (new)


@dataclass
class LinearScalingFormat(TensorFormat):
    """A group/channel/tensor scaling scheme for tensors.

    group_shape -- size of groups in each dimension
                   e.g. (1, 8)       input-groups of size 8
                        (2, 2)       square groups of 2x2 (4 elements)
                        (1, None)    per-output-channel scaling
                        (None, None) per-tensor scaling

    scaling -- "absmax" - ensure the abs(max(x)) is within range of the `element_format`
               "signmax" - ensure the signed max-abs value is within range (for use with
                           formats with more range to one side of zero; `scale_format`
                           must be signed)
               "rms" - ensure that the RMS of elements is =1 (the user must ensure that
                       `element_format` has a sensible range to represent such values)
    """

    GroupShape = Tuple[Optional[int], ...]

    element_format: ScalarFormat
    scale_format: TensorFormat
    group_shape: GroupShape
    scaling: Literal["absmax", "signmax", "rms"]

    _type: str = "linear"

    def __str__(self) -> str:
        group = ",".join("*" if g is None else str(g) for g in self.group_shape)
        return f"{self.element_format}{{{group}:{self.scale_format}:{self.scaling}}}"

    @staticmethod
    def _group_shape_for(tensor_shape: Shape, group_shape: GroupShape) -> Shape:
        assert len(tensor_shape) == len(group_shape), f"{tensor_shape} vs {group_shape}"
        return tuple((t if g is None else g) for t, g in zip(tensor_shape, group_shape))

    def count_bits(self, shape: Shape) -> int:
        element_bits = self.element_format.count_bits(shape)
        scale_bits = self.scale_format.count_bits(
            tuple(
                t // g
                for t, g in zip(shape, self._group_shape_for(shape, self.group_shape))
            )
        )
        return element_bits + scale_bits

    def _get_scale(self, grouped_tensor: Tensor) -> Tensor:
        """Reduce over odd dimensions (1, 3, ...) to get the scale."""
        group_dims = tuple(range(1, grouped_tensor.ndim, 2))
        if self.scaling == "absmax":
            element_min, element_max = self.element_format.range
            absmax = min(-element_min, element_max)
            return grouped_tensor.abs().div(absmax).amax(dim=group_dims, keepdim=True)
        if self.scaling == "signmax":
            element_min, element_max = self.element_format.range
            signmax = element_min if -element_min > element_max else element_max
            group_min = grouped_tensor.amin(dim=group_dims, keepdim=True)
            group_max = grouped_tensor.amax(dim=group_dims, keepdim=True)
            return torch.where(-group_min > group_max, group_min, group_max).div(
                signmax
            )
        if self.scaling == "rms":
            return grouped_tensor.pow(2).mean(dim=group_dims, keepdim=True).sqrt()
        assert False, f"unexpected scaling={self.scaling}"

    def scale_for(self, tensor: Tensor) -> Tensor:
        """Get the quantised scaling tensor to apply to quantise a given tensor."""
        group_shape = self._group_shape_for(tensor.shape, self.group_shape)
        full_grouped_shape = tuple(
            s
            for size, group_size in zip(tensor.shape, group_shape)
            for s in [size // group_size, group_size]
        )
        scale = (
            self._get_scale(tensor.reshape(full_grouped_shape))
            .broadcast_to(full_grouped_shape)
            .reshape(tensor.shape)
        )
        return self.scale_format.quantise(scale)

    def quantise(self, tensor: Tensor) -> Tensor:
        scale = self.scale_for(tensor)
        return self.element_format.quantise(tensor / scale) * scale


@dataclass
class ChannelAndSparseFormat(TensorFormat):
    """A scheme where input/output channels are scaled & outliers stored separately."""

    element_format: ScalarFormat
    scale_format: TensorFormat
    scale_dim: Optional[int]
    sparse_format: ScalarFormat
    sparse_ratio: float

    _type: str = "channel_and_sparse"

    def __str__(self) -> str:
        return (
            f"{self.element_format}{{dim={self.scale_dim}:{self.scale_format}}}"
            f"+{self.sparse_format}{{{self.sparse_ratio:.1%}}}"
        )

    def n_sparse(self, shape: Shape) -> int:
        return int(self.sparse_ratio * math.prod(shape))

    def count_bits(self, shape: Shape) -> int:
        element_bits = self.element_format.count_bits(shape)
        scale_bits = self.scale_format.count_bits(
            () if self.scale_dim is None else (shape[self.scale_dim],)
        )
        n_sparse = self.n_sparse(shape)
        sparse_value_bits = self.sparse_format.count_bits((n_sparse,))
        sparse_mask_bits = 32 * n_sparse  # flat-COO format
        return element_bits + scale_bits + sparse_value_bits + sparse_mask_bits

    def quantise(self, tensor: Tensor) -> Tensor:
        n_sparse = self.n_sparse(tensor.shape)

        # Find outliers to represent with sparity
        reduce_dims = tuple(d for d in range(tensor.ndim) if d != self.scale_dim)
        rms_ratio = (
            tensor.div(tensor.pow(2).mean(dim=reduce_dims, keepdim=True).sqrt())
            .abs_()
            .flatten()
        )
        sparse_idx = torch.where(
            rms_ratio >= rms_ratio.neg().kthvalue(n_sparse - 1).values.neg()
        )[0][:n_sparse]

        # Perform channel quantisation, except for sparse values
        qtensor = tensor.flatten().clone()
        qtensor[sparse_idx] = 0
        scale = self.scale_format.quantise(
            qtensor.reshape(tensor.shape)
            .pow(2)
            .mean(dim=reduce_dims, keepdim=True)
            .sqrt()
        ).flatten()
        qtensor = self.element_format.quantise(qtensor / scale) * scale
        qtensor[sparse_idx] = self.sparse_format.quantise(tensor.flatten()[sparse_idx])
        return qtensor.reshape(tensor.shape)


# Model parameters


def _match_shape(shape: Shape, pattern: Tuple[Optional[int], ...]) -> bool:
    return (len(shape) == len(pattern)) and all(
        p is None or p == s for s, p in zip(shape, pattern)
    )


@dataclass
class ParameterRule:
    pattern: Optional[str]
    shape: Optional[Tuple[Optional[int], ...]]
    format: TensorFormat

    def match(self, name: str, parameter: Tensor) -> bool:
        if self.pattern is not None and not re.search(self.pattern, name):
            return False
        if self.shape is not None and not _match_shape(parameter.shape, self.shape):
            return False
        return True


def quantise_model(model: nn.Module, rules: Iterable[ParameterRule] = []) -> float:
    """In-place quantise a model, returning the size of the quantised model (bytes).

    rules -- an ordered list or rules. For the first rule which matches `pattern` and
             `shape` (if specified), use the given quantisation format.
             If no rule matches, use a `TorchFormat` with the existing parameter type
             (no quantisation).
    """
    bitcount = 0
    for name, p in model.named_parameters():
        for rule in rules:
            if rule.match(name, p):
                format_ = rule.format
                break
        else:
            format_ = TorchFormat(p.dtype)
        assert not hasattr(p, "quantisation_format"), "double-quantisation not allowed"
        p.data = format_.quantise(p.data)
        p.quantisation_format = format_  # type:ignore[attr-defined]
        bitcount += format_.count_bits(p.data.shape)
    return bitcount / 8
