# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

"""Analysis code to support the notebook"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from . import quantisation as Q


def rms(tensor: Tensor) -> Tensor:
    return tensor.pow(2).mean().sqrt()


def rms_normalise(tensor: Tensor) -> Tensor:
    """Divide a tensor by its RMS."""
    return tensor / rms(tensor)


def block_normalise(tensor: Tensor, block_size: int) -> Tensor:
    """Divide a tensor by its block-absmax (on the last dimension)."""
    return (
        tensor.view(-1, block_size)
        .div(tensor.view(-1, block_size).abs().amax(-1, keepdim=True))
        .view(tensor.shape)
    )


def qrmse_norm(fmt: Q.TensorFormat, tensor: Tensor) -> Tensor:
    """RMS error of quantisation, normalised by original tensor RMS."""
    return Q.rmse_norm(tensor, fmt.quantise(tensor))


@dataclass
class Distribution:
    def torch_distribution(
        self, device: torch.device
    ) -> torch.distributions.Distribution:
        cls = getattr(torch.distributions, type(self).__name__)
        cls_args = {
            k: torch.tensor(v, dtype=torch.float32, device=device)
            for k, v in dict(loc=0, **self.__dict__).items()
        }
        return cls(**cls_args)

    def sample(self, n: int, *, seed: int, device: torch.device) -> Tensor:
        torch.manual_seed(int(np.random.SeedSequence(seed).generate_state(1)[0]))
        return self.torch_distribution(device).sample((n,))

    def rms_quantiser(
        self,
        bits: float,
        mode: Literal["symmetric", "repeat_zero", "asymmetric"] = "symmetric",
        **args: Any
    ):
        d = dict(self.__dict__)
        d.pop("scale")
        return dict(
            Normal=Q.crd_normal,
            Laplace=Q.crd_laplace,
            StudentT=Q.crd_t,
        )[
            type(self).__name__
        ](bits, mode=mode, **d, **args)

    def absmax_quantiser(
        self,
        bits: float,
        block_size: int,
        scaling: Literal["absmax", "signmax"] = "absmax",
        mode: Literal["symmetric", "repeat_zero", "asymmetric"] = "symmetric",
        **args: Any
    ):
        d = dict(self.__dict__)
        d.pop("scale")
        return dict(
            Normal=Q.crd_block_normal,
            Laplace=Q.crd_block_laplace,
            StudentT=Q.crd_block_t,
        )[type(self).__name__](
            bits, block_size, scaling=scaling, mode=mode, **d, **args
        )


@dataclass
class Normal(Distribution):
    scale: float = 1.0


@dataclass
class Laplace(Distribution):
    scale: float = 1.0


@dataclass
class StudentT(Distribution):
    df: float
    scale: float = 1.0
