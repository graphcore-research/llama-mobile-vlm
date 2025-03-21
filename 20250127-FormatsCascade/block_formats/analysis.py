# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

"""Analysis code to support the notebook"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import scipy.optimize
import torch
from torch import Tensor

from . import quantisation as Q


def rms(tensor: Tensor) -> Tensor:
    return tensor.pow(2).mean().sqrt()


def rms_normalise(tensor: Tensor) -> Tensor:
    """Divide a tensor by its RMS."""
    return tensor / rms(tensor)


def block_normalise(
    tensor: Tensor, scaling: Literal["absmax", "signmax", "rms"], block_size: int | None
) -> Tensor:
    """Divide a tensor by its block-absmax (on the last dimension)."""
    t = tensor.view(-1, block_size) if block_size is not None else tensor.view(-1)
    t = t.float()
    norm = dict(
        absmax=lambda: t.abs().amax(-1),
        signmax=lambda: torch.where(-t.amin(-1) > t.amax(-1), t.amin(-1), t.amax(-1)),
        rms=lambda: t.pow(2).mean(-1).sqrt(),
    )[scaling]()
    return t.div(norm[..., None]).to(tensor.dtype).view(tensor.shape)


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

    def quantiser(
        self,
        bits: float,
        scaling: Literal["rms", "absmax", "signmax"] = "rms",
        block_size: int | None = None,
        mode: Literal["symmetric", "repeat_zero", "asymmetric"] = "symmetric",
        **args: Any
    ):
        d = dict(self.__dict__)
        d.pop("scale")
        if scaling == "rms":
            assert block_size is None, "rms scaling doesn't support block size"
            return dict(
                Normal=Q.crd_normal,
                Laplace=Q.crd_laplace,
                StudentT=Q.crd_t,
            )[
                type(self).__name__
            ](bits, mode=mode, **d, **args)
        return dict(
            Normal=Q.crd_block_normal,
            Laplace=Q.crd_block_laplace,
            StudentT=Q.crd_block_t,
        )[type(self).__name__](
            bits, block_size, scaling=scaling, mode=mode, **d, **args
        )

    def find_compressed_quantiser(
        self,
        bits: int,
        X: Tensor,
        X_train: Tensor,
        scaling: Literal["rms", "absmax", "signmax"] = "rms",
        block_size: int | None = None,
        power: float = 0,
    ) -> Q.CompressedLUTFormat:
        def _format(b0: float) -> Q.CompressedLUTFormat:
            if power == 0 and scaling == "rms":
                amax = X_train.abs().amax()
                fmt = Q.LUTFormat.create(
                    torch.linspace(-amax, amax, int(2**b0)), "GRID"
                )
            else:
                fmt = self.quantiser(
                    b0, power=power, scaling=scaling, block_size=block_size
                )
            return Q.CompressedLUTFormat.train(fmt, X_train)

        opt = scipy.optimize.minimize_scalar(
            lambda b0: (_format(b0).count_bits_tensor(X) / X.nelement() - bits) ** 2,
            bounds=(bits, bits + 8),
        )
        return _format(opt.x)


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
