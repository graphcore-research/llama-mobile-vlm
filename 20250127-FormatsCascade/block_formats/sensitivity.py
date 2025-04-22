# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import torch
from torch import Tensor, nn


class TwoStageAccumulator:
    """An on-device low-precision accumulator and on-CPU float32 accumulator."""

    def __init__(self, headroom_bits: float = 2):
        self.headroom_bits = headroom_bits
        self.reset()

    def reset(self) -> None:
        self.sum0, self.count0 = None, 0
        self.sum1, self.count1 = None, 0

    def accumulate(self, value: Tensor) -> None:
        if self.sum0 is None:
            self.sum0 = value
        else:
            self.sum0 += value
        self.count0 += 1

        if self.count0 >= 2 / torch.finfo(torch.bfloat16).eps / 2**self.headroom_bits:
            self.count1 += self.count0
            self.count0 = 0
            if self.sum1 is None:
                self.sum1 = self.sum0.to(torch.device("cpu"), torch.float32)
            else:
                self.sum1 += self.sum0.to(self.sum1)
            self.sum0 = None

    def sum(self) -> Tensor:
        if self.sum1 is not None:
            if self.sum0 is not None:
                return self.sum1 + self.sum0.to(self.sum1)
            return self.sum1
        return self.sum0.to(torch.device("cpu"), torch.float32)


class Wrapper(nn.Module):
    pass


class LinearWrapper(Wrapper):
    """Wraps a linear layer, to calculate sum(grad_weight**2), sum(input**2), sum(grad_output**2)."""

    def __init__(self, wrapped: nn.Linear):
        super().__init__()
        self.wrapped = wrapped
        self.input_sq = TwoStageAccumulator()
        self.grad_output_sq = TwoStageAccumulator()
        self.grad_weight_sq = TwoStageAccumulator()

    def forward(self, input: Tensor) -> Tensor:
        y = nn.functional.linear(input, self.wrapped.weight, self.wrapped.bias)
        y.requires_grad_(True).register_hook(
            lambda grad_output: self._ongrad(input.detach(), grad_output.detach())
        )
        return y

    def _ongrad(self, input: Tensor, grad_output: Tensor) -> None:
        input_sq = input.flatten(end_dim=-2).float().square()
        grad_output_sq = grad_output.flatten(end_dim=-2).float().square()
        grad_weight_sq = grad_output_sq.T @ input_sq

        self.input_sq.accumulate(input_sq.sum(0).to(input.dtype))
        self.grad_output_sq.accumulate(grad_output_sq.sum(0).to(input.dtype))
        self.grad_weight_sq.accumulate(grad_weight_sq.to(input.dtype))


class EmbeddingWrapper(Wrapper):
    def __init__(self, wrapped: nn.Embedding):
        super().__init__()
        # We can ignore `padding_idx`, as it has the same behaviour as any other index
        # in the forward pass, so we can calculate sensitivity in the same way
        self.wrapped = wrapped
        self.input_sq = TwoStageAccumulator()
        self.grad_output_sq = TwoStageAccumulator()
        self.grad_weight_sq = TwoStageAccumulator()

    def forward(self, input: Tensor) -> Tensor:
        y = nn.functional.embedding(
            input,
            self.wrapped.weight,
            self.wrapped.padding_idx,
            self.wrapped.max_norm,
            self.wrapped.norm_type,
            self.wrapped.scale_grad_by_freq,
        )
        y.requires_grad_(True).register_hook(
            lambda grad_output: self._ongrad(input.detach(), grad_output.detach())
        )
        return y

    def _ongrad(self, input: Tensor, grad_output: Tensor) -> None:
        input_sq = torch.bincount(
            input.flatten(), minlength=self.wrapped.num_embeddings
        )
        grad_output_sq = grad_output.flatten(end_dim=-2).square()

        self.input_sq.accumulate(input_sq.to(grad_output.dtype))
        self.grad_output_sq.accumulate(
            grad_output_sq.float().sum(0).to(grad_output.dtype)
        )
        self.grad_weight_sq.accumulate(
            torch.zeros_like(self.wrapped.weight).scatter_add_(
                0,
                input.flatten()[:, None].expand(
                    (input.nelement(), self.wrapped.embedding_dim)
                ),
                grad_output_sq,
            )
        )


def wrap(model: nn.Module) -> None:
    for m in model.modules():
        if not isinstance(m, Wrapper):
            for name, child in m.named_children():
                if isinstance(child, nn.Linear):
                    setattr(m, name, LinearWrapper(child))
                if isinstance(child, nn.Embedding):
                    setattr(m, name, EmbeddingWrapper(child))


def unwrap(model: nn.Module) -> None:
    for m in model.modules():
        for name, child in m.named_children():
            if isinstance(child, Wrapper):
                setattr(m, name, child.wrapped)
