# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

from typing import Optional

import torch
from torch import Tensor, nn


class _LinearWithGradSq(torch.autograd.Function):
    @staticmethod
    def forward(input: Tensor, weight: Tensor) -> Tensor:
        return input @ weight.T

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor
    ) -> tuple[Optional[Tensor], Tensor]:
        input, weight = ctx.saved_tensors
        grad_output_flat = grad_output.flatten(end_dim=-2).float()
        input_flat = input.flatten(end_dim=-2).float()
        grad_sq_weight = grad_output_flat.T.square() @ input_flat.square()
        return (None if weight is None else grad_output @ weight, grad_sq_weight)

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: tuple[Tensor, Tensor],
        output: Tensor,
    ) -> None:
        input, weight = inputs
        ctx.save_for_backward(input, weight if input.requires_grad else None)


class LinearGradSqWrapper(nn.Module):
    """A linear layer with no bias, that stores sum(grad**2) in `weight.grad`."""

    def __init__(self, wrapped: nn.Linear):
        super().__init__()
        assert not wrapped.bias
        assert not isinstance(wrapped, type(self))
        self.wrapped = wrapped

    def forward(self, input: Tensor) -> Tensor:
        return _LinearWithGradSq.apply(input, self.wrapped.weight)

    @classmethod
    def wrap(cls, model: nn.Module):
        for m in model.modules():
            if not isinstance(m, cls):
                for name, child in m.named_children():
                    if isinstance(child, nn.Linear):
                        setattr(m, name, cls(child))

    @classmethod
    def unwrap(cls, model: nn.Module):
        for m in model.modules():
            for name, child in m.named_children():
                if isinstance(child, cls):
                    setattr(m, name, child.wrapped)
