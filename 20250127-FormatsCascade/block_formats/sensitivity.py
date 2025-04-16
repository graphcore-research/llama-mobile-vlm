# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

from typing import Any, Optional

import torch
from torch import Tensor, nn

# Legacy


class _LinearWithGradSq(torch.autograd.Function):
    @staticmethod
    def forward(input: Tensor, weight: Tensor) -> Tensor:
        return input @ weight.T

    @staticmethod
    def backward(  # type:ignore[override]
        ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor
    ) -> tuple[Optional[Tensor], Tensor]:
        input, weight = ctx.saved_tensors  # type:ignore[attr-defined]
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
        ctx.save_for_backward(
            input, weight if input.requires_grad else None  # type:ignore[arg-type]
        )


class LinearGradSqWrapper(nn.Module):
    """A linear layer with no bias, that stores sum(grad**2) in `weight.grad`."""

    def __init__(self, wrapped: nn.Linear):
        super().__init__()
        assert wrapped.bias is None
        assert not isinstance(wrapped, type(self))
        self.wrapped = wrapped

    def forward(self, input: Tensor) -> Tensor:
        return _LinearWithGradSq.apply(input, self.wrapped.weight)

    @classmethod
    def wrap(cls, model: nn.Module) -> None:
        for m in model.modules():
            if not isinstance(m, cls):
                for name, child in m.named_children():
                    if isinstance(child, nn.Linear):
                        setattr(m, name, cls(child))

    @classmethod
    def unwrap(cls, model: nn.Module) -> None:
        for m in model.modules():
            for name, child in m.named_children():
                if isinstance(child, cls):
                    setattr(m, name, child.wrapped)


# New


class Wrapper(nn.Module):
    def _accumulate(self, name: str, value: Tensor) -> None:
        if (v := getattr(self, name)) is not None:
            v += value
        else:
            setattr(self, name, value.to(self.dtype))


class LinearWrapper(Wrapper):
    """Wraps a linear layer with no bias, to calculate sum(grad_weight**2), sum(input**2), sum(grad_output**2)."""

    def __init__(self, wrapped: nn.Linear, dtype: torch.dtype | None):
        super().__init__()
        assert wrapped.bias is None
        self.wrapped = wrapped
        self.dtype = dtype
        self.input_sq = self.grad_output_sq = self.grad_weight_sq = None

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
        dtype = self.dtype or input.dtype

        self._accumulate("input_sq", input_sq.sum(0).to(dtype))
        self._accumulate("grad_output_sq", grad_output_sq.sum(0).to(dtype))
        self._accumulate("grad_weight_sq", grad_weight_sq.to(dtype))


class EmbeddingWrapper(Wrapper):
    def __init__(self, wrapped: nn.Embedding, dtype: torch.dtype | None):
        super().__init__()
        assert wrapped.padding_idx is None
        self.wrapped = wrapped
        self.dtype = dtype
        self.input_sq = self.grad_output_sq = self.grad_weight_sq = None

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
        grad_output_sq = grad_output.flatten(end_dim=-2).float().square()
        dtype = self.dtype or grad_output.dtype

        self._accumulate("input_sq", input_sq.to(dtype))
        self._accumulate("grad_output_sq", grad_output_sq.sum(0).to(dtype))
        # Accumulate manually, to avoid a memory spike
        if self.grad_weight_sq is None:
            self.grad_weight_sq = torch.zeros(
                self.wrapped.weight.shape, device=input.device, dtype=dtype
            )
        self.grad_weight_sq.scatter_add_(
            0,
            input.flatten()[:, None].expand(
                (input.nelement(), self.wrapped.embedding_dim)
            ),
            grad_output_sq.to(self.grad_weight_sq),
        )


def wrap(model: nn.Module, dtype: torch.dtype | None = None) -> None:
    for m in model.modules():
        if not isinstance(m, Wrapper):
            for name, child in m.named_children():
                if isinstance(child, nn.Linear):
                    setattr(m, name, LinearWrapper(child, dtype=dtype))
                if isinstance(child, nn.Embedding):
                    setattr(m, name, EmbeddingWrapper(child, dtype=dtype))


def unwrap(model: nn.Module) -> None:
    for m in model.modules():
        for name, child in m.named_children():
            if isinstance(child, Wrapper):
                setattr(m, name, child.wrapped)
