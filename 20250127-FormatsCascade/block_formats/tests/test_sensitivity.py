# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import torch
from torch import nn

from .. import sensitivity as S


def test_two_stage_accumulator() -> None:
    for n in [40, 64, 600]:  # 64 = 256/4, for spill to 2nd stage
        a = S.TwoStageAccumulator()
        for _ in range(n):
            a.accumulate(torch.tensor(1.0, dtype=torch.bfloat16))
        torch.testing.assert_close(a.sum(), torch.tensor(float(n)))

    # Baseline - doesn't work!
    flat = torch.tensor(0.0, dtype=torch.bfloat16)
    for _ in range(600):
        flat += 1
    torch.testing.assert_close(flat.float(), torch.tensor(256.0))


def test_linear_wrapper() -> None:
    model = nn.Sequential(nn.Linear(2, 3, bias=False))
    S.wrap(model)

    x = torch.full((4, 2), 0.123)
    grady = torch.arange(3)[None, :].expand(4, 3).float()
    for _ in range(10):
        model(x).backward(grady)

    assert not model[0].grad_weight_sq.sum0.requires_grad
    torch.testing.assert_close(
        model[0].input_sq.sum(), torch.full((2,), 0.123**2 * 4 * 10)
    )
    torch.testing.assert_close(
        model[0].grad_output_sq.sum(), torch.arange(3).float() ** 2 * 4 * 10
    )
    torch.testing.assert_close(
        model[0].grad_weight_sq.sum(),
        (torch.arange(3).float()[:, None] * torch.full((2,), 0.123)) ** 2 * 4 * 10,
    )

    S.unwrap(model)
    assert isinstance(model[0], nn.Linear)


def test_embedding_wrapper() -> None:
    model = nn.Sequential(nn.Embedding(4, 2))
    S.wrap(model)

    x = torch.tensor([0, 2, 3, 2])
    grady = torch.full((4, 2), 7)
    for _ in range(10):
        model(x).backward(grady)

    torch.testing.assert_close(
        model[0].input_sq.sum(), 10.0 * torch.tensor([1, 0, 2, 1])
    )
    torch.testing.assert_close(
        model[0].grad_output_sq.sum(), torch.full((2,), 49.0 * 40)
    )
    torch.testing.assert_close(
        model[0].grad_weight_sq.sum(),
        torch.tensor([1, 0, 2, 1])[:, None].expand(4, 2) * 49.0 * 10,
    )

    S.unwrap(model)
    assert isinstance(model[0], nn.Embedding)
