# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import dataclasses
import json

import torch
from math import log2
from torch import tensor

from .. import quantisation as Q

# Formats


def test_int_format() -> None:
    x = torch.linspace(-8, 8, 1001)

    fmt = Q.IntFormat(log2(8))
    assert str(fmt) == "E0M2"
    assert fmt.range == (-4, 3)
    assert set(fmt.quantise(x).tolist()) == set(range(-4, 3 + 1))

    fmt = Q.IntFormat(log2(9))
    assert fmt.range == (-4, 4)
    assert set(fmt.quantise(x).tolist()) == set(range(-4, 4 + 1))

    fmt = Q.IntFormat(log2(8), mode="symmetric")
    assert str(fmt) == "E0M2-S"
    assert fmt.range == (-3.5, 3.5)
    assert sorted(set(fmt.quantise(x).tolist())) == [
        -3.5,
        -2.5,
        -1.5,
        -0.5,
        0.5,
        1.5,
        2.5,
        3.5,
    ]

    fmt = Q.IntFormat(log2(9), mode="symmetric")  # already symmetric
    assert fmt.range == (-4, 4)
    assert set(fmt.quantise(x).tolist()) == set(range(-4, 4 + 1))


# Wrappers


def test_random_rotation_format() -> None:
    # A random rotation on the heavy-tailed laplace should reduce RMSE
    torch.manual_seed(100)
    x = torch.distributions.Laplace(0, 1).sample((2**10, 2**12))
    fmt = Q.RandomRotationFormat(Q.parse("E2M1"), (0,), 100)
    rmse_rotated = Q.qrmse_norm(fmt, x).item()
    rmse_original = Q.qrmse_norm(fmt.format, x).item()
    assert rmse_rotated < 0.9 * rmse_original
    assert fmt.count_bits((100,)) == 400


def test_sparse_format() -> None:
    x = torch.tensor([1, 2, -1, -1000, 0, 0, 0, 900]).float().view(2, -1)
    fmt = Q.SparseFormat(Q.parse("E2M1"), Q.FP32, 1 / 4)
    assert Q.qrmse_norm(fmt, x).item() == 0
    assert Q.qrmse_norm(fmt.format, x).item() > 0.1
    assert fmt.count_bits((2, 4)) == 8 * 4 + (8 / 4) * (32 + 32)


def test_block_normalise() -> None:
    # Check the case where the normalisation axis is all-zero
    xs = torch.arange(3, dtype=torch.float32)[:, None].broadcast_to((3, 4))
    for scaling in ["absmax", "rms"]:
        torch.testing.assert_close(
            Q.block_normalise(xs, (1, None), scaling, (-1, 1), Q.FP32)[0],
            torch.tensor([0.0, 1.0, 1.0])[:, None].broadcast_to((3, 4)),
            msg=f"scaling={scaling}",
        )


def test_linear_scaling_format() -> None:
    fmt = Q.LinearScalingFormat(Q.IntFormat(2), Q.FP32, (3,), "absmax")
    assert "absmax" in str(fmt)
    assert fmt.count_bits((12,)) == 2 * 12 + 4 * 32
    torch.testing.assert_close(
        fmt.quantise(tensor([100.0, -60, 40, -40, 10, 21])),
        tensor([100.0, -100, 0, -40, 0, 40]),
    )
    assert json.loads(json.dumps(dataclasses.asdict(fmt)))["block_shape"] == [3]

    fmt = Q.LinearScalingFormat(Q.IntFormat(2), Q.FP32, (2,), "signmax")
    torch.testing.assert_close(
        fmt.quantise(tensor([-20.0, 9, 40, -15])),
        tensor([-20.0, 10, 40, -20]),
    )
