# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import dataclasses
import json

import torch
from torch import tensor

from .. import quantisation as Q


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
        fmt.quantise(tensor([100, -60, 40, -40, 10, 21])),
        tensor([100.0, -100, 0, -40, 0, 40]),
    )
    assert json.loads(json.dumps(dataclasses.asdict(fmt)))["block_shape"] == [3]

    fmt = Q.LinearScalingFormat(Q.IntFormat(2), Q.FP32, (2,), "signmax")
    torch.testing.assert_close(
        fmt.quantise(tensor([-20, 9, 40, -15])),
        tensor([-20.0, 10, 40, -20]),
    )
