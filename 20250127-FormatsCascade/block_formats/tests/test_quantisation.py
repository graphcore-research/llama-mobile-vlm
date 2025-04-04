import torch
from torch import tensor

from .. import quantisation as Q


def test_linear_scaling_format() -> None:
    fmt = Q.LinearScalingFormat(Q.IntFormat(2), Q.FP32, (3,), "absmax")
    assert "absmax" in str(fmt)
    assert fmt.count_bits((12,)) == 2 * 12 + 4 * 32
    torch.testing.assert_close(
        fmt.quantise(tensor([100, -60, 40, -40, 10, 21])),
        tensor([100.0, -100, 0, -40, 0, 40]),
    )

    fmt = Q.LinearScalingFormat(Q.IntFormat(2), Q.FP32, (2,), "signmax")
    torch.testing.assert_close(
        fmt.quantise(tensor([-20, 9, 40, -15])),
        tensor([-20.0, 10, 40, -20]),
    )
