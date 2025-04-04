import torch
from torch import nn

from .. import sensitivity


def test_linear_grad_sq_wrapper() -> None:
    layer = nn.Linear(2, 2, bias=False)
    x = torch.ones(4, 2)
    grady = torch.arange(8).remainder(2).mul(2).sub(1).view(2, 4).T

    layer.zero_grad()
    layer(x).backward(grady)
    torch.testing.assert_close(layer.weight.grad, torch.zeros(2, 2))

    layer.zero_grad()
    sensitivity.LinearGradSqWrapper(layer)(x).backward(grady)
    torch.testing.assert_close(layer.weight.grad, torch.full((2, 2), 4.0))
