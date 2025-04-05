import torch

from .. import fit as F
from .. import quantisation as Q


def test_scaled_quantiser() -> None:
    torch.manual_seed(100)
    x = 3 * torch.randn(2**16)
    tol = 0.025

    # Compressed (RMS)
    fmt = F.scaled_quantiser(
        x, 4, "int", Q.BFLOAT16, (None,), "rms", compressor="optimal"
    )
    assert 4 - tol < fmt.count_bits_tensor(x) / x.nelement() < 4 + tol
    assert Q.qrmse_norm(fmt, x).item() < 0.08  # empirical

    # Compressed (block)
    fmt = F.scaled_quantiser(
        x, 4, "int", Q.BFLOAT16, (32,), "absmax", compressor="optimal"
    )
    assert 4.5 - tol < fmt.count_bits_tensor(x) / x.nelement() < 4.5 + tol
    assert Q.qrmse_norm(fmt, x).item() < 0.08  # empirical

    # Lloyd-Max
    fmt = F.scaled_quantiser(
        x, 4, "lloyd_max", Q.BFLOAT16, (32,), "signmax", compressor=None
    )
    assert fmt.count_bits_tensor(x) / x.nelement() == 4.5
    assert Q.qrmse_norm(fmt, x).item() < 0.081  # empirical

    # Sweep
    for element_family in ["int", "fp", "normal", "laplace", "t"]:
        for block_size, scaling in [(None, "rms"), (16, "rms"), (32, "absmax")]:
            fmt = F.scaled_quantiser(
                x,
                4,
                element_family,
                Q.BFLOAT16,
                (block_size,),
                scaling,
                compressor=None,
            )
            expected_b = 4 + 16 / (block_size or x.nelement())
            actual_b = fmt.count_bits_tensor(x) / x.nelement()
            assert expected_b - tol < actual_b < expected_b + tol
            assert Q.qrmse_norm(fmt, x).item() < 0.12  # empirical
