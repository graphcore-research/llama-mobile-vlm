# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import torch

from .. import fit as F
from .. import quantisation as Q


def test_scaled_quantiser() -> None:
    torch.manual_seed(100)
    x = 3 * torch.randn(2**16)
    tol = 0.1

    # Compressed (RMS)
    fmt = F.Scaled(4, "int", Q.BFLOAT16, (None,), "rms", compressor="optimal").fit(x)
    assert 4 - tol < fmt.count_bits_tensor(x) / x.nelement() < 4 + tol
    assert Q.qrmse_norm(fmt, x).item() < 0.08  # empirical

    # Compressed (block + sparse)
    fmt = F.Scaled(
        4,
        "int",
        Q.BFLOAT16,
        (32,),  # 16 bits per 32 elements
        "absmax",
        compressor="optimal",
        sparse_format=Q.FP32,
        sparse_ratio=2**-8,  # 64 bits (32 idx + 32 val) per 256 elements
    ).fit(x)
    assert 4.75 - tol < fmt.count_bits_tensor(x) / x.nelement() < 4.75 + tol
    assert Q.qrmse_norm(fmt, x).item() < 0.08  # empirical

    # Lloyd-Max
    fmt = F.Scaled(4, "lloyd_max", Q.BFLOAT16, (32,), "signmax", compressor=None).fit(x)
    assert fmt.count_bits_tensor(x) / x.nelement() == 4.5
    assert Q.qrmse_norm(fmt, x).item() < 0.081  # empirical

    # Sweep
    for element_family in ["int", "fp", "normal", "laplace", "t"]:
        for block_size, scaling, sparse_ratio in [
            (None, "rms", 0),
            (16, "rms", 2**-8),
            (32, "absmax", 0),
        ]:
            fmt = F.Scaled(
                4,
                element_family,  # type:ignore[arg-type]
                Q.BFLOAT16,
                (block_size,),
                scaling,  # type:ignore[arg-type]
                compressor=None,
                sparse_ratio=sparse_ratio,
                sparse_format=Q.BFLOAT16,
            ).fit(x)
            expected_b = (
                4 + 16 / (block_size or x.nelement()) + sparse_ratio * (16 + 32)
            )
            actual_b = fmt.count_bits_tensor(x) / x.nelement()
            assert expected_b - tol < actual_b < expected_b + tol
            assert Q.qrmse_norm(fmt, x).item() < 0.12  # empirical
