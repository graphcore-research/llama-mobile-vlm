import sys
from typing import Iterable

import torch

import block_formats.experiments as E
import block_formats.experiments.token_prediction as ET
import block_formats.fit as F
import block_formats.quantisation as Q


def _fisher(step: float) -> Iterable[ET.Test]:
    for element_bits in torch.arange(3, 5.01, step).tolist():
        for element_family, compressor, scaling, block_shape, sparse_ratio in [
            ("int", "optimal", "rms", (None, None), 0),
            ("t", None, "absmax", (1, 128), 0),
            ("t", None, "rms", (None, None), 2**-10),
        ]:
            for mode_args in (
                [dict(mode="symmetric"), dict(mode="asymmetric")]
                if element_family in ["normal", "laplace", "t"]
                or (element_family, compressor) == ("int", None)
                else [{}]
            ):
                fmt = F.Scaled(
                    element_bits=element_bits,
                    element_family=element_family,
                    scale_format=Q.BFLOAT16,
                    block_shape=block_shape,
                    scaling=scaling,
                    sparse_format=Q.BFLOAT16,
                    sparse_ratio=sparse_ratio,
                    compressor=compressor,
                    args=dict(**mode_args),
                )
                for cls in [
                    ET.QuantiseFixed,
                    ET.QuantiseVariable,
                    ET.QuantiseHeuristic,
                ]:
                    yield cls(fmt)


if __name__ == "__main__":
    MOD_ALL = E.MODELS
    MOD_LLAMA8B = ["meta-llama/Llama-3.1-8B"]
    MOD_NOT_LLAMA8B = [m for m in MOD_ALL if m not in MOD_LLAMA8B]

    s = []
    s.append(dict(name="fisher-code", tests=list(_fisher(0.25)), models=MOD_LLAMA8B))
    s.append(dict(name="fisher-code", tests=list(_fisher(1)), models=MOD_NOT_LLAMA8B))

    for sweep in s:
        print(
            f"### {sweep['name']} ({len(sweep['models'])} x {len(sweep['tests'])})",
            file=sys.stderr,
        )
        ET.run_sweep(
            [
                ET.Run(
                    f"20250506-results-{sweep['name']}",
                    test,
                    model,
                    dataset="github-code",
                    sequence_limit=256,
                    line_limit=4096,
                )
                for model in sweep["models"]
                for test in sweep["tests"]
            ]
        )
