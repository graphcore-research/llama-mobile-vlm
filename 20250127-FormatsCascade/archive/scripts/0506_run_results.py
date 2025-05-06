import torch
from typing import Iterable
import block_formats.experiments as E
import block_formats.experiments.token_prediction as ET
import block_formats.fit as F
import block_formats.quantisation as Q


def _tests_main() -> Iterable[ET.Test]:
    for element_bits in torch.arange(3, 5.01, 0.25).tolist():
        for element_family, compressor, args in [
            ("int", "optimal", {}),
            ("int", None, {}),
            ("t", None, {}),
        ]:
            for mode_args in (
                [dict(mode="symmetric"), dict(mode="asymmetric")]
                if element_family in ["normal", "laplace", "t"]
                or (element_family, compressor) == ("int", None)
                else [{}]
            ):
                for scaling in ["rms", "absmax"]:
                    for block_shape in [(None, None), (1, None)] + [
                        (1, b) for b in [64, 128]
                    ]:
                        for sparse_ratio in [0, 2**-7, 2**-10]:
                            yield ET.QuantiseFixed(
                                F.Scaled(
                                    element_bits=element_bits,
                                    element_family=element_family,
                                    scale_format=Q.BFLOAT16,
                                    block_shape=block_shape,
                                    scaling=scaling,
                                    sparse_format=Q.BFLOAT16,
                                    sparse_ratio=sparse_ratio,
                                    compressor=compressor,
                                    args=dict(**args, **mode_args),
                                )
                            )


if __name__ == "__main__":
    all_tests = dict(
        baseline=[ET.Baseline()],
        main=list(_tests_main()),
    )
    models = ["meta-llama/Llama-3.1-8B"]

    for name, tests in all_tests.items():
        ET.run_sweep(
            [
                ET.Run(f"20250506-results-{name}", test, model)
                for model in models
                for test in tests
            ]
        )
