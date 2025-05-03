import block_formats.experiments as E
import block_formats.experiments.token_prediction as ET
import block_formats.fit as F
import block_formats.quantisation as Q

if __name__ == "__main__":
    tests = [ET.Baseline()]
    for element_bits in [3, 4, 5]:
        for element_family, compressor, args in [
            ("int", None, {}),
            ("int", "optimal", {}),
            ("fp", None, {}),
            ("fp", None, dict(exponent_bits=2)),
            ("normal", None, {}),
            ("laplace", None, {}),
            ("t", None, {}),
            ("t", None, dict(df=10)),
            ("t", None, dict(df=30)),
            ("lloyd_max", None, {}),
        ]:
            for mode_args in (
                [dict(mode="symmetric"), dict(mode="asymmetric")]
                if element_family in ["normal", "laplace", "t"]
                else [{}]
            ):
                for scaling in ["rms", "absmax"] + (
                    ["signmax"] if (element_family, compressor) == ("int", None) else []
                ):
                    for scale_format in [Q.BFLOAT16] + (
                        [] if scaling == "rms" else [Q.parse("EXP8")]
                    ):
                        for block_shape in [(None, None)] + [
                            (1, b) for b in [None, 16, 32, 64, 128, 256]
                        ]:
                            for sparse_ratio in [0, 2**-7, 2**-10]:
                                fmt = F.Scaled(
                                    element_bits=element_bits,
                                    element_family=element_family,
                                    scale_format=scale_format,
                                    block_shape=block_shape,
                                    scaling=scaling,
                                    sparse_format=Q.BFLOAT16,
                                    sparse_ratio=sparse_ratio,
                                    compressor=compressor,
                                    args=dict(**args, **mode_args),
                                )
                                for error_weight in [None] + (
                                    ["fisher"] if fmt.supports_error_weight else []
                                ):
                                    for test_cls in [
                                        ET.QuantiseFixed,
                                        ET.QuantiseVariable,
                                        ET.QuantiseHeuristic,
                                    ]:
                                        tests.append(
                                            test_cls(fmt, error_weight=error_weight)
                                        )

    ET.run_sweep(
        [
            ET.Run("20250425-grid", test, model, batch_size=2)
            for model in [
                # 1B models
                "meta-llama/Llama-3.2-1B",
                "google/gemma-3-1b-pt",
                "Qwen/Qwen2.5-1.5B",
                # 3B models
                "meta-llama/Llama-3.2-3B",
                "google/gemma-3-4b-pt",
                "Qwen/Qwen2.5-3B",
            ]
            for test in tests
        ]
    )
