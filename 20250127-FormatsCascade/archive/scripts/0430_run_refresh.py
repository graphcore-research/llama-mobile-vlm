import block_formats.experiments as E
import block_formats.experiments.token_prediction as ET
import block_formats.fit as F
import block_formats.quantisation as Q

if __name__ == "__main__":
    tests = []
    for element_bits in [3, 4, 5]:
        for element_family, compressor, args in [
            ("int", None, {}),
            ("int", "optimal", {}),
            ("fp", None, {}),
            ("fp", None, dict(exponent_bits=2)),
            ("normal", None, {}),
            ("laplace", None, {}),
            ("t", None, {}),
            ("t", None, dict(df=30)),
            ("lloyd_max", None, {}),
        ]:
            for scaling in ["absmax", "rms"]:
                for block_shape in [(None, None)] + [(1, b) for b in [None, 32, 64]]:
                    for sparse_ratio in [0, 2**-7, 2**-10]:
                        fmt = F.Scaled(
                            element_bits=element_bits,
                            element_family=element_family,
                            scale_format=Q.BFLOAT16,
                            block_shape=block_shape,
                            scaling=scaling,
                            sparse_format=Q.BFLOAT16,
                            sparse_ratio=sparse_ratio,
                            compressor=compressor,
                            args=args,
                        )
                        tests.append(ET.QuantiseVariable(fmt))
                        if fmt.supports_error_weight:
                            tests.append(
                                ET.QuantiseVariable(fmt, error_weight="fisher")
                            )

    ET.run_sweep(
        [
            ET.Run("20250430-refresh", test, "meta-llama/Llama-3.1-8B", batch_size=1)
            for test in tests
        ]
    )
