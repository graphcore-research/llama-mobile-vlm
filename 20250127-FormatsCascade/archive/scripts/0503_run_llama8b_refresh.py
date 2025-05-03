import block_formats.experiments as E
import block_formats.experiments.token_prediction as ET
import block_formats.fit as F
import block_formats.quantisation as Q

if __name__ == "__main__":
    tests = []
    for element_family, compressor, args in [
        ("lloyd_max", None, {}),
        ("int", None, {}),
        ("fp", None, {}),
        ("normal", None, {}),
        ("laplace", None, {}),
        ("t", None, {}),
    ]:
        for element_bits in [5, 4, 3]:
            for scaling in ["absmax", "rms"]:
                for block_shape in [(None, None)] + [(1, b) for b in [None, 32, 64]]:
                    fmt = F.Scaled(
                        element_bits=element_bits,
                        element_family=element_family,
                        scale_format=Q.BFLOAT16,
                        block_shape=block_shape,
                        scaling=scaling,
                        compressor=compressor,
                        args=args,
                    )
                    tests.append(ET.QuantiseFixed(fmt))

    ET.run_sweep(
        [
            ET.Run("20250503-llama8b-refresh", test, model="meta-llama/Llama-3.1-8B")
            for test in tests
        ]
    )
