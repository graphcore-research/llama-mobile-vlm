import block_formats.experiments as E
import block_formats.experiments.token_prediction as ET
import block_formats.fit as F
import block_formats.quantisation as Q

if __name__ == "__main__":
    tests = []
    for element_bits in [3, 3.5, 4, 4.5, 5]:
        for scaling in ["rms", "absmax"]:
            for block_shape in [(None, None)] + [(1, b) for b in [None, 16, 64, 256]]:
                for sparse_ratio in [0, 2**-10, 2**-7]:
                    fmt = F.Scaled(
                        element_bits=element_bits,
                        element_family="int",
                        scale_format=Q.BFLOAT16,
                        block_shape=block_shape,
                        scaling=scaling,
                        sparse_format=Q.BFLOAT16,
                        sparse_ratio=sparse_ratio,
                        compressor="optimal",
                    )
                    tests.append(ET.QuantiseVariable(fmt))
    ET.run_sweep(
        [
            ET.Run(
                "20250429-compress-sparse-v3",
                test,
                "meta-llama/Llama-3.2-1B",
                batch_size=1,
            )
            for test in tests
        ]
    )
