import block_formats.experiments.token_prediction as ET
import block_formats.fit as F
import block_formats.quantisation as Q

if __name__ == "__main__":
    tests = [
        ET.QuantiseFixed(
            F.Scaled(
                element_bits=4,
                element_family="fp",
                scale_format=Q.BFLOAT16,
                block_shape=(1, 64),
                scaling="absmax",
            )
        ),
        ET.QuantiseVariable(
            F.Scaled(
                element_bits=4,
                element_family="int",
                scale_format=Q.BFLOAT16,
                block_shape=(None, None),
                scaling="rms",
                sparse_format=Q.BFLOAT16,
                sparse_ratio=2**-10,
                compressor="optimal",
            )
        ),
    ]
    ET.run_sweep(
        [ET.Run("dev", test, "meta-llama/Llama-3.2-1B", batch_size=1) for test in tests]
    )
