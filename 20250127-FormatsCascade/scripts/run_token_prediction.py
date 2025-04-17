import block_formats.experiments as E
import block_formats.fit as F
import block_formats.quantisation as Q

if __name__ == "__main__":
    E.token_prediction.Sweep(
        "20250417-token-prediction",
        model=["meta-llama/Llama-3.2-1B"],
        format=[
            {
                "*": F.Scaled(
                    3, "int", Q.BFLOAT16, (None, None), "rms", compressor="optimal"
                )
            }
        ],
    ).run()
