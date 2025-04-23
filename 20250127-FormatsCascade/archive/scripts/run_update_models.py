import block_formats.experiments.token_prediction as ET
import block_formats.fit as F
import block_formats.quantisation as Q

if __name__ == "__main__":
    ET.Sweep(
        "20250422-update-models",
        test=[
            ET.Baseline(),
            ET.QuantiseFixed(
                Q.LinearScalingFormat(Q.parse("E2M1"), Q.BFLOAT16, (1, 64), "absmax")
            ),
            ET.QuantiseFixed(
                F.Scaled(4, "int", Q.BFLOAT16, (1, None), "rms", "optimal")
            ),
            ET.QuantiseFixed(
                Q.LinearScalingFormat(Q.parse("E0M2"), Q.BFLOAT16, (1, 64), "signmax")
            ),
            ET.QuantiseFixed(
                F.Scaled(3, "int", Q.BFLOAT16, (1, None), "rms", "optimal")
            ),
        ],
    ).run()
