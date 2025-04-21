import block_formats.experiments.token_prediction as ET
import block_formats.fit as F
import block_formats.quantisation as Q

if __name__ == "__main__":
    ET.Sweep(
        "20250421-param-sensitivity",
        test=[
            ET.Baseline(),
            ET.PerturbEachParam(1.0),
            ET.PerturbEachParam(1 / 2),
            ET.PerturbEachParam(1 / 4),
            ET.QuantiseEachParam(
                Q.LinearScalingFormat(Q.parse("E2M1"), Q.BFLOAT16, (1, 64), "absmax")
            ),
            ET.QuantiseEachParam(
                Q.LinearScalingFormat(Q.parse("E2M0"), Q.BFLOAT16, (1, 64), "absmax")
            ),
            ET.QuantiseEachParam(
                F.Scaled(4, "int", Q.BFLOAT16, (1, None), "rms", "optimal")
            ),
            ET.QuantiseEachParam(
                F.Scaled(3, "int", Q.BFLOAT16, (1, None), "rms", "optimal")
            ),
        ],
    ).run()
