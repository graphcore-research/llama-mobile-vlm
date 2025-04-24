import block_formats.experiments as E
import block_formats.experiments.token_prediction as ET
import block_formats.fit as F
import block_formats.quantisation as Q

if __name__ == "__main__":
    tests = [
        ET.Baseline(),
        ET.QuantiseFixed(
            F.Scaled(
                3, "fp", Q.BFLOAT16, (1, 64), "absmax", None, dict(exponent_bits=2)
            )
        ),
        ET.QuantiseVariable(
            F.Scaled(
                3, "fp", Q.BFLOAT16, (1, 64), "absmax", None, dict(exponent_bits=2)
            )
        ),
    ]
    ET.run_sweep(
        [ET.Run("dev", test, model) for model in E.MODELS[:2] for test in tests]
    )
