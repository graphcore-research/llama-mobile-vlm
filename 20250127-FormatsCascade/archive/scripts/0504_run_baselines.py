import block_formats.experiments as E
import block_formats.experiments.token_prediction as ET

if __name__ == "__main__":
    ET.run_sweep(
        [ET.Run("20250504-baselines", ET.Baseline(), model=model) for model in E.MODELS]
    )
