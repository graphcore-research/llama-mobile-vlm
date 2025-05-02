import block_formats.experiments as E
import block_formats.experiments.token_prediction as ET

if __name__ == "__main__":
    ET.run_sweep(
        [
            ET.Run("20250502-noise-sensitivity", ET.PerturbEachParam(scale), model)
            for model in E.MODELS
            for scale in [1 / 4, 1 / 2, 1]
        ]
    )
