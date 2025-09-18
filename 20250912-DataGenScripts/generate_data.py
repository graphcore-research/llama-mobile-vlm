"""
Script for generating training and validation data:
    * imagenet-train-generation/{98cf95, 870805, c91afe}
    * coco-validation-generation/{9ba9e8, 76bcc0}
"""

import train_data
from utility import LLAMA_PROMPT_TEMPLATES

if __name__ == "__main__":
    # === Training data ===

    # Default (chat) prompt
    config1 = train_data.GenerationConfig.default(data_range=(0, 256_000))

    # Simple prompt
    simple_prompt = LLAMA_PROMPT_TEMPLATES["simple"].format(
        prompt="Describe the image:\n"
    )
    config2 = train_data.GenerationConfig.default(
        prompt=simple_prompt, data_range=(256_000, 320_000)
    )
    config3 = train_data.GenerationConfig.default(
        prompt=simple_prompt, data_range=(320_000, 384_000)
    )

    for config in [config1, config2, config3]:
        train_data.generate_data(config)

    # === Validation data ===

    # Chat prompt
    config1 = train_data.GenerationConfig.default(
        dataset_name="coco", split="validation", data_range=(0, 2048)
    )

    # Simple prompt
    config2 = train_data.GenerationConfig.default(
        dataset_name="coco",
        split="validation",
        data_range=(2048, 4096),
        prompt=simple_prompt,
    )

    for config in [config1, config2]:
        train_data.generate_data(config)
