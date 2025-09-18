"""
Example training script with new data (clear-morning-942)
"""

import train

if __name__ == "__main__":
    settings = train.Settings.default()
    settings.data.validation = [
        train.DataShard("coco-validation-generation/9ba9e8", 128),
        train.DataShard("coco-validation-generation/76bcc0", 512),
    ]
    print("Training data:\n", settings.data.train)
    print("Validation data:\n", settings.data.validation)
    settings.run_name = "test-new-ds"
    settings.training.n_steps = 1536
    settings.save_checkpoint = True
    train.run_experiment(settings)
