"""
Compare fixed prompt ("Describe the image") and prompt sampling
- Fix format: INT3 + 64-block scaling + INT8, channel-scaled activations
- Vary number steps: [256, 512, 1024, 2048]
"""

import weight_formats.quantisation as Q

import train
from cluster import Job, Submission, submit

if __name__ == "__main__":
    settings = train.Settings.default()
    settings.run_name = "prompt-comparison-24-04-26"
    train_data_path = "generation/llama-3.2-11b-vision-instruct/imagenet-train"

    path = "/Users/lukar/volt-cluster/.env"
    with open(path) as f:
        env = {}
        for line in f.readlines():
            k, v = line.strip().split("=")
            env[k] = v

    settings.save_checkpoint = False

    settings.training.rotate_text_residual = False
    settings.quantisation.activation_fmt = train.FMT_CHANNEL_INT8
    settings.training.optimiser.lr = 2**-17
    settings.quantisation.fmt = Q.LinearScalingFormat(
        Q.IntFormat(3),
        scale_format=Q.BFLOAT16,
        block_shape=(1, 64),
        scaling="absmax",
    )

    for data_name in ["new-prompts-1280k", "single-prompt-1280k"]:
        settings.data.train[0].path = f"{train_data_path}/{data_name}"
        for n_steps in [256, 512, 1024, 2048]:
            settings.training.n_steps = n_steps
            sub = Submission(
                user="lukar",
                project="llama-mobile",
                env=env,
                job=Job(train.run_experiment, (settings,), {}),
                priority="high",
            )
            submit(sub)
