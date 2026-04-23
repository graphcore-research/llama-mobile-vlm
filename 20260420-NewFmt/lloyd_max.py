"""
Scalar Lloyd-Max (point comparison with new format)

Note: lr fixed at 2**-17
"""

import weight_formats.fit as F
import weight_formats.quantisation as Q

import train
from cluster import Job, Submission, submit

if __name__ == "__main__":
    settings = train.Settings.default()
    settings.run_name = "new-fmt-21-04-26"
    gen_path = "generation/llama-3.2-11b-vision-instruct"
    settings.data.train[0].path = f"{gen_path}/imagenet-train/new-prompts-1280k"

    path = "/Users/lukar/volt-cluster/.env"
    with open(path) as f:
        env = {}
        for line in f.readlines():
            k, v = line.strip().split("=")
            env[k] = v

    settings.training.optimiser.lr = 2**-17
    settings.training.n_steps = 2048
    settings.save_checkpoint = True

    settings.quantisation.fmt = F.Scaled(
        8 / 3,
        "lloyd_max",
        scale_format=Q.BFLOAT16,
        block_shape=(1, None),
        scaling="absmax",
        args=dict(threshold=1e-3),
    )

    # Run 0 steps + 4k steps
    settings.training.rotate_text_residual = False
    settings.quantisation.activation_fmt = train.FMT_CHANNEL_INT8
    settings.training.n_steps = 2048
    sub = Submission(
        user="lukar",
        project="llama-mobile",
        env=env,
        job=Job(train.run_experiment, (settings,), {}),
        priority="high",
    )
    submit(sub)
