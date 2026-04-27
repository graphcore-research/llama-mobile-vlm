"""
Show that lr=2**-17 is ~optimal for the default format INT3, 64 block size
"""

import weight_formats.quantisation as Q

import train
from cluster import Job, Submission, submit

if __name__ == "__main__":
    settings = train.Settings.default()
    settings.run_name = "lr-sweep-27-04-26"
    train_data_path = "generation/llama-3.2-11b-vision-instruct/imagenet-train"
    settings.data.train[0].path = f"{train_data_path}/new-prompts-1280k"

    path = "/Users/lukar/volt-cluster/.env"
    with open(path) as f:
        env = {}
        for line in f.readlines():
            k, v = line.strip().split("=")
            env[k] = v

    settings.save_checkpoint = False
    settings.training.rotate_text_residual = False
    settings.quantisation.activation_fmt = train.FMT_CHANNEL_INT8
    settings.quantisation.fmt = Q.LinearScalingFormat(
        Q.IntFormat(3),
        scale_format=Q.BFLOAT16,
        block_shape=(1, 64),
        scaling="absmax",
    )
    settings.training.n_steps = 2048

    for lr_exp in [-19, -18, -17, -16, -15]:
        settings.training.optimiser.lr = 2**lr_exp
        sub = Submission(
            user="lukar",
            project="llama-mobile",
            env=env,
            job=Job(train.run_experiment, (settings,), {}),
            priority="high",
        )
        submit(sub)
