"""
Baseline format sweep:
- student-t fitted element format, with [4, 6, 8, 12, 16] codepoints
- block size [32, 64, 128]
- absmax scaling
- Activations in INT8, channel-scaled
- No rotations
- lr = 2**(-(n_bits + 14))
    - Optimal lr for ~3bits 2**-17
    - For each extra bit, decrease learning rate x0.5
"""

from math import log2

import weight_formats.fit as F
import weight_formats.quantisation as Q

import train
from cluster import Job, Submission, submit

if __name__ == "__main__":
    settings = train.Settings.default()
    settings.run_name = "student-t-baseline-22-04-26"
    gen_path = "generation/llama-3.2-11b-vision-instruct"
    settings.data.train[0].path = f"{gen_path}/imagenet-train/new-prompts-1280k"

    path = "/Users/lukar/volt-cluster/.env"
    with open(path) as f:
        env = {}
        for line in f.readlines():
            k, v = line.strip().split("=")
            env[k] = v

    settings.save_checkpoint = False

    settings.training.rotate_text_residual = False
    settings.quantisation.activation_fmt = train.FMT_CHANNEL_INT8

    for n_steps in [0, 2048]:
        settings.training.n_steps = n_steps
        for n_points in [4, 6, 8, 12, 16]:
            n_bits = log2(n_points)
            settings.training.optimiser.lr = 2 ** (-(n_bits + 14))
            for group_size in [32, 64, 128]:
                settings.quantisation.fmt = F.Scaled(
                    element_bits=n_bits,
                    element_family="t",
                    scale_format=Q.parse("BFLOAT16"),
                    block_shape=(1, group_size),
                    scaling="absmax",
                )
                sub = Submission(
                    user="lukar",
                    project="llama-mobile",
                    env=env,
                    job=Job(train.run_experiment, (settings,), {}),
                    priority="high",
                )
                submit(sub)
