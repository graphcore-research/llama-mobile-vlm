"""
Baseline format sweep:
- Scalar Lloyd-Max format, with [4, 6, 8, 12, 16] codepoints
- Channel-scaling, absmax
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
    settings.run_name = "lloyd-max-baseline-22-04-26"
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
        # NOTE: n_bits = 8 / 3 gives 6.35 codepoints, which is rounded to 6
        for n_points in [4, 6, 8, 12, 16]:
            n_bits = log2(n_points)
            settings.training.optimiser.lr = 2 ** (-(n_bits + 14))
            settings.quantisation.fmt = F.Scaled(
                n_bits,
                "lloyd_max",
                scale_format=Q.BFLOAT16,
                block_shape=(1, None),
                scaling="absmax",
                args=dict(threshold=1e-3),
            )
            sub = Submission(
                user="lukar",
                project="llama-mobile",
                env=env,
                job=Job(train.run_experiment, (settings,), {}),
                priority="high",
            )
            submit(sub)
