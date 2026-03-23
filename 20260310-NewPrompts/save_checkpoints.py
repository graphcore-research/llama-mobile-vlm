import weight_formats.fit as F
import weight_formats.quantisation as Q

import train
from cluster import Job, Submission, submit


if __name__ == "__main__":
    settings = train.Settings.default()
    settings.save_checkpoint = True
    settings.run_name = "checkpoints-23-03-26"
    gen_path = "generation/llama-3.2-11b-vision-instruct"
    settings.data.train[0].path = f"{gen_path}/imagenet-train/new-prompts-1280k"

    path = "/Users/lukar/volt-cluster/.env"
    with open(path) as f:
        env = {}
        for line in f.readlines():
            k, v = line.strip().split("=")
            env[k] = v

    n_bits = 3
    lr = 2**-17
    for group_size in [128, 256]:
        settings.training.optimiser.lr = lr
        settings.training.n_steps = 4096
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
        )
        submit(sub)
