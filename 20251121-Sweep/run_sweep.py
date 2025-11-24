import weight_formats.quantisation as Q

import train
from cluster import Job, Submission, submit

if __name__ == "__main__":
    settings = train.Settings.default()
    settings.run_name = "sweep-21-11-25"
    settings.data.validation = [
        train.DataShard("vqav2-validation-generation/61e74b", 1024)
    ]
    settings.save_checkpoint = False

    el_fmts = ["E0M2"]
    block_shapes = [(1, 32), (1, 64), (1, 128)]

    path = "/Users/lukar/volt-cluster/.env"
    with open(path) as f:
        env = {}
        for line in f.readlines():
            k, v = line.strip().split("=")
            env[k] = v

    for el_fmt in el_fmts:
        for block_shape in block_shapes:
            for n_steps in [0, 8192]:
                for lr in [2**-i for i in range(17, 22)]:
                    settings.quantisation = train.QuantisationSettings(
                        fmt=Q.LinearScalingFormat(
                            Q.parse(el_fmt),
                            scale_format=Q.parse("BFLOAT16"),
                            block_shape=block_shape,
                            scaling="absmax",
                        )
                    )
                    settings.training.n_steps = n_steps
                    settings.training.optimiser.lr = lr

                    sub = Submission(
                        user="lukar",
                        project="llama-mobile",
                        env=env,
                        job=Job(train.run_experiment, (settings,), {}),
                    )
                    submit(sub)
