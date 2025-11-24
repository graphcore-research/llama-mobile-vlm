import wandb
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

    path = "/Users/lukar/volt-cluster/.env"
    with open(path) as f:
        env = {}
        for line in f.readlines():
            k, v = line.strip().split("=")
            env[k] = v

    api = wandb.Api()
    missing_runs = runs = api.runs(
        "llama-mobile",
        filters={"config.name": "sweep-21-11-25", "state": {"$ne": "finished"}},
    )

    for run in missing_runs:
        block_shape = tuple(run.config["quantisation"]["fmt"]["block_shape"])
        lr = run.config["training"]["optimiser"]["lr"]
        n_steps = run.config["training"]["n_steps"]

        settings.quantisation = train.QuantisationSettings(
            fmt=Q.LinearScalingFormat(
                Q.parse("E0M2"),
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
