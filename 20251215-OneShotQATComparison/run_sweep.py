import train
from cluster import Job, Submission, submit

if __name__ == "__main__":
    settings = train.Settings.default()
    settings.run_name = "one-shot-quant-11-12-25"

    path = "/Users/lukar/volt-cluster/.env"
    with open(path) as f:
        env = {}
        for line in f.readlines():
            k, v = line.strip().split("=")
            env[k] = v

    for lr in [2**-i for i in range(15, 20)]:
        for n_steps in [256, 512, 1024, 2048]:
            for mode in ["qat", "one-shot"]:
                settings.training.optimiser.lr = lr
                settings.training.n_steps = n_steps
                settings.quantisation.mode = mode

                sub = Submission(
                    user="lukar",
                    project="llama-mobile",
                    env=env,
                    job=Job(train.run_experiment, (settings,), {}),
                )
                submit(sub)

    # Without training
    settings.training.n_steps = 0
    sub = Submission(
        user="lukar",
        project="llama-mobile",
        env=env,
        job=Job(train.run_experiment, (settings,), {}),
    )
    submit(sub)
