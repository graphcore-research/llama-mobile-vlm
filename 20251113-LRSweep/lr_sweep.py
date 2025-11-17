import train
from cluster import Job, Submission, submit

if __name__ == "__main__":
    settings = train.Settings.default()
    settings.run_name = "lr-sweep-13-11-25"
    settings.training.n_steps = 2048
    lrs = [2**-i for i in range(14, 22)]

    path = "/Users/lukar/volt-cluster/.env"
    with open(path) as f:
        env = {}
        for line in f.readlines():
            k, v = line.strip().split("=")
            env[k] = v

    for lr in lrs:
        settings.training.optimiser.lr = lr
        sub = Submission(
            user="lukar",
            project="llama-mobile",
            env=env,
            job=Job(train.run_experiment, (settings,), {}),
        )
        submit(sub)
