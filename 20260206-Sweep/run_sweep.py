import train
from cluster import Job, Submission, submit

if __name__ == "__main__":
    settings = train.Settings.default()
    settings.run_name = "sweep-05-02-26"

    path = "/Users/lukar/volt-cluster/.env"
    with open(path) as f:
        env = {}
        for line in f.readlines():
            k, v = line.strip().split("=")
            env[k] = v

    for lr in [2**-i for i in range(15, 20)]:
        for n_steps in [256, 512, 1024, 2048]:
            settings.training.optimiser.lr = lr
            settings.training.n_steps = n_steps

            sub = Submission(
                user="lukar",
                project="llama-mobile",
                env=env,
                job=Job(train.run_experiment, (settings,), {}),
            )
            submit(sub)
