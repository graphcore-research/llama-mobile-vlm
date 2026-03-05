import train
from cluster import Job, Submission, submit

if __name__ == "__main__":
    settings = train.Settings.default()
    settings.run_name = "single-prompt-02-03-26"
    settings.data.train[0].path = settings.data.train[0].path.replace(
        "default", "single-prompt"
    )

    path = "/Users/lukar/volt-cluster/.env"
    with open(path) as f:
        env = {}
        for line in f.readlines():
            k, v = line.strip().split("=")
            env[k] = v

    settings.training.optimiser.lr = 2**-17

    for n_steps in [256, 512, 1024, 2048, 4096]:
        settings.training.n_steps = n_steps
        sub = Submission(
            user="lukar",
            project="llama-mobile",
            env=env,
            job=Job(train.run_experiment, (settings,), {}),
            gpu_clique="cc914f6f-5ed5-ca2b-745b-b0b7cf09b43c.2",
        )
        submit(sub)
