import train
from cluster import Job, Submission, submit

if __name__ == "__main__":
    settings = train.Settings.default()
    settings.run_name = "new-prompts-10-03-26"
    gen_path = "generation/llama-3.2-11b-vision-instruct"
    settings.data.train[0].path = f"{gen_path}/imagenet-train/new-prompts-1280k"

    path = "/Users/lukar/volt-cluster/.env"
    with open(path) as f:
        env = {}
        for line in f.readlines():
            k, v = line.strip().split("=")
            env[k] = v

    settings.training.optimiser.lr = 2**-17

    for n_steps in [256, 512, 1024, 2048, 4096, 4096 + 2048, 8192]:
        settings.training.n_steps = n_steps
        sub = Submission(
            user="lukar",
            project="llama-mobile",
            env=env,
            job=Job(train.run_experiment, (settings,), {}),
        )
        submit(sub)
