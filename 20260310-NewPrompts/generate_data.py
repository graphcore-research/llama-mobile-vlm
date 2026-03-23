import random

import train_data
from cluster import Job, Submission, submit

if __name__ == "__main__":
    path = "/Users/lukar/volt-cluster/.env"
    with open(path) as f:
        env = {}
        for line in f.readlines():
            k, v = line.strip().split("=")
            env[k] = v

    # ====== Default seed ======
    config = train_data.GenerationConfig.default()
    n = 1_280_000
    n_per_run = 80_000
    for i in range(n // n_per_run):
        config.data_range = (i * n_per_run, (i + 1) * n_per_run)
        sub = Submission(
            user="lukar",
            project="llama-mobile",
            env=env,
            job=Job(
                train_data.generate_data,
                (config,),
                {"dir_name": f"new-prompts-1280k-{i}"},
            ),
        )
        submit(sub)

    # ====== Generate more examples with a different seed ======
    master_seed = 110326
    rng = random.Random(master_seed)
    config.seed, config.prompt_config.seed = rng.randrange(2**32), rng.randrange(2**32)

    for i in range(n // n_per_run):
        config.data_range = (i * n_per_run, (i + 1) * n_per_run)
        sub = Submission(
            user="lukar",
            project="llama-mobile",
            env=env,
            job=Job(
                train_data.generate_data,
                (config,),
                {"dir_name": f"new-prompts-1280k-seed-{master_seed}-{i}"},
            ),
        )
        submit(sub)
