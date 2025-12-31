import train
from cluster import Job, Submission, submit

if __name__ == "__main__":
    settings = train.Settings.default()
    settings.run_name = "no-vision-16-12-25"
    exclude = ["vision_model", "multi_modal_projector"]
    settings.quantisation.exclude = exclude

    path = "/Users/lukar/volt-cluster/.env"
    with open(path) as f:
        env = {}
        for line in f.readlines():
            k, v = line.strip().split("=")
            env[k] = v

    for lr in [2**-i for i in range(15, 20)]:
        for n_steps in [256, 512, 1024, 2048]:
            for freeze in [False, True]:
                # for mode in ["qat", "one-shot"]:
                settings.training.optimiser.lr = lr
                settings.training.n_steps = n_steps
                if freeze:
                    settings.training.freeze_params = exclude
                # settings.quantisation.mode = mode

                sub = Submission(
                    user="lukar",
                    project="llama-mobile",
                    env=env,
                    job=Job(train.run_experiment, (settings,), {}),
                )
                submit(sub)
