import train
from cluster import Job, Submission, submit
import weight_formats.quantisation as Q


if __name__ == "__main__":
    settings = train.Settings.default()
    settings.run_name = "formats-19-02-26"
    settings.training.n_steps = 2048

    path = "/Users/lukar/volt-cluster/.env"
    with open(path) as f:
        env = {}
        for line in f.readlines():
            k, v = line.strip().split("=")
            env[k] = v

    # Block formats
    for el_fmt, lr in zip(("E0M1", "E0M2", "E2M1"), (2**-16, 2**-17, 2**-18)):
        settings.training.optimiser.lr = lr
        for group_size in (16, 32, 64, 128):
            settings.quantisation.fmt = Q.LinearScalingFormat(
                Q.parse(el_fmt),
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

    # Student-t
    for n_bits, lr in zip(range(2, 5), (2**-16, 2**-17, 2**-18)):
        settings.training.optimiser.lr = lr
        for group_size in (16, 32, 64, 128):
            settings.quantisation.fmt = Q.LinearScalingFormat(
                Q.crd_block_t(
                    bits=n_bits,
                    block_size=group_size,
                    df=30,
                ),
                scale_format=Q.parse("BFLOAT16"),
                block_shape=(1, group_size),
                scaling="absmax",
            )

            sub = Submission(
                user="lukar",
                project="llama-mobile",
                env=env,
                job=Job(train.run_experiment, (settings,), {}),
                gpu_clique="cc914f6f-5ed5-ca2b-745b-b0b7cf09b43c.2",
            )
            submit(sub)
