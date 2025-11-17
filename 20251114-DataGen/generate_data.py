import train_data
from cluster import Job, Submission, submit

if __name__ == "__main__":
    config = train_data.GenerationConfig.default()
    n_per_job = 160_000
    data_ranges = []

    # NOTE: There are around 1_280_000 images in ImageNet train ds
    for i in range(0, 160_000 * 8, n_per_job):
        data_range = (i, i + n_per_job)
        print(data_range)
        data_ranges.append(data_range)

    path = "/Users/lukar/volt-cluster/.env"
    with open(path) as f:
        env = {}
        for line in f.readlines():
            k, v = line.strip().split("=")
            env[k] = v

    for data_range in data_ranges:
        config.data_range = data_range
        sub = Submission(
            user="lukar",
            project="llama-mobile",
            env=env,
            job=Job(train_data.generate_data, (config,), {}),
        )
        submit(sub)
