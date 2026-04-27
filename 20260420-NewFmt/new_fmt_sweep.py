"""
New format sweep
- [R]: apply rotational matrices
- [A]: quantise activations to INT8 (channel-scaled)

Run:
    - 2k steps, ablate no [R/A/RA]
    - 2k steps, with [RA], add INT8 for ViT / LM head / ViT + LM head

Note: lr fixed at 2**-17
"""

import train
from cluster import Job, Submission, submit

if __name__ == "__main__":
    settings = train.Settings.default()
    settings.run_name = "new-fmt-21-04-26"
    gen_path = "generation/llama-3.2-11b-vision-instruct"
    settings.data.train[0].path = f"{gen_path}/imagenet-train/new-prompts-1280k"

    path = "/Users/lukar/volt-cluster/.env"
    with open(path) as f:
        env = {}
        for line in f.readlines():
            k, v = line.strip().split("=")
            env[k] = v

    settings.training.optimiser.lr = 2**-17
    settings.training.n_steps = 2048
    settings.save_checkpoint = True

    # New format
    settings.quantisation.fmt = train.FMT_CHANNEL_S3D8

    # Ablate rotations + activation quantisation
    for rotate in [True, False]:
        for activation_fmt in [train.FMT_CHANNEL_INT8, None]:
            settings.training.rotate_text_residual = rotate
            settings.quantisation.activation_fmt = activation_fmt
            sub = Submission(
                user="lukar",
                project="llama-mobile",
                env=env,
                job=Job(train.run_experiment, (settings,), {}),
                priority="high",
            )
            submit(sub)

    # Fix R + A, exclude projection/ViT params
    settings.quantisation.activation_fmt = train.FMT_CHANNEL_INT8
    for rotate in [False, True]:
        settings.training.rotate_text_residual = rotate
        for exclude_proj in [False, True]:
            for exclude_vit in [False, True]:
                overrides = {}
                if exclude_proj:
                    overrides["language_model.lm_head"] = train.FMT_CHANNEL_INT8
                if exclude_vit:
                    overrides["vision_model"] = train.FMT_CHANNEL_INT8
                    overrides["multi_modal_projector"] = train.FMT_CHANNEL_INT8
                if overrides:
                    settings.quantisation.overrides = overrides
                    sub = Submission(
                        user="lukar",
                        project="llama-mobile",
                        env=env,
                        job=Job(train.run_experiment, (settings,), {}),
                        priority="high",
                    )
                    submit(sub)

    # Run 4k steps
    settings.training.rotate_text_residual = False
    settings.quantisation.activation_fmt = train.FMT_CHANNEL_INT8
    settings.training.n_steps = 4096
    sub = Submission(
        user="lukar",
        project="llama-mobile",
        env=env,
        job=Job(train.run_experiment, (settings,), {}),
        priority="high",
    )
    submit(sub)

    # Run 0 steps, rotations ON/OFF
    settings.training.n_steps = 0
    for rotate in [False, True]:
        settings.training.rotate_text_residual = rotate
        sub = Submission(
            user="lukar",
            project="llama-mobile",
            env=env,
            job=Job(train.run_experiment, (settings,), {}),
            priority="high",
        )
        submit(sub)
