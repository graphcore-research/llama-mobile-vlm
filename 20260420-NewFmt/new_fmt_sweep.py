"""
New format sweep
- [R]: apply rotational matrices
- [A]: quantise activations to INT8 (channel-scaled)

Run:
- New format [RA]: 2k, 4k steps
    - 2k steps, ablate no [R/A/RA]
    - 2k steps, add INT8 for ViT, LM head, ViT + LM head
- Old format [3bit student-t + 64-group]
    - Same ablations

Note: lr fixed at 2**-17
"""

import weight_formats.fit as F
import weight_formats.quantisation as Q

import train
from cluster import Job, Submission, submit

if __name__ == "__main__":
    settings = train.Settings.default()
    settings.run_name = "new-fmt-20-04-26"
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
            )
            submit(sub)

    # Fix R + A, exclude projection/ViT params
    settings.training.rotate_text_residual = True
    settings.quantisation.activation_fmt = train.FMT_CHANNEL_INT8
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
                )
                submit(sub)

    # Run 4k steps
    settings.training.rotate_text_residual = True
    settings.quantisation.activation_fmt = train.FMT_CHANNEL_INT8
    settings.training.n_steps = 4096
    for exclude in [False, True]:
        overrides = {}
        if exclude:
            overrides["language_model.lm_head"] = train.FMT_CHANNEL_INT8
            overrides["vision_model"] = train.FMT_CHANNEL_INT8
            overrides["multi_modal_projector"] = train.FMT_CHANNEL_INT8
        settings.quantisation.overrides = overrides
        sub = Submission(
            user="lukar",
            project="llama-mobile",
            env=env,
            job=Job(train.run_experiment, (settings,), {}),
        )
        submit(sub)

    # Old format 2k steps, ablate
    settings.training.n_steps = 2048
    settings.quantisation.fmt = F.Scaled(
        element_bits=3,
        element_family="t",
        scale_format=Q.parse("BFLOAT16"),
        block_shape=(1, 64),
        scaling="absmax",
    )
    settings.quantisation.overrides = {}
    for rotate in [True, False]:
        for activation_fmt in [train.FMT_CHANNEL_INT8, None]:
            settings.training.rotate_text_residual = rotate
            settings.quantisation.activation_fmt = activation_fmt
            sub = Submission(
                user="lukar",
                project="llama-mobile",
                env=env,
                job=Job(train.run_experiment, (settings,), {}),
            )
            submit(sub)

    settings.training.rotate_text_residual = True
    settings.quantisation.activation_fmt = train.FMT_CHANNEL_INT8
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
                )
                submit(sub)
