import datetime
import json
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

import training.quantisation.quantisation as Q
from training import experiments as E

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    out_path = (
        Path("out")
        / datetime.datetime.now()
        .isoformat(sep="/", timespec="seconds")
        .replace(":", "-")
    ).with_suffix(".jsonl")
    if out_path.exists():
        raise FileExistsError(f"File {out_path} already exists")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def formats() -> Iterable[Q.TensorFormat]:
        for fmt in [
            Q.parse("E5M2"),  # 8 bits
            Q.parse("E2M1"),  # 4 bits
        ]:
            yield Q.group_scaling_format(fmt, grouping="input", group_size=32)

    # Skip quantising these layers (NOTE: also will skip any 1D tensors)
    do_not_quantise = [
        "vision_model.patch_embedding",
        "vision_model.gated_positional",
        "vision_model.pre_tile",
        "vision_model.post_tile",
    ]

    task = E.Task(name="outcompare", n_samples=None)
    execution = E.Execution(device="cuda", batch_size=16, wandb=True)

    # Single layer patterns
    patterns = []
    patterns.append("language_model.model.embed_tokens.weight")
    patterns.extend([f"vision_model.transformer.layers.{i}" for i in range(32)])
    patterns.extend([f"vision_model.global_transformer.layers.{i}" for i in range(8)])
    patterns.extend([f"language_model.model.layers.{i}" for i in range(40)])
    patterns.append("language_model.lm_head.weight")

    experiments = [
        E.Experiment(
            name="individual-layers-v1",
            model=model_name,
            task=task,
            quantisation_formats=[
                ("|".join(do_not_quantise), Q.BFLOAT16),
                (pattern, fmt),
                ("", Q.BFLOAT16),
            ],
            execution=execution,
        )
        for fmt in formats()
        for pattern in patterns
    ]

    with out_path.open("w") as f:
        for xp in tqdm(experiments, "Experiment"):
            out = E.run_experiment(xp)
            print(json.dumps(out), file=f, flush=True)
