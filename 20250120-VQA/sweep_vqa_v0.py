import datetime
import json
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

import experiments as E
import quantisation as Q

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
        yield Q.BFLOAT16
        for fmt in [
            Q.parse("E5M2"),  # 8 bits
            Q.parse("E4M3"),  # 8 bits
            Q.parse("E0M7"),  # 8 bits
            Q.parse("E3M2"),  # 6 bits
            Q.parse("E2M3"),  # 6 bits
            Q.parse("E0M5"),  # 6 bits
            Q.parse("E2M1"),  # 4 bits
            Q.parse("E0M3"),  # 4 bits
            Q.parse("E0M2"),  # 3 bits
        ]:
            yield Q.tensor_scaling_format(fmt)
            yield Q.channel_scaling_format(fmt, per="input")
            yield Q.channel_scaling_format(fmt, per="output")
            for group_size in [8, 16, 32, 64]:
                yield Q.group_scaling_format(
                    fmt, grouping="input", group_size=group_size
                )

    # Skip quantising these layers (NOTE: also will skip any 1D tensors)
    do_not_quantise = [
        "vision_model.patch_embedding",
        "vision_model.gated_positional",
        "vision_model.pre_tile",
        "vision_model.post_tile",
    ]
    task = E.Task(name="vqa", n_samples=1000, metrics=None)
    execution = E.Execution(device="cuda", batch_size=8, wandb=True)
    experiments = [
        E.Experiment(
            name="sweep-formats-vqa-test",
            model=model_name,
            task=task,
            # NOTE: Should have made this more concise
            quantisation=[
                Q.ParameterRule("|".join(do_not_quantise), None, Q.BFLOAT16),
                Q.ParameterRule(None, (None,), Q.BFLOAT16),  # no 1D
                Q.ParameterRule(None, None, fmt),
            ],
            execution=execution,
            notes="VQA validation set, small sample",
        )
        for fmt in formats()
    ]

    with out_path.open("w") as f:
        for xp in tqdm(experiments, "Experiment"):
            out = E.run_experiment(xp)
            print(json.dumps(out), file=f, flush=True)
