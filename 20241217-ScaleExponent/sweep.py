import sys
from typing import Iterable
import os

import training.experiments as E
import training.quantisation as Q

if __name__ == "__main__":
    def formats() -> Iterable[Q.TensorFormat]:
        for efmt in ["E2M1", "E0M3", "E2M2", "E0M4"][int(sys.argv[1]):][:1]:
            for group_size in [8, 16, 32, 64, 128]:
                for smantissa in [0, 1, 2, 3, 4, 7]:
                    yield Q.group_scaling_format(
                        element_format=Q.parse(efmt),
                        grouping="input",
                        group_size=group_size,
                        scale_format=Q.FPFormat(8, smantissa, "to_inf"),
                    )

    experiments = [
        E.Experiment(
            name="202412-scaling-exponents",
            model="meta-llama/Llama-3.2-11B-Vision-Instruct",
            task=E.Task(
                name="outcompare",
                metrics=["entropy_rmse"],
            ),
            quantisation=[
                Q.ParameterRule(r".*positional_embedding", None, Q.BFLOAT16),
                Q.ParameterRule(None, (None, None), weight_fmt),
            ],
            execution=E.Execution(
                device="cuda",
                batch_size=8,
                wandb=True,
            ),
        )
        for weight_fmt in formats()
    ]
    os.environ["WANDB_SILENT"] = "true"
    print(f"Running {len(experiments)} experiments", file=sys.stderr)
    for n, xp in enumerate(experiments):
        print(f"[{n}/{len(experiments)}] {xp}", file=sys.stderr)
        E.run_experiment(xp)
