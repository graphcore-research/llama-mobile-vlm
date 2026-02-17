import json
from pathlib import Path
from typing import Any

import wandb
from wandb.apis.public.runs import Run

from utility import LOCAL_DATA_PATH, LLAMA_PROMPT_TEMPLATES

from eval import vqa


def get_task_outputs(run: Run) -> dict[str, Any]:
    artifacts = [a for a in run.logged_artifacts() if a.type == "task_outputs"]
    assert len(artifacts) == 1, "Found none or multiple matches"
    a = artifacts[0]
    a_dir = a.download(
        Path(LOCAL_DATA_PATH).parent / f"artifacts/{run.name}/task_outputs"
    )
    out = {}
    for path in Path(a_dir).rglob("*.jsonl"):
        with path.open() as f:
            out[path.stem] = [json.loads(line) for line in f]
    return out


api = wandb.Api()
runs = api.runs(
    "graphcore/llama-mobile",
    filters={"config.name": "sweep-05-02-26", "state": "finished"},
)

for run in runs:
    task_outputs = get_task_outputs(run)
    results = {}

    for task_config in run.config["evaluation"]["tasks"]:
        eval_out = []
        task_name, n_examples = task_config["name"], task_config["n_examples"]
        task = vqa.TASKS[task_name]
        data = task.data(limit=n_examples)
        batch = task.prepare_batch(
            data, system_template=LLAMA_PROMPT_TEMPLATES["instruct"]
        )
        eval_out = [
            task.evaluate_prediction(x["output"], y, include_relaxed_metrics=True)
            for x, y in zip(task_outputs[task_name], batch.answers)
        ]
        results[task_name] = {}
        for metric in eval_out[0]:
            o = [x[metric] for x in eval_out]
            results[task_name][metric] = sum(o) / len(o)

    for task_name in results:
        run.summary[f"downstream.{task_name}.accuracy_relaxed"] = results[task_name][
            "accuracy_relaxed"
        ]

    run.summary.update()
