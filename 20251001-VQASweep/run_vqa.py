"""
Run VQA on baseline, DC, and QAT (512 steps) on:
* element format: E0M4, E0M3, E0M2
* group size: (1, None), (1, 128), (1, 64)

VQA evaluation:
* accuracy: Standard calculation, one word answers
* accuracy-easy: Answer appears *anywhere* in the output
"""

import json
import subprocess
from pathlib import Path

import safetensors
import torch
import transformers
import wandb
import weight_formats.quantisation as Q
import weight_formats.quantisation_training as QT

from eval import vqa
from train import CHECKPOINT_PATH

# Set number of VQA examples
n_examples = 1024

# Set batch size
batch_size = 16


def log(results: list[dict], name: str, val_loss: float | None) -> None:
    out = {}
    if val_loss:
        out["validation_loss"] = val_loss
        print(f"{name} val loss: {val_loss}")
    for acc_type in ["accuracy", "accuracy_easy"]:
        accs = [x[acc_type] for x in results]
        acc = sum(accs) / len(accs)
        out[acc_type] = acc
        print(f"{name} {acc_type}: {acc}")

    out["results"] = results
    path = Path("out") / "vqa" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        print(path)
        json.dump(out, f, indent=4)


# QAT runs
api = wandb.Api()
qat_runs = list(api.runs("graphcore/llama-mobile", filters={"config.name": "sweep-v0"}))
model_name = qat_runs[0].config["model_name"]

# Baseline
processor = transformers.AutoProcessor.from_pretrained(model_name)
model = transformers.MllamaForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
results = list(
    vqa.evaluate(
        model, processor, data=vqa.VQA.data(limit=n_examples), batch_size=batch_size
    )
)
log(results, name="baseline", val_loss=None)

for run in qat_runs:
    name = run.name
    quantisation_config = run.config["quantisation"]

    val_loss = run.history(keys=["val/loss"], pandas=False)
    val_loss_dc, val_loss_qat = val_loss[0]["val/loss"], val_loss[-1]["val/loss"]

    # Direct cast
    QT.convert(
        model,
        Q.TensorFormat.load(quantisation_config["fmt"]),
        scaling_mode=quantisation_config["scaling_mode"],
        clip_gradient=quantisation_config["clip_gradient"],
        error_weight=None,
    )
    results = list(
        vqa.evaluate(
            model, processor, data=vqa.VQA.data(limit=n_examples), batch_size=batch_size
        )
    )
    log(results, name=f"{name}_dc", val_loss=val_loss_dc)

    # QAT
    s3_path = CHECKPOINT_PATH.format(name=name)
    path = Path("checkpoints") / f"{name}.safetensors"
    if not path.exists():
        subprocess.run(["aws", "s3", "cp", s3_path, path], check=True)
    QT.load_convert(model, safetensors.torch.load_file(path))
    results = list(
        vqa.evaluate(
            model, processor, data=vqa.VQA.data(limit=n_examples), batch_size=batch_size
        )
    )
    log(results, name=f"{name}", val_loss=val_loss_qat)
