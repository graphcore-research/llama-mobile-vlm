import json
from datetime import datetime
from pathlib import Path

import transformers

from eval import vqa

model = transformers.MllamaForConditionalGeneration.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    torch_dtype="bfloat16",
    device_map="cuda",
)
processor = transformers.AutoProcessor.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision-Instruct"
)

path = (
    Path("out") / "vqa" / datetime.now().isoformat(timespec="seconds").replace(":", "-")
)
path.mkdir(parents=True, exist_ok=True)
print(f"Writing data to {path}")

n_examples = 1024
batch_size = 32

for task_name, task in vqa.TASKS.items():
    data = task.data(limit=n_examples)
    with open(path / f"{task_name}.jsonl", "w") as f:
        for o in vqa.evaluate(model, processor, task_name, data, batch_size):
            print(json.dumps(o), file=f, flush=True)
