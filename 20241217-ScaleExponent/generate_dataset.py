import torch

import training.eval.outcompare

model = "meta-llama/Llama-3.2-11B-Vision-Instruct"

data = training.eval.outcompare.generate_dataset(
    model,
    prompt_length=5,
    completion_length=64,
    batch_size=8,
    dtype=torch.bfloat16,
    limit=None,
)
data.save(f"data/{model.split('/')[-1].lower()}.pt")
