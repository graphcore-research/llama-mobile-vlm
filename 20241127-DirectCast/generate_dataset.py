from training.eval import outcompare
import torch

model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

ds = outcompare.generate_dataset(
    model_name,
    prompt_length=5,
    completion_length=64,
    batch_size=16,
    dtype=torch.bfloat16,
)
ds.save(f"data/{model_name.replace('meta-llama/', '').lower()}.pt")
