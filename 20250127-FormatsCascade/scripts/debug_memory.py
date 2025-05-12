import torch
from pathlib import Path

import block_formats.experiments as E

if __name__ == "__main__":
    torch.cuda.memory._record_memory_history(max_entries=100000)
    try:
        model = E.RequantisableModel.load(
            "meta-llama/Llama-3.2-1B",
            torch.device("cuda"),
            torch.bfloat16,
        )
        data = E.token_prediction.Dataset.load(model.model, 4096, 1, 0, 4)
        with E.fisher.activation_checkpointing_enabled(model.model):
            result = E.fisher.diag_fisher(
                data, model.model, mode="single_sample", progress=True
            )
    finally:
        out = Path("out/memory")
        out.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(
            f"{out}/{model.model.config._name_or_path.replace('/', '--')}.pickle"
        )
        torch.cuda.memory._record_memory_history(enabled=None)
