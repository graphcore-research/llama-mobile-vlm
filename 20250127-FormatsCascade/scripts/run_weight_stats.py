from pathlib import Path

import block_formats.experiments as E

if __name__ == "__main__":
    name = "20250312-weight-stats"
    E.WeightStatsSweep(name).run(Path(f"out/{name}.jsonl"))
