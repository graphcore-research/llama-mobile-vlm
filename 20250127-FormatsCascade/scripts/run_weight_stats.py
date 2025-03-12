from pathlib import Path

import block_formats.experiments as E

if __name__ == "__main__":
    name = "20250312-weight-stats"
    # E.run_weight_stats(E.StatsExperiment(name), Path(f"out/{name}.jsonl"))
    E.run_weight_stats(
        E.StatsExperiment(name, models=E.TEST_MODELS[-2:]), Path(f"out/{name}.jsonl")
    )
