from pathlib import Path

import format_experiments as E

if __name__ == "__main__":
    name = "20250221-weight-stats"
    E.run_weight_stats(E.StatsExperiment(name), Path(f"out/{name}.jsonl"))
