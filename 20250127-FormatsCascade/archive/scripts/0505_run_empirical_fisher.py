from pathlib import Path

import block_formats.experiments as E

if __name__ == "__main__":
    E.fisher.Sweep("20250505-empirical-fisher", mode="empirical").run(
        Path(f"out/20250505-empirical-fisher")
    )
