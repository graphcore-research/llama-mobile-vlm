from pathlib import Path

import block_formats.experiments as E

if __name__ == "__main__":
    E.fisher.Sweep("20250417-fisher").run(Path(f"out/fisher"))
