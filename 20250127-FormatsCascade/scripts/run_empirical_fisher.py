from pathlib import Path

import block_formats.experiments as E

if __name__ == "__main__":
    E.EmpiricalFisherSweep(models=E.TEST_MODELS[:2]).run(Path(f"out/efisher"))
