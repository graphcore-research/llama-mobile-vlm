from pathlib import Path
from typing import Iterable

import format_experiments as E
import quantisation as Q
import transformers

transformers.utils.logging.disable_progress_bar()

if __name__ == "__main__":

    def _formats() -> Iterable[Q.TensorFormat]:
        scale_format = Q.BFLOAT16
        for ebits in [2, 3, 4, 5, 6]:
            # Group-scaled formats
            for group_size in [16, 32, 64, 128, 256]:
                for element_format, scaling in [
                    # Int
                    (Q.IntFormat(ebits), "absmax"),
                    (Q.IntFormat(ebits), "signmax"),
                    # Float
                    (
                        Q.FPFormat(2, ebits - 3, "nearest") if ebits >= 3 else None,
                        "absmax",
                    ),
                    (
                        Q.FPFormat(3, ebits - 4, "nearest") if ebits >= 4 else None,
                        "absmax",
                    ),
                    (
                        Q.FPFormat(4, ebits - 5, "nearest") if ebits >= 5 else None,
                        "absmax",
                    ),
                    # Special
                    (Q.NF4 if ebits == 4 else None, "absmax"),
                    (Q.crp_trunc_gauss(ebits, group_size, True), "absmax"),
                    (Q.crp_trunc_laplace(ebits, group_size, True), "absmax"),
                    (Q.crp_trunc_gauss(ebits, group_size, False), "signmax"),
                    (Q.crp_trunc_laplace(ebits, group_size, False), "signmax"),
                ]:
                    if element_format is not None:
                        yield Q.LinearScalingFormatV2(
                            element_format=element_format,
                            scale_format=scale_format,
                            group_shape=(1, group_size),
                            scaling=scaling,
                        )

            # Channel-scaled formats
            for scaling in ["rms", "absmax"]:
                for element_format in [
                    # Int
                    Q.IntFormat(ebits) if scaling == "absmax" else None,
                    # Float
                    Q.FPFormat(2, ebits - 3, "nearest") if ebits >= 3 else None,
                    Q.FPFormat(3, ebits - 4, "nearest") if ebits >= 4 else None,
                    Q.FPFormat(4, ebits - 5, "nearest") if ebits >= 5 else None,
                    # Special
                    Q.NF4 if ebits == 4 and scaling == "absmax" else None,
                    Q.crp_gauss(ebits),
                    Q.crp_laplace(ebits),
                    Q.crp_t(ebits, 4.5),
                ]:
                    if element_format is not None:
                        yield Q.LinearScalingFormatV2(
                            element_format=element_format,
                            scale_format=scale_format,
                            group_shape=(1, None),
                            scaling=scaling,
                        )

            # Tensor-scaled formats
            for scaling in ["rms", "absmax"]:
                for element_format in [
                    # Int
                    Q.IntFormat(ebits) if scaling == "absmax" else None,
                    # Float
                    Q.FPFormat(2, ebits - 3, "nearest") if ebits >= 3 else None,
                    Q.FPFormat(3, ebits - 4, "nearest") if ebits >= 4 else None,
                    Q.FPFormat(4, ebits - 5, "nearest") if ebits >= 5 else None,
                    # Special
                    Q.crp_gauss(ebits),
                    Q.crp_laplace(ebits),
                    Q.crp_t(ebits, 4.5),
                ]:
                    if element_format is not None:
                        yield Q.LinearScalingFormatV2(
                            element_format=element_format,
                            scale_format=scale_format,
                            group_shape=(None, None),
                            scaling=scaling,
                        )

    # for format in _formats():
    #     print(format)
    # print(sum(1 for _ in _formats()))
    name = "20250219-initial-sweep-v2"
    E.run_sweep(E.Sweep(name, formats=list(_formats())), Path(f"out/{name}.jsonl"))
