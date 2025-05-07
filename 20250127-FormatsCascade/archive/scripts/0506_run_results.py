import sys
from typing import Iterable

import torch

import block_formats.experiments as E
import block_formats.experiments.token_prediction as ET
import block_formats.fit as F
import block_formats.quantisation as Q


def _tests_main(step: float) -> Iterable[ET.Test]:
    for element_bits in torch.arange(3, 5.01, step).tolist():
        for element_family, compressor, args in [
            ("int", "optimal", {}),
            ("int", None, {}),
            ("t", None, {}),
        ]:
            for mode_args in (
                [dict(mode="symmetric"), dict(mode="asymmetric")]
                if element_family in ["normal", "laplace", "t"]
                or (element_family, compressor) == ("int", None)
                else [{}]
            ):
                for scaling in ["rms", "absmax"]:
                    for block_shape in [(None, None), (1, None)] + [
                        (1, b) for b in [64, 128]
                    ]:
                        for sparse_ratio in [0, 2**-7, 2**-10]:
                            yield ET.QuantiseFixed(
                                F.Scaled(
                                    element_bits=element_bits,
                                    element_family=element_family,
                                    scale_format=Q.BFLOAT16,
                                    block_shape=block_shape,
                                    scaling=scaling,
                                    sparse_format=Q.BFLOAT16,
                                    sparse_ratio=sparse_ratio,
                                    compressor=compressor,
                                    args=dict(**args, **mode_args),
                                )
                            )


def _tests_fisher(step: float) -> Iterable[ET.Test]:
    for element_bits in torch.arange(3, 5.01, step).tolist():
        for element_family, compressor, scaling, block_shape, sparse_ratio in [
            ("int", "optimal", "rms", (None, None), 0),
            ("t", None, "absmax", (1, 128), 0),
            ("t", None, "rms", (None, None), 2**-10),
        ]:
            for mode_args in (
                [dict(mode="symmetric"), dict(mode="asymmetric")]
                if element_family in ["normal", "laplace", "t"]
                or (element_family, compressor) == ("int", None)
                else [{}]
            ):
                fmt = F.Scaled(
                    element_bits=element_bits,
                    element_family=element_family,
                    scale_format=Q.BFLOAT16,
                    block_shape=block_shape,
                    scaling=scaling,
                    sparse_format=Q.BFLOAT16,
                    sparse_ratio=sparse_ratio,
                    compressor=compressor,
                    args=dict(**mode_args),
                )
                for cls in [ET.QuantiseVariable, ET.QuantiseHeuristic]:
                    yield cls(fmt)
                if element_family == "t":
                    for cls in [
                        ET.QuantiseFixed,
                        ET.QuantiseVariable,
                        ET.QuantiseHeuristic,
                    ]:
                        for error_weight in ["fisher", "parameter"]:
                            yield cls(fmt, error_weight=error_weight)


def _tests_block_size() -> Iterable[ET.Test]:
    for element_bits in torch.arange(3, 5.01, 1).tolist():
        for block_size in [16, 32, 64, 128, 256]:
            fmt = F.Scaled(
                element_bits=element_bits - 16 / block_size,
                element_family="t",
                scale_format=Q.BFLOAT16,
                block_shape=(1, block_size),
                scaling="absmax",
            )
            yield ET.QuantiseFixed(fmt)


def _tests_scale_mantissa() -> Iterable[ET.Test]:
    for element_bits in torch.arange(3, 5.01, 1).tolist():
        for mbits in range(0, 8):
            fmt = F.Scaled(
                element_bits=element_bits - (mbits + 8) / 128,
                element_family="t",
                scale_format=Q.FPFormat(7, mbits, "to_inf"),
                block_shape=(1, 128),
                scaling="absmax",
            )
            yield ET.QuantiseFixed(fmt)


def _tests_symmetry() -> Iterable[ET.Test]:
    for element_bits in torch.arange(3, 5.01, 1).tolist():
        for element_family in ["int", "t"]:
            for mode, scaling in [
                ("asymmetric", "absmax"),
                ("symmetric", "absmax"),
                ("asymmetric", "signmax"),
            ]:
                fmt = F.Scaled(
                    element_bits=element_bits,
                    element_family=element_family,
                    scale_format=Q.BFLOAT16,
                    block_shape=(1, 128),
                    scaling=scaling,
                    args=dict(mode=mode),
                )
                yield ET.QuantiseFixed(fmt)


def _tests_element_formats() -> Iterable[ET.Test]:
    for element_bits in torch.arange(3, 5.01, 1).tolist():
        for scaling, block_shape, sparse_ratio in [
            ("absmax", (1, 128), 0),
            ("rms", (None, None), 2**-10),
        ]:
            for element_family, args in [
                ("int", {}),
                ("fp", dict(exponent_bits=2)),
                ("normal", {}),
                ("laplace", {}),
                ("t", {}),
                ("t", dict(df=30)),
                ("lloyd_max", {}),
            ]:
                for mode_args in (
                    [dict(mode="symmetric"), dict(mode="asymmetric")]
                    if element_family in ["int", "normal", "laplace", "t"]
                    else [{}]
                ):
                    for scaling_match in ["search"] + (["moments"] if element_family != "lloyd_max" else []):
                        fmt = F.Scaled(
                            element_bits=element_bits,
                            element_family=element_family,
                            scale_format=Q.BFLOAT16,
                            block_shape=block_shape,
                            scaling=scaling,
                            scaling_match=scaling_match,
                            sparse_format=Q.BFLOAT16,
                            sparse_ratio=sparse_ratio,
                            args=dict(**args, **mode_args),
                        )
                        for error_weight in [None] + (["fisher"] if fmt.supports_error_weight else []):
                            yield ET.QuantiseFixed(fmt, error_weight=error_weight)


if __name__ == "__main__":
    MODELS_ALL = E.MODELS
    MODELS_LLAMA8B = ["meta-llama/Llama-3.1-8B"]
    MODELS_NOT_GEMMA = [m for m in MODELS_ALL if "gemma" not in m]
    MODELS_NOT_GEMMA_OR_LLAMA8B = [m for m in MODELS_NOT_GEMMA if m not in MODELS_LLAMA8B]

    sweeps = []
    sweeps.append(dict(name="baseline", tests=[ET.Baseline()], models=MODELS_ALL))

    sweeps.append(dict(name="main", tests=list(_tests_main(0.25)), models=MODELS_LLAMA8B))
    sweeps.append(dict(name="main", tests=list(_tests_main(1)), models=MODELS_NOT_GEMMA_OR_LLAMA8B))

    sweeps.append(dict(name="fisher", tests=list(_tests_fisher(0.25)), models=MODELS_LLAMA8B))
    sweeps.append(dict(name="fisher", tests=list(_tests_fisher(1)), models=MODELS_NOT_GEMMA_OR_LLAMA8B))

    sweeps.append(dict(name="blocksize", tests=list(_tests_block_size()), models=MODELS_NOT_GEMMA))
    sweeps.append(dict(name="scalemantissa", tests=list(_tests_scale_mantissa()), models=MODELS_NOT_GEMMA))
    sweeps.append(dict(name="symmetry", tests=list(_tests_symmetry()), models=MODELS_NOT_GEMMA))
    sweeps.append(dict(name="elementformats", tests=list(_tests_element_formats()), models=MODELS_NOT_GEMMA))

    for sweep in sweeps:
        print(f"### {sweep['name']} ({len(sweep['models'])} x {len(sweep['tests'])})", file=sys.stderr)
        ET.run_sweep(
            [
                ET.Run(f"20250506-results-{sweep['name']}", test, model)
                for model in sweep['models']
                for test in sweep['tests']
            ]
        )
