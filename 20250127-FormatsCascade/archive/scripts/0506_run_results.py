import sys
from typing import Iterable

import torch

import block_formats.experiments as E
import block_formats.experiments.token_prediction as ET
import block_formats.fit as F
import block_formats.quantisation as Q


def _main(step: float) -> Iterable[ET.Test]:
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


def _huffman(step: float) -> Iterable[ET.Test]:
    for element_bits in torch.arange(3, 5.01, step).tolist():
        yield ET.QuantiseFixed(
            F.Scaled(
                element_bits=element_bits,
                element_family="int",
                scale_format=Q.BFLOAT16,
                block_shape=(None, None),
                scaling="rms",
                compressor=("huffman", "optimal"),
            )
        )


def _fisher(step: float) -> Iterable[ET.Test]:
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


def _block_size() -> Iterable[ET.Test]:
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


def _scale_mantissa() -> Iterable[ET.Test]:
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


def _symmetry() -> Iterable[ET.Test]:
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


def _element_formats() -> Iterable[ET.Test]:
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
                    for scaling_match in ["search"] + (
                        ["moments"] if element_family != "lloyd_max" else []
                    ):
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
                        for error_weight in [None] + (
                            ["fisher"] if fmt.supports_error_weight else []
                        ):
                            yield ET.QuantiseFixed(fmt, error_weight=error_weight)


if __name__ == "__main__":
    MOD_ALL = E.MODELS
    MOD_LLAMA8B = ["meta-llama/Llama-3.1-8B"]
    MOD_NOT_LLAMA8B = [m for m in MOD_ALL if m not in MOD_LLAMA8B]

    s = []
    s.append(dict(name="baseline", tests=[ET.Baseline()], models=MOD_ALL))

    s.append(dict(name="main", tests=list(_main(0.25)), models=MOD_LLAMA8B))
    s.append(dict(name="main", tests=list(_main(1)), models=MOD_NOT_LLAMA8B))
    s.append(dict(name="huffman", tests=list(_huffman(0.25)), models=MOD_LLAMA8B))
    s.append(dict(name="fisher", tests=list(_fisher(0.25)), models=MOD_LLAMA8B))
    s.append(dict(name="fisher", tests=list(_fisher(1)), models=MOD_NOT_LLAMA8B))

    s.append(dict(name="blocksize", tests=list(_block_size()), models=MOD_ALL))
    s.append(dict(name="scalemantissa", tests=list(_scale_mantissa()), models=MOD_ALL))
    s.append(dict(name="symmetry", tests=list(_symmetry()), models=MOD_ALL))
    s.append(
        dict(name="elementformats", tests=list(_element_formats()), models=MOD_ALL)
    )

    for sweep in s:
        print(
            f"### {sweep['name']} ({len(sweep['models'])} x {len(sweep['tests'])})",
            file=sys.stderr,
        )
        ET.run_sweep(
            [
                ET.Run(f"20250506-results-{sweep['name']}", test, model)
                for model in sweep["models"]
                for test in sweep["tests"]
            ]
        )
