# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import dataclasses
import itertools as it
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import datasets
import torch
import tqdm
import transformers
from torch import Tensor, multiprocessing, nn

from .. import model_quantisation as M
from . import core, fisher


@dataclass
class Dataset:
    name: str
    tokens: Tensor  # (n_batch, batch_size, sequence_length - 1; int64)
    masks: Tensor  # (n_batch, batch_size; bool)
    bos_token_id: int
    topk_indices: Tensor  # (n_batch, batch_size, sequence_length, kl_topk; int64)
    topk_logp: Tensor  # (n_batch, batch_size, sequence_length, kl_topk; float32)

    def __repr__(self) -> str:
        return f"Dataset({self.name}, ({self.n_batch}, {self.batch_size}, {self.sequence_length}))"

    @property
    def device(self) -> torch.device:
        return self.tokens.device

    @property
    def n_batch(self) -> int:
        return self.tokens.shape[0]

    @property
    def batch_size(self) -> int:
        return self.tokens.shape[1]

    @property
    def sequence_length(self) -> int:
        return self.tokens.shape[2] + 1

    @property
    def kl_topk(self) -> int:
        return self.topk_indices.shape[-1]

    @staticmethod
    def _load_wikitext(
        split: tuple[str, ...], line_limit: int | None
    ) -> tuple[str, list[str]]:
        """Returns (name, lines)"""
        dataset_name = ("Salesforce/wikitext", "wikitext-103-raw-v1")
        data = [
            line
            for s in split
            for line in datasets.load_dataset(*dataset_name, split=s)["text"]
        ]
        if line_limit:
            data = data[:line_limit]
        return ":".join(dataset_name), data

    @staticmethod
    def _load_github_code(
        split: tuple[str, ...], line_limit: int | None
    ) -> tuple[str, list[str]]:
        """Returns (name, lines)"""
        assert line_limit is not None, "github-code is too big - specify a line_limit"
        assert split == ("train",)
        dataset_name = "codeparrot/github-code"
        data = [
            line["code"]
            for line in it.islice(
                datasets.load_dataset(
                    dataset_name, split="train", streaming=True, trust_remote_code=True
                ),
                line_limit,
            )
        ]
        return dataset_name, data

    WIKITEXT_DEFAULT = ("wikitext", ("validation", "test"))

    @classmethod
    def load(
        cls,
        model: transformers.PreTrainedModel,
        sequence_length: int,
        batch_size: int,
        kl_topk: int,
        sequence_limit: int | None = None,
        line_limit: int | None = None,
        seed: int = 120081,
        dataset: tuple[
            Literal["wikitext", "github-code"], tuple[str, ...]
        ] = WIKITEXT_DEFAULT,
        progress: bool = False,
    ) -> "Dataset":
        """Load and tokenize the dataset, then use the model to provide reference logits.

        dataset: ("wikitext", ("validation", "test"))
                 ("github-code", ("train",))
        """
        (device,) = set(p.device for p in model.parameters())

        if dataset[0] == "wikitext":
            dataset_name, data = cls._load_wikitext(dataset[1], line_limit)
        elif dataset[0] == "github-code":
            dataset_name, data = cls._load_github_code(dataset[1], line_limit)
        else:
            raise ValueError(f"Dataset {dataset[0]!r} not found")

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model.config._name_or_path
        )
        flat_tokens = [
            t
            for d in tqdm.tqdm(data, desc="tokenising", disable=not progress)
            for t in tokenizer(d, add_special_tokens=False).input_ids
        ]

        # Trim any final incomplete sequence and create sequences
        flat_tokens = flat_tokens[
            : len(flat_tokens) // (sequence_length - 1) * (sequence_length - 1)
        ]
        tokens = torch.tensor(flat_tokens, dtype=torch.int64, device=device).view(
            -1, sequence_length - 1
        )

        # Shuffle & truncate
        idx = torch.randperm(
            tokens.shape[0],
            generator=torch.Generator(device).manual_seed(seed),
            device=device,
        )
        tokens = tokens[idx]
        if sequence_limit is not None:
            tokens = tokens[:sequence_limit]
        n_sequence = tokens.shape[0]

        # Pad and batch
        tokens = nn.functional.pad(
            tokens, (0, 0, 0, -n_sequence % batch_size), value=tokenizer.eos_token_id
        ).view(-1, batch_size, sequence_length - 1)
        n_batch = tokens.shape[0]
        masks = (torch.arange(n_batch * batch_size, device=device) < n_sequence).view(
            n_batch, batch_size
        )

        # Run `model` to get topk_indices, topk_logp
        topk_indices = torch.zeros(
            (n_batch, batch_size, sequence_length, kl_topk),
            device=device,
            dtype=torch.int64,
        )
        topk_logp = torch.zeros(
            (n_batch, batch_size, sequence_length, kl_topk),
            device=device,
            dtype=torch.float32,
        )
        if kl_topk:
            with torch.no_grad():
                for tokens_, topk_indices_, topk_logp_ in zip(
                    tqdm.tqdm(tokens, desc="reference topk", disable=not progress),
                    topk_indices,
                    topk_logp,
                ):
                    logp_ = model(
                        nn.functional.pad(tokens_, (1, 0), value=tokenizer.bos_token_id)
                    ).logits.log_softmax(-1)
                    topk_logp_[...], topk_indices_[...] = logp_.topk(kl_topk, dim=-1)

        return cls(
            name=":".join((dataset_name, "-".join(dataset[1]))),
            tokens=tokens,
            masks=masks,
            bos_token_id=tokenizer.bos_token_id,
            topk_indices=topk_indices,
            topk_logp=topk_logp,
        )

    def batch_losses(self, model: nn.Module, index: int) -> dict[str, Tensor]:
        """Return the average (per-token) loss for each (non-masked) sequence in the batch.

        "cross_entropy" -- (n_sequence,)

        "kl_div" -- (n_sequence,)
        """
        tokens = self.tokens[index]
        mask = self.masks[index]
        logits = model(
            nn.functional.pad(tokens, (1, 0), value=self.bos_token_id)
        ).logits

        # Cross entropy, over sequence_length-1
        xent = (
            nn.functional.cross_entropy(
                logits[:, :-1].flatten(end_dim=-2),
                tokens.flatten(),
                reduction="none",
            )
            .view(tokens.shape[-2:])
            .mean(-1, dtype=torch.float32)[mask]
        )

        # KL divergence using the reference model's topk + tail
        topk_indices = self.topk_indices[index]
        topk_logp = self.topk_logp[index].float()
        model_topk_logp = logits.log_softmax(-1).gather(-1, topk_indices).float()

        # Add a contribution from the tail
        # Values very close to zero cause numerical issues & exploding KL,
        # so we clip the tail minimum
        tail_p = (1 - topk_logp.exp().sum(-1)).clip(min=1e-6)
        model_tail_p = (1 - model_topk_logp.exp().sum(-1)).clip(min=1e-6)
        tail_kl = tail_p * (tail_p.log() - model_tail_p.log())

        kl_div = (
            topk_logp.exp()
            .mul(topk_logp - model_topk_logp)
            .sum(-1)
            .add(tail_kl)
            .mean(-1)[mask]
        )
        return dict(cross_entropy=xent, kl_div=kl_div)

    def evaluate(self, model: transformers.PreTrainedModel) -> dict[str, float]:
        with torch.no_grad():
            losses = [self.batch_losses(model, i) for i in range(self.n_batch)]
            return {k: torch.concat([x[k] for x in losses]) for k in losses[0]}


# Tests


class Test:
    """Specifies a run or set of runs."""

    def to_config(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def args(
        self, model: core.RequantisableModel, data: Dataset
    ) -> list[dict[str, Any]]:
        return [{}]

    def run(self, model: core.RequantisableModel, data: Dataset) -> dict[str, Any]:
        raise NotImplementedError()


@dataclass
class Baseline(Test):
    type: str = "baseline"

    def run(self, model: core.RequantisableModel, data: Dataset) -> dict[str, Any]:
        return data.evaluate(model.model)


@dataclass
class _QuantiseModel(Test):
    fmt: M.FmtSpec
    error_weight: Literal["fisher", "parameter"] | None = None

    def to_config(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["fmt_str"] = str(self.fmt)
        return d

    def _quantise(
        self, model: nn.Module, error_weight: dict[str, Tensor] | None
    ) -> dict[str, Any]:
        raise NotImplementedError()

    def run(self, model: core.RequantisableModel, data: Dataset) -> dict[str, Any]:
        if self.error_weight is None:
            error_weight = None
        elif self.error_weight == "fisher":
            error_weight = fisher.fetch_fisher(
                model.model.config._name_or_path, model.device
            )
        elif self.error_weight == "parameter":
            error_weight = {k: v.abs() for k, v in model.model.named_parameters()}
        else:
            raise ValueError(f"Unexpected error_weight={self.error_weight}")
        log = self._quantise(model.model, error_weight)
        return dict(**log, **data.evaluate(model.model))


@dataclass
class QuantiseFixed(_QuantiseModel):
    type: str = "quantise_fixed"

    def _quantise(
        self, model: nn.Module, error_weight: dict[str, Tensor] | None
    ) -> dict[str, Any]:
        return M.quantise_2d_fixed(model, self.fmt, error_weight=error_weight)


@dataclass
class QuantiseVariable(_QuantiseModel):
    min_element_bits: float | None = None
    type: str = "quantise_variable"

    def _quantise(
        self, model: nn.Module, error_weight: dict[str, Tensor] | None
    ) -> dict[str, Any]:
        fisher_sum = fisher.fetch_fisher_sum(model.config._name_or_path)
        return M.quantise_2d_variable(
            model,
            self.fmt,
            fisher_sum=fisher_sum,
            min_element_bits=self.min_element_bits,
            error_weight=error_weight,
        )


@dataclass
class QuantiseHeuristic(_QuantiseModel):
    highp_add_bits: float = 2
    highp_names: tuple[str, ...] = ("embed_tokens", "lm_head")
    highp_first_layers: int = 2
    highp_last_layers: int = 2
    type: str = "quantise_heuristic"

    def _quantise(
        self, model: nn.Module, error_weight: dict[str, Tensor] | None
    ) -> dict[str, Any]:
        return M.quantise_2d_heuristic(
            model,
            self.fmt,
            highp_add_bits=self.highp_add_bits,
            highp_names=self.highp_names,
            highp_first_layers=self.highp_first_layers,
            highp_last_layers=self.highp_last_layers,
            error_weight=error_weight,
        )


@dataclass
class QuantiseEachParam:
    fmt: M.FmtSpec
    type: str = "quantise_each_param"

    def to_config(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["fmt_str"] = str(self.fmt)
        return d

    def args(
        self, model: core.RequantisableModel, data: Dataset
    ) -> list[dict[str, Any]]:
        return [
            dict(parameter=name)
            for name, p in model.model.named_parameters()
            if p.ndim == 2 and not any(p in M.DEFAULT_IGNORE for p in name.split("."))
        ]

    def run(
        self, model: core.RequantisableModel, data: Dataset, parameter: str
    ) -> dict[str, Any]:
        p = dict(model.model.named_parameters())[parameter]
        try:
            M.quantise_parameter_(p, self.fmt)
            return dict(**p._quantised, **data.evaluate(model.model))
        finally:
            model.reset_parameter(parameter)


@dataclass
class PerturbEachParam:
    scale: float
    distribution: str = "normal"
    type: str = "perturb_each_param"

    def to_config(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def args(
        self, model: core.RequantisableModel, data: Dataset
    ) -> list[dict[str, Any]]:
        return [
            dict(parameter=name)
            for name, p in model.model.named_parameters()
            if p.ndim == 2 and not any(p in M.DEFAULT_IGNORE for p in name.split("."))
        ]

    def run(
        self, model: core.RequantisableModel, data: Dataset, parameter: str
    ) -> dict[str, Any]:
        p = dict(model.model.named_parameters())[parameter]
        try:
            rms = p.float().square().mean().sqrt()
            p.data[...] += torch.randn_like(p).mul_(rms * self.scale)
            return dict(rms=rms.item(), **data.evaluate(model.model))
        finally:
            model.reset_parameter(parameter)


@dataclass
class Run:
    experiment: str
    test: Test
    model: str
    sequence_length: int = 4096
    kl_topk: int = 128
    batch_size: int = 1
    sequence_limit: int | None = None
    line_limit: int | None = None
    dataset: Literal["wikitext", "github-code"] = "wikitext"
    device: torch.device = core.FIELD_DEVICE
    type: str = "token_prediction"


class _Runner:
    def __init__(self):
        self.loaded_run = None
        self.model = None
        self.data = None

    def __call__(self, run: Run, progress: bool) -> None:
        if self.loaded_run is None or any(
            getattr(run, k) != getattr(self.loaded_run, k)
            for k in [
                "model",
                "sequence_length",
                "kl_topk",
                "batch_size",
                "sequence_limit",
                "device",
            ]
        ):
            self.model = self.data = None  # allow device memory to be freed
            self.model = core.RequantisableModel.load(
                run.model,
                device=run.device,
                dtype=torch.bfloat16,
            )
            self.data = Dataset.load(
                self.model.model,
                sequence_length=run.sequence_length,
                batch_size=run.batch_size,
                kl_topk=run.kl_topk,
                sequence_limit=run.sequence_limit,
                line_limit=run.line_limit,
                dataset={
                    "wikitext": Dataset.WIKITEXT_DEFAULT,
                    "github-code": ("github-code", ("train",)),
                }[run.dataset],
            )
            self.loaded_run = run

        self.model.reset()
        test_id = core.generate_id()
        for run_args in tqdm.tqdm(
            run.test.args(self.model, self.data), disable=not progress
        ):
            config = dataclasses.asdict(run)
            config["test"] = dict(**run.test.to_config(), **run_args)
            try:
                with core.Experiment(config) as experiment:
                    experiment.summary(
                        **run.test.run(self.model, self.data, **run_args),
                        test_id=test_id,
                    )
            except Exception:
                print(f"### Sweep run error for {config}", file=sys.stderr)
                traceback.print_exc()


_SWEEP_RUNNER: _Runner | None = None


def _sweep_init(queue: multiprocessing.Queue) -> None:
    # CUDA_VISIBLE_DEVICES seems better than using torch.device at
    # reusing the torch.compile cache between GPUs
    device = queue.get_nowait()
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    torch.set_num_threads(16)  # avoid CPU contention when sweeping
    global _SWEEP_RUNNER
    _SWEEP_RUNNER = _Runner()


def _sweep_run(run: Run) -> None:
    _SWEEP_RUNNER(run, progress=False)


def run_sweep(runs: Iterable[Run], processes: int | None = None) -> None:
    if processes is None:
        processes = torch.cuda.device_count() if torch.cuda.is_available() else 1

    if processes == 1:
        # Run directly in the host process (easier to debug)
        runner = _Runner()
        for run in runs:
            runner(run, progress=True)
    else:
        # Start subprocesses that "own" device IDs then use a pool to divide work
        queue = multiprocessing.Manager().Queue()
        for idx in range(processes):
            queue.put(
                (idx % torch.cuda.device_count()) if torch.cuda.is_available() else None
            )
        pool = multiprocessing.get_context("spawn").Pool(
            processes, _sweep_init, (queue,)
        )
        for run in runs:
            pool.apply_async(_sweep_run, (run,))
        pool.close()
        pool.join()
