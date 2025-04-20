# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import dataclasses
from dataclasses import dataclass
from typing import Any

import datasets
import torch
import tqdm
import transformers
from torch import Tensor, nn

from .. import model_quantisation as M
from . import core


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

    @classmethod
    def load_wikitext(
        cls,
        model: transformers.PreTrainedModel,
        sequence_length: int,
        batch_size: int,
        kl_topk: int,
        sequence_limit: int | None = None,
        line_limit: int | None = None,
        seed: int = 120081,
        split: str = "test",
        progress: bool = False,
    ) -> "Dataset":
        """Load and tokenize the dataset, then use the model to provide reference logits."""

        dataset_name = ("Salesforce/wikitext", "wikitext-103-raw-v1")
        (device,) = set(p.device for p in model.parameters())
        data = datasets.load_dataset(*dataset_name, split=split)["text"]
        if line_limit:
            data = data[:line_limit]
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
            name=":".join(dataset_name + (split,)),
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


@dataclass
class Baseline:
    type: str = "baseline"

    def to_config(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def run(
        self, model: core.RequantisableModel, data: Dataset, progress: bool
    ) -> dict[str, Any]:
        return data.evaluate(model.model)


@dataclass
class QuantiseFixed:
    fmt: M.FmtSpec
    type: str = "quantise_fixed"

    def to_config(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["fmt_str"] = str(self.fmt)
        return d

    def run(
        self, model: core.RequantisableModel, data: Dataset, progress: bool
    ) -> dict[str, Any]:
        log = M.quantise_2d_fixed_(model.model, self.fmt)
        return dict(**log, **data.evaluate(model.model))


@dataclass
class QuantiseEachParam:
    fmt: M.FmtSpec
    type: str = "quantise_each_param"

    def to_config(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["fmt_str"] = str(self.fmt)
        return d

    def run(
        self, model: core.RequantisableModel, data: Dataset, progress: bool
    ) -> dict[str, Any]:
        results = {}
        for name, param in tqdm.tqdm(
            list(model.model.named_parameters()), disable=not progress
        ):
            if param.ndim == 2:
                M.quantise_parameter_(param, self.fmt)
                results[name] = dict(**param._quantised, **data.evaluate(model.model))
                model.reset_parameter(name)
        return results


@dataclass
class PerturbEachParam:
    scale: float
    distribution: str = "normal"
    type: str = "perturb_each_param"

    def to_config(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def run(
        self, model: core.RequantisableModel, data: Dataset, progress: bool
    ) -> dict[str, Any]:
        results = {}
        for name, param in tqdm.tqdm(
            list(model.model.named_parameters()), disable=not progress
        ):
            if param.ndim == 2:
                norm = param.float().square().mean().sqrt()
                param.data[...] += torch.randn_like(param).mul_(norm * self.scale)
                results[name] = dict(**data.evaluate(model.model), norm=norm.item())
                model.reset_parameter(name)
        return results


@dataclass
class Sweep:
    experiment: str
    test: list[QuantiseFixed | QuantiseEachParam | PerturbEachParam]  # sweep
    model: list[str] = core.FIELD_MODELS  # sweep
    sequence_length: int = 4096
    kl_topk: int = 128
    batch_size: int = 1
    sequence_limit: int | None = None
    device: torch.device = core.FIELD_DEVICE
    type: str = "token_prediction"

    def run(self, progress: bool = True) -> None:
        for mconfig in core.iter_dict_product(
            self.__dict__, "model", progress=progress
        ):
            model = core.RequantisableModel.load(
                mconfig["model"], device=self.device, dtype=torch.bfloat16
            )
            data = Dataset.load_wikitext(
                model.model,
                sequence_length=self.sequence_length,
                batch_size=self.batch_size,
                kl_topk=self.kl_topk,
                sequence_limit=self.sequence_limit,
            )
            mconfig["dataset"] = data.name
            for tconfig in core.iter_dict_product(mconfig, "test", progress=progress):
                test = tconfig.pop("test")
                tconfig["test"] = test.to_config()
                with core.Experiment(tconfig) as experiment:
                    model.reset()
                    experiment.summary(**test.run(model, data, progress=progress))
            del model
            del data
