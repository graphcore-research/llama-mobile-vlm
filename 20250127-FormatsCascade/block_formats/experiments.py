# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

"""Main experiments for block formats investigation"""

import dataclasses
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets
import safetensors.torch
import torch
import tqdm
import transformers
from torch import Tensor, nn, tensor

import wandb

from . import fit as F
from . import quantisation as Q
from . import sensitivity as S

PROJECT = "block-number-formats"
CODE_CHANGES = ("lut-bucketize", "rename-to-block", "sequence-losses")


### token_prediction


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
        tokens = tensor(flat_tokens, dtype=torch.int64, device=device).view(
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
        masks = (torch.arange(n_batch * batch_size) < n_sequence).view(
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


# Maps parameter name (or a default "") to quantisation format or fit spec
ModelFormats = dict[str, Q.TensorFormat | F.Scaled]


@dataclass
class RequantisableModel:
    """Wraps transformers.PreTrainedModel, storing original parameters on CPU,
    so that they can be restored when needed.
    """

    model: transformers.PreTrainedModel
    original_params: dict[str, nn.Parameter]

    @classmethod
    def load(
        cls, name: str, device: torch.device, dtype: torch.dtype
    ) -> "RequantisableModel":
        model = transformers.AutoModelForCausalLM.from_pretrained(
            name, device_map=device, torch_dtype=dtype
        )
        original_params = {
            k: v.to("cpu", copy=True) for k, v in model.state_dict().items()
        }
        return cls(model=model, original_params=original_params)

    @property
    def device(self) -> torch.device:
        (device,) = set(p.device for p in self.model.parameters())
        return device

    def reset(self) -> None:
        for name, p in self.model.state_dict().items():
            p[...] = self.original_params[name].to(p.device)

    def quantise(self, formats: ModelFormats) -> list[dict[str, Any]]:
        """Quantise parameters of the model, returning a log of the outcomes.

        formats -- dict[ParamName, Quantiser]

            Tries `formats[param_name] or formats[""]`, if neither is found, the parameter
            is unquantised.

            Quantiser can be a Q.TensorFormat, which is used directly or an F.Scaled which
            is first fitted to each tensor being quantised

        returns -- list[ParamRecord] -- records bits, quantisation error etc.
        """
        log = []
        for name, p in self.model.state_dict().items():
            if p.ndim == 2:
                p0 = self.original_params[name].to(p.device)
                fmt_or_fit = formats.get(name, formats.get(""))
                if fmt_or_fit:
                    if isinstance(fmt_or_fit, Q.TensorFormat):
                        fmt = fmt_or_fit
                    elif isinstance(fmt_or_fit, F.Scaled):
                        fmt = fmt_or_fit.fit(p0)
                    p[...] = fmt.quantise(p0)
                    log.append(
                        dict(
                            name=name,
                            quantised=True,
                            nelement=p.nelement(),
                            bits=fmt.count_bits_tensor(p),
                            rmse=(p - p0).float().pow(2).mean().sqrt().item(),
                            norm=p0.float().pow(2).mean().sqrt().item(),
                            fmt=dataclasses.asdict(fmt),
                            fmt_str=str(fmt),
                        )
                    )
                    continue
            log.append(
                dict(
                    name=name,
                    quantised=False,
                    nelement=p.nelement(),
                    bits=Q.TorchFormat(p.dtype).count_bits(p.shape),
                )
            )
        return log


TEST_MODELS = [
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B",
    "google/gemma-2-2b",
    "google/gemma-2-9b",
    "microsoft/phi-4",
]


@dataclass
class TokenPredictionSweep:
    experiment: str
    formats: list[ModelFormats]
    sequence_length: int = 4096
    kl_topk: int = 128
    batch_size: int = 1
    sequence_limit: int | None = None
    models: list[str] = dataclasses.field(default_factory=lambda: TEST_MODELS.copy())
    device: torch.device = dataclasses.field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    def run(self, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            raise ValueError(f"Output log {out} already exists - please delete first")
        with out.open("w") as outf:
            for i, model_name in enumerate(self.models):
                model = RequantisableModel.load(
                    model_name, device=self.device, dtype=torch.bfloat16
                )
                n_params = sum(p.nelement() for p in model.model.parameters())
                data = Dataset.load_wikitext(
                    model.model,
                    sequence_length=self.sequence_length,
                    batch_size=self.batch_size,
                    kl_topk=self.kl_topk,
                    sequence_limit=self.sequence_limit,
                )
                for j, mformats in enumerate(self.formats):
                    print(
                        f"-- model {i+1}/{len(self.models)}, format {j+1}/{len(self.formats)}",
                        file=sys.stderr,
                    )
                    config = self.__dict__.copy()
                    config["test"] = "token_prediction"
                    config["dataset"] = data.name
                    del config["models"]
                    config["model"] = model_name
                    del config["formats"]
                    config["format"] = {
                        k: dataclasses.asdict(v) for k, v in mformats.items()
                    }
                    config["format_str"] = {k: str(v) for k, v in mformats.items()}
                    config["device"] = config["device"].type
                    config["code_changes"] = CODE_CHANGES
                    wandb.init(
                        entity="graphcore",
                        project=PROJECT,
                        reinit=True,
                        config=config,
                    )
                    outcome = dict(
                        n_sequences=data.masks.sum().item(),
                        n_params=n_params,
                    )
                    try:
                        log = model.quantise(mformats)
                        metrics = data.evaluate(model.model)
                        metrics.update(
                            {f"{k}_mean": v.mean() for k, v in metrics.items()}
                        )
                        outcome.update(
                            params={d.pop("name"): d for d in log},
                            bits_per_param=sum(d["bits"] for d in log) / n_params,
                            **{k: v.tolist() for k, v in metrics.items()},
                        )
                    except Exception as exc:
                        print(repr(exc), file=sys.stderr)
                        outcome.update(
                            error=repr(exc), backtrace=traceback.format_exc()
                        )
                    finally:
                        print(
                            json.dumps(dict(**config, **outcome)), file=outf, flush=True
                        )
                        wandb.summary.update(outcome)
                        wandb.finish(1 if "error" in outcome else 0)
                del model
                del data


### weight_stats


def _mean_block_amax(t: Tensor, b: int) -> Tensor:
    t = t.flatten()
    return t[: b * (t.nelement() // b)].view(-1, b).abs().amax(1).mean()


def _scaled_hist(t: Tensor, bin_edges: Tensor, dim: tuple[int, ...] | None) -> Tensor:
    """Compute a histogram of elements, after being normalised by RMS."""
    return (
        torch.bucketize(
            t.div(t.pow(2).mean(dim=dim, keepdim=True).sqrt()).flatten().abs(),
            bin_edges,
        )
        .bincount(minlength=bin_edges.shape[0] + 1)
        .div(t.nelement())
    )


_STUDENTT_FIT_SCALE_THRESHOLD = 0.001
_STUDENTT_FIT_DF_VALUES = torch.cat(
    [torch.arange(1, 10, 0.5), torch.arange(10, 20, 2), torch.arange(20, 100 + 1, 10)]
).tolist()


def _studentt_fit_scale(
    t: Tensor, df: float, threshold: float = _STUDENTT_FIT_SCALE_THRESHOLD
) -> Tensor:
    """Compute maximum-likelhood fit of the scale of a zero-mean Student-T distribution to samples `t`.

    Stops when the relative change in scale is less than `threshold`.
    """
    weights = torch.ones_like(t)
    t2 = t.pow(2)
    last_scale = None
    while True:
        scale = (weights * t2).mean().sqrt()
        weights = (df + 1) * scale.pow(2) / (t2 + df * scale.pow(2))
        if last_scale is not None and ((scale - last_scale).abs() / scale) < threshold:
            break
        last_scale = scale
    return scale


def _studentt_fit(
    t: Tensor,
    scale_threshold: float = _STUDENTT_FIT_SCALE_THRESHOLD,
    dfs: list[float] = _STUDENTT_FIT_DF_VALUES,
) -> tuple[Tensor, Tensor]:
    """Compute maximum-likelhood fit of (df, scale) of a zero-mean Student-T distribution to samples `t`.

    returns (df, scale)
    """
    best_log_likelihood = tensor(-torch.inf, device=t.device)
    best_params = None
    for df in torch.tensor(dfs, device=t.device):
        scale = _studentt_fit_scale(t, df, threshold=scale_threshold)
        log_likelihood = (
            torch.distributions.StudentT(df=df, scale=scale).log_prob(t).mean()
        )
        if log_likelihood > best_log_likelihood:
            best_params = (df, scale)
            best_log_likelihood = log_likelihood
    return best_params


def _dist_fit_stats(t: Tensor) -> dict[str, Any]:
    dist_args = []
    dist_args.append((torch.distributions.Normal, dict(scale=t.std(unbiased=False))))
    dist_args.append((torch.distributions.Laplace, dict(scale=t.abs().mean())))
    t_df, t_scale = _studentt_fit(t)
    dist_args.append((torch.distributions.StudentT, dict(df=t_df, scale=t_scale)))
    return {
        dist.__name__: dict(
            log_likelihood=dist(loc=tensor(0.0, device=t.device), **args)
            .log_prob(t)
            .mean()
            .item(),
            **{k: v.item() for k, v in args.items()},
        )
        for dist, args in dist_args
    }


def tensor_stats(w: Tensor) -> dict[str, Any]:
    with torch.no_grad():
        w = w.float()
        rm2 = w.pow(2).mean().sqrt()
        hist_bins = torch.arange(1, 20 + 1, device=w.device)
        block_sizes = 2 ** torch.arange(0, 1 + int(tensor(w.nelement()).log2().floor()))
        return dict(
            shape=tuple(w.shape),
            # Moments
            mean=w.mean().item(),
            std=w.std(correction=0).item(),
            rm2=rm2.item(),
            rm4=w.div(rm2).pow_(4).mean().pow(1 / 4).mul(rm2).item(),
            # Maxima
            max=w.abs().amax().item(),
            block_max=[_mean_block_amax(w, b).item() for b in block_sizes],
            block_max_shuffled=[
                _mean_block_amax(Q.shuffle(w), b).item() for b in block_sizes
            ],
            # Histograms
            hist=_scaled_hist(w, hist_bins, dim=None).tolist(),
            channel_hist=[
                _scaled_hist(
                    w, hist_bins, dim=tuple(d for d in range(w.ndim) if d != dim)
                ).tolist()
                for dim in range(w.ndim)
            ],
            # Distributions
            fit=_dist_fit_stats(w),
        )


@dataclass
class WeightStatsSweep:
    experiment: str
    models: list[str] = dataclasses.field(default_factory=lambda: TEST_MODELS.copy())
    device: torch.device = dataclasses.field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    def run(self, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            raise ValueError(f"Output log {out} already exists - please delete first")
        with out.open("w") as outf:
            for i, model_name in enumerate(self.models):
                print(f"-- model {i+1}/{len(self.models)}", file=sys.stderr)
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.bfloat16
                )
                config = self.__dict__.copy()
                config["test"] = "weight_stats"
                del config["models"]
                config["model"] = model_name
                config["device"] = config["device"].type
                config["code_changes"] = CODE_CHANGES
                wandb.init(
                    entity="graphcore",
                    project=PROJECT,
                    reinit=True,
                    config=config,
                )
                outcome = dict(
                    weight_stats={
                        name: tensor_stats(p.to(self.device))
                        for name, p in tqdm.tqdm(
                            list(model.state_dict().items()), desc=model_name
                        )
                    }
                )
                print(json.dumps(dict(**config, **outcome)), file=outf, flush=True)
                wandb.summary.update(outcome)
                wandb.finish()
                del model


# empirical_fisher


def empirical_diag_fisher(
    data: Dataset,
    model: nn.Module,
    loss: str = "cross_entropy",
    progress: bool = False,
) -> dict[str, Tensor]:
    """Compute the diagonal of the empirical Fisher information for Linear/Embedding weight parameters."""

    param_to_name = {}  # Handle parameter sharing
    for name, p in model.named_parameters():
        p.requires_grad_(False)  # Save memory by skipping parameter gradients
        param_to_name[p] = name
    S.wrap(model)
    try:
        for index in tqdm.tqdm(
            list(range(data.n_batch)), desc="fisher", disable=not progress
        ):
            losses = data.batch_losses(model, index)
            # Compute as a sum - scaling by sequence_length to cancel the mean-over-sequence
            losses[loss].backward(torch.full_like(losses[loss], data.sequence_length))
        results = {}
        for module in model.modules():
            if isinstance(module, S.Wrapper):
                name = param_to_name[module.wrapped.weight]
                # Convert to a mean over batch and sequence
                grad_weight_sq = module.grad_weight_sq / (
                    data.masks.sum() * data.sequence_length
                )
                if name in results:
                    results[name] += grad_weight_sq
                else:
                    results[name] = grad_weight_sq
        return results
    finally:
        S.unwrap(model)


@dataclass
class EmpiricalFisherSweep:
    sequence_length: int = 4096
    sequence_limit: int | None = 1024
    line_limit: int | None = int(1e5)
    batch_size: int = 1
    models: list[str] = dataclasses.field(default_factory=lambda: TEST_MODELS.copy())
    device: torch.device = dataclasses.field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    def run(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        for model_name in self.models:
            print(model_name, file=sys.stderr)
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map=self.device
            )
            data = Dataset.load_wikitext(
                model,
                sequence_length=self.sequence_length,
                sequence_limit=self.sequence_limit,
                line_limit=self.line_limit,
                batch_size=self.batch_size,
                kl_topk=0,
                split="train",
                progress=True,
            )
            sensitivity = empirical_diag_fisher(
                data, model, loss="cross_entropy", progress=True
            )
            safetensors.torch.save_file(
                sensitivity, out / f"{model_name.replace('/', '--')}.safetensors"
            )
            del model
