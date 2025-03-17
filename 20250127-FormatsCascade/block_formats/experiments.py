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
import tqdm
import torch
import transformers
from torch import Tensor, tensor
import wandb

from . import quantisation as Q

CODE_CHANGES = ("lut-bucketize", "rename-to-block")


### token_prediction


@dataclass
class Dataset:
    name: str
    tokens: Tensor  # (n_batch, batch_size, sequence_length - 1; int64)
    masks: Tensor  # (n_batch, batch_size, sequence_length - 1; bool)
    bos_token_id: int
    topk_indices: Tensor  # (n_batch, batch_size, sequence_length, kl_topk; int64)
    topk_logp: Tensor  # (n_batch, batch_size, sequence_length, kl_topk; float32)

    @property
    def device(self) -> torch.device:
        return self.tokens.device

    @property
    def sequence_length(self) -> int:
        return self.tokens.shape[2] + 1

    @property
    def kl_topk(self) -> int:
        return self.topk_indices.shape[-1]

    @classmethod
    def load_wikitext2(
        cls,
        model: transformers.PreTrainedModel,
        sequence_length: int,
        batch_size: int,
        kl_topk: int,
        token_limit: int | None = None,
    ) -> "Dataset":
        """Load and tokenize the dataset, then use the model to provide reference logits.

        (Note topk_indices and topk_logp will be filled with dummy data.)
        """
        dataset_name = ("Salesforce/wikitext", "wikitext-2-raw-v1")
        (device,) = set(p.device for p in model.parameters())
        data = datasets.load_dataset(*dataset_name, split="test")["text"]
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model.config._name_or_path
        )
        tokens = [
            t for d in data for t in tokenizer(d, add_special_tokens=False).input_ids
        ]
        if token_limit is not None:
            tokens = tokens[:token_limit]
        npad = -len(tokens) % ((sequence_length - 1) * batch_size)
        tokens_batched = torch.nn.functional.pad(
            tensor(tokens, dtype=torch.int64, device=device),
            (0, npad),
            value=tokenizer.eos_token_id,
        ).view(-1, batch_size, sequence_length - 1)

        masks = torch.nn.functional.pad(
            torch.ones(len(tokens), dtype=torch.bool, device=device), (0, npad)
        ).view(-1, batch_size, sequence_length - 1)

        n_batch = tokens_batched.shape[0]
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
        with torch.no_grad():
            for tokens_, topk_indices_, topk_logp_ in zip(
                tokens_batched, topk_indices, topk_logp
            ):
                logp_ = model(
                    torch.nn.functional.pad(
                        tokens_, (1, 0), value=tokenizer.bos_token_id
                    )
                ).logits.log_softmax(-1)
                topk_logp_[...], topk_indices_[...] = logp_.topk(kl_topk, dim=-1)

        return cls(
            name=":".join(dataset_name),
            tokens=tokens_batched,
            masks=masks,
            bos_token_id=tokenizer.bos_token_id,
            topk_indices=topk_indices,
            topk_logp=topk_logp,
        )


def evaluate_model(
    data: Dataset, model: transformers.PreTrainedModel
) -> dict[str, float]:
    with torch.no_grad():
        cross_entropy_sum = tensor(0.0, device=data.device)
        kl_sum = tensor(0.0, device=data.device)
        kl_count = tensor(0, device=data.device, dtype=torch.int64)
        for tokens, mask, topk_indices, topk_logp in zip(
            data.tokens, data.masks, data.topk_indices, data.topk_logp
        ):
            logits = model(
                torch.nn.functional.pad(tokens, (1, 0), value=data.bos_token_id)
            ).logits
            cross_entropy_sum += (
                torch.nn.functional.cross_entropy(
                    logits[:, :-1].flatten(end_dim=-2),
                    tokens.flatten(),
                    reduction="none",
                )
                .float()
                .mul(mask.flatten())
                .sum()
            )

            # KL divergence can use the full sequence length (vs xent, which
            # uses `sequence_length - 1`)
            kl_mask = torch.nn.functional.pad(mask, (1, 0), value=True)
            model_topk_logp = logits.log_softmax(-1).gather(-1, topk_indices)

            # Values very close to zero cause numerical issues & exploding KL,
            # so we clip the tail minimum
            res_p = (1 - topk_logp.exp().sum(-1)).clip(min=1e-6)
            model_res_p = (1 - model_topk_logp.exp().sum(-1)).clip(min=1e-6)
            res_kl = res_p * (res_p.log() - model_res_p.log())

            kl_sum += (
                topk_logp.exp()
                .mul(topk_logp - model_topk_logp)
                .sum(-1)
                .add(res_kl)
                .mul(kl_mask)
                .sum()
            )
            kl_count += kl_mask.sum()

        cross_entropy = cross_entropy_sum / data.masks.sum()
        kl_div = kl_sum / kl_count
        return dict(cross_entropy=cross_entropy.item(), kl_div=kl_div.item())


@dataclass
class RequantisableModel:
    model: transformers.PreTrainedModel
    original_params: dict[str, torch.nn.Parameter]

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

    def quantise(self, format: Q.TensorFormat) -> list[dict[str, str | bool | float]]:
        log = []
        for name, p in self.model.state_dict().items():
            if p.ndim == 2:
                p0 = self.original_params[name].to(p.device)
                p[...] = format.quantise(p0)
                log.append(
                    dict(
                        name=name,
                        quantised=True,
                        rmse=(p - p0).float().pow(2).mean().sqrt().item(),
                        snr=Q.snr(p0, p).item(),
                        bits=format.count_bits(p.shape),
                    )
                )
            else:
                log.append(
                    dict(
                        name=name,
                        quantised=False,
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
class Sweep:
    experiment: str
    formats: list[Q.TensorFormat]
    sequence_length: int = 4096
    kl_topk: int = 128
    batch_size: int = 1
    data_max_tokens: int | None = None
    models: list[str] = dataclasses.field(default_factory=lambda: TEST_MODELS.copy())
    device: torch.device = dataclasses.field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )


def run_sweep(xp: Sweep, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise ValueError(f"Output log {out} already exists - please delete first")
    with out.open("w") as outf:
        for i, model_name in enumerate(xp.models):
            model = RequantisableModel.load(
                model_name, device=xp.device, dtype=torch.bfloat16
            )
            n_params = sum(p.nelement() for p in model.model.parameters())
            data = Dataset.load_wikitext2(
                model.model,
                sequence_length=xp.sequence_length,
                batch_size=xp.batch_size,
                kl_topk=xp.kl_topk,
                token_limit=xp.data_max_tokens,
            )
            for j, format in enumerate(xp.formats):
                print(
                    f"-- model {i+1}/{len(xp.models)}, format {j+1}/{len(xp.formats)}",
                    file=sys.stderr,
                )
                config = xp.__dict__.copy()
                config["test"] = "token_prediction"
                config["dataset"] = data.name
                del config["models"]
                config["model"] = model_name
                del config["formats"]
                config["format"] = dataclasses.asdict(format)
                config["format_str"] = str(format)
                config["device"] = config["device"].type
                config["code_changes"] = CODE_CHANGES
                wandb.init(
                    entity="graphcore",
                    project="sparse-attention-formats",
                    reinit=True,
                    config=config,
                )
                outcome = dict(n_tokens=data.masks.sum().item(), n_params=n_params)
                try:
                    log = model.quantise(format)
                    outcome.update(
                        weights={d.pop("name"): d for d in log},
                        bits_per_param=sum(d["bits"] for d in log) / n_params,
                    )
                    outcome.update(evaluate_model(data, model.model))
                except Exception as exc:
                    print(repr(exc), file=sys.stderr)
                    outcome.update(error=repr(exc), backtrace=traceback.format_exc())
                finally:
                    print(json.dumps(dict(**config, **outcome)), file=outf, flush=True)
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

    returns (dof, scale)
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
class StatsExperiment:
    experiment: str
    models: list[str] = dataclasses.field(default_factory=lambda: TEST_MODELS.copy())
    device: torch.device = dataclasses.field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )


def run_weight_stats(xp: StatsExperiment, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise ValueError(f"Output log {out} already exists - please delete first")
    with out.open("w") as outf:
        for i, model_name in enumerate(xp.models):
            print(f"-- model {i+1}/{len(xp.models)}", file=sys.stderr)
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16
            )
            config = xp.__dict__.copy()
            config["test"] = "weight_stats"
            del config["models"]
            config["model"] = model_name
            config["device"] = config["device"].type
            config["code_changes"] = CODE_CHANGES
            wandb.init(
                entity="graphcore",
                project="sparse-attention-formats",
                reinit=True,
                config=config,
            )
            outcome = dict(
                weight_stats={
                    name: tensor_stats(p.to(xp.device))
                    for name, p in tqdm.tqdm(
                        list(model.state_dict().items()), desc=model_name
                    )
                }
            )
            print(json.dumps(dict(**config, **outcome)), file=outf, flush=True)
            wandb.summary.update(outcome)
            wandb.finish()
            del model
