# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import dataclasses
from dataclasses import dataclass
from typing import Any

import torch
import tqdm
import transformers
from torch import Tensor, tensor

from .. import quantisation as Q
from .. import model_quantisation as M
from . import core


def mean_block_amax(t: Tensor, b: int) -> Tensor:
    t = t.flatten()
    return t[: b * (t.nelement() // b)].view(-1, b).abs().amax(1).mean()


def scaled_hist(t: Tensor, bin_edges: Tensor, dim: tuple[int, ...] | None) -> Tensor:
    """Compute a histogram of elements, after being normalised by RMS."""
    return (
        torch.bucketize(
            t.div(t.pow(2).mean(dim=dim, keepdim=True).sqrt()).flatten().abs(),
            bin_edges,
        )
        .bincount(minlength=bin_edges.shape[0] + 1)
        .div(t.nelement())
    )


EXPERIMENT_DEFAULT = "20250423-weight-stats"


def fetch_weight_stats(
    model_name: str, experiment_name: str = EXPERIMENT_DEFAULT
) -> dict[str, float]:
    """Fetch the weight stats from a previous experiment."""
    for run in core.runs(experiment_name):
        if run.config.model == model_name and "weight_stats" in run.summary:
            return run.summary.weight_stats
    raise KeyError(
        f"Weight stats for model {model_name!r} not found in experiment {experiment_name!r}"
    )


STUDENTT_FIT_SCALE_THRESHOLD = 0.001
STUDENTT_FIT_DF_VALUES = torch.cat(
    [torch.arange(1, 10, 0.5), torch.arange(10, 20, 2), torch.arange(20, 100 + 1, 10)]
).tolist()


def studentt_fit_scale(
    t: Tensor, df: float, threshold: float = STUDENTT_FIT_SCALE_THRESHOLD
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


def studentt_fit(
    t: Tensor,
    scale_threshold: float = STUDENTT_FIT_SCALE_THRESHOLD,
    dfs: list[float] = STUDENTT_FIT_DF_VALUES,
) -> tuple[Tensor, Tensor]:
    """Compute maximum-likelhood fit of (df, scale) of a zero-mean Student-T distribution to samples `t`.

    returns (df, scale)
    """
    best_log_likelihood = tensor(-torch.inf, device=t.device)
    best_params = None
    for df in torch.tensor(dfs, device=t.device):
        scale = studentt_fit_scale(t, df, threshold=scale_threshold)
        log_likelihood = (
            torch.distributions.StudentT(df=df, scale=scale).log_prob(t).mean()
        )
        if log_likelihood > best_log_likelihood:
            best_params = (df, scale)
            best_log_likelihood = log_likelihood
    return best_params


def dist_fit_stats(t: Tensor) -> dict[str, Any]:
    dist_args = []
    dist_args.append((torch.distributions.Normal, dict(scale=t.std(unbiased=False))))
    dist_args.append((torch.distributions.Laplace, dict(scale=t.abs().mean())))
    t_df, t_scale = studentt_fit(t)
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
            block_max=[mean_block_amax(w, b).item() for b in block_sizes],
            # Histograms
            hist=scaled_hist(w, hist_bins, dim=None).tolist(),
            channel_hist=[
                scaled_hist(
                    w, hist_bins, dim=tuple(d for d in range(w.ndim) if d != dim)
                ).tolist()
                for dim in range(w.ndim)
            ],
            # Distributions
            fit=dist_fit_stats(w),
        )


@dataclass
class Sweep:
    experiment: str
    model: list[str] = core.FIELD_MODELS
    device: torch.device = core.FIELD_DEVICE
    type: str = "weight_stats"

    def run(self) -> None:
        for config in core.iter_dict_product(
            dataclasses.asdict(self), "model", progress=True
        ):
            with core.Experiment(config) as experiment:
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    config["model"], torch_dtype=torch.bfloat16
                )
                params = [
                    (name, param)
                    for name, param in model.named_parameters()
                    if param.ndim == 2
                    and not any(p in M.DEFAULT_IGNORE for p in name.split("."))
                ]
                experiment.summary(
                    weight_stats={
                        name: tensor_stats(p.to(self.device))
                        for name, p in tqdm.tqdm(params, desc=config["model"])
                    }
                )
                del model
