# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import contextlib
import dataclasses
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import safetensors.torch
import torch
import tqdm
import transformers
from torch import Tensor, nn

from .. import sensitivity as S
from .. import model_quantisation as M
from . import core, token_prediction


EXPERIMENT_DEFAULT = "20250423-fisher"


def fetch_fisher(
    model_name: str, device: torch.device, experiment_name: str = EXPERIMENT_DEFAULT
) -> dict[str, Tensor]:
    file_name = f"{model_name.replace('/', '--')}.safetensors"
    path = Path(__file__).parent.parent.parent / "out" / experiment_name / file_name
    if not path.is_file():
        raise ValueError(
            f"Fisher checkpoint for {model_name} expected at {path}."
            f"\nTry: `aws s3 sync s3://graphcore-research/2025-04-block-formats/{experiment_name}/{file_name} {path}"
        )
    return safetensors.torch.load_file(path, device=str(device))


def fetch_fisher_sum(
    model_name: str, experiment_name: str = EXPERIMENT_DEFAULT
) -> dict[str, float]:
    """Fetch the sum-Fisher stats from a previous experiment."""
    for run in core.runs(experiment_name):
        if run.config.model == model_name and "fisher" in run.summary:
            return run.summary.fisher
    raise KeyError(
        f"Fisher stats for model {model_name!r} not found in experiment {experiment_name!r}"
    )


@contextlib.contextmanager
def activation_checkpointing_enabled(
    model: transformers.PreTrainedModel,
) -> Iterable[transformers.PreTrainedModel]:
    """A context manager to enable activation checkpointing, without enabling dropout.

    Warning - care might be necessary, in case the .training flag on `LlamaModel` etc
    is used to enable dropout.
    """
    assert not model.is_gradient_checkpointing
    try:
        # Don't use .train(), as we don't want to enable dropout.
        # Just hope none of the PreTrainedModel classes don't set dropout themselves.
        for m in model.modules():
            if isinstance(m, transformers.modeling_utils.PreTrainedModel):
                assert not m.training
                m.training = True
        model.gradient_checkpointing_enable()
        yield model
    finally:
        model.gradient_checkpointing_disable()
        for m in model.modules():
            if isinstance(m, transformers.modeling_utils.PreTrainedModel):
                m.training = False


def diag_fisher(
    data: token_prediction.Dataset,
    model: nn.Module,
    mode: Literal["empirical", "single_sample"],
    progress: bool = False,
    ignore: tuple[str] = M.DEFAULT_IGNORE,
) -> dict[str, Tensor]:
    """Compute the diagonal of the Fisher information for Linear/Embedding weight parameters."""

    param_to_name = {}  # Handle parameter sharing
    for name, p in model.named_parameters():
        p.requires_grad_(False)  # Save memory by skipping parameter gradients
        param_to_name[p] = name
    S.wrap(model)
    try:
        for index in tqdm.tqdm(
            list(range(data.n_batch)), desc="fisher", disable=not progress
        ):
            tokens = data.tokens[index]
            logits = model(
                nn.functional.pad(tokens, (1, 0), value=data.bos_token_id),
                use_cache=False,
            ).logits[:, :-1]
            if mode == "empirical":
                targets = tokens
            elif mode == "single_sample":
                targets = logits.add(
                    torch.rand_like(logits).log_().neg_().log_().neg_()
                ).argmax(-1)
            nn.functional.cross_entropy(
                logits.flatten(end_dim=-2), targets.flatten(), reduction="none"
            ).view(targets.shape).backward(
                data.masks[index]
                .to(logits.dtype)
                .unsqueeze(1)
                .broadcast_to(targets.shape)
            )
        results = {}
        for module in model.modules():
            if isinstance(module, S.Wrapper):
                name = param_to_name[module.wrapped.weight]
                if not any(p in ignore for p in name.split(".")):
                    # Convert to a mean over batch and sequence
                    grad_weight_sq = module.grad_weight_sq.sum() / (
                        data.masks.sum().cpu() * data.sequence_length
                    )
                    if name in results:
                        results[name] += grad_weight_sq
                    else:
                        results[name] = grad_weight_sq
        return results
    finally:
        S.unwrap(model)


@dataclass
class Sweep:
    experiment: str
    mode: Literal["single_sample", "empirical"] = "single_sample"
    sequence_length: int = 4096
    sequence_limit: int | None = 1024
    line_limit: int | None = int(1e5)
    batch_size: int = 1
    model: list[str] = core.FIELD_MODELS
    device: torch.device = core.FIELD_DEVICE
    type: str = "fisher"

    def run(self, out: Path) -> None:
        for config in core.iter_dict_product(
            dataclasses.asdict(self), "model", progress=True
        ):
            with core.Experiment(config) as experiment:
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    config["model"], torch_dtype=torch.bfloat16, device_map=self.device
                )
                data = token_prediction.Dataset.load(
                    model,
                    sequence_length=self.sequence_length,
                    sequence_limit=self.sequence_limit,
                    line_limit=self.line_limit,
                    batch_size=self.batch_size,
                    kl_topk=0,
                    dataset=("wikitext", ("train",)),
                    progress=True,
                )
                with activation_checkpointing_enabled(model):
                    sensitivity = diag_fisher(
                        data, model, mode=self.mode, progress=True
                    )

                out.mkdir(parents=True, exist_ok=True)
                out_file = out / f"{config['model'].replace('/', '--')}.safetensors"
                safetensors.torch.save_file(
                    {k: v.bfloat16() for k, v in sensitivity.items()}, out_file
                )
                experiment.summary(
                    fisher={k: v.sum() for k, v in sensitivity.items()},
                    file_sha=(
                        subprocess.check_output(["sha256sum", str(out_file)])
                        .decode()
                        .split(" ")[0]
                    ),
                )
                del model
