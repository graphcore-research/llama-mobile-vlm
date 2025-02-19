"""Main experiments"""

from dataclasses import dataclass

import datasets
import torch
import transformers
from torch import Tensor, tensor

import quantisation as Q


@dataclass
class Dataset:
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
        device: torch.device,
        token_limit: int | None = None,
    ) -> "Dataset":
        """Load and tokenize the dataset, then use the model to provide reference logits.

        (Note topk_indices and topk_logp will be filled with dummy data.)
        """
        data = datasets.load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1", split="test"
        )["text"]
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
            # uses `sequence_length - 1``)
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

    def reset(self) -> None:
        for name, p in self.model.state_dict().items():
            p[...] = self.original_params[name].to(p.device)

    def quantise(
        self, format: Q.TensorFormat, compile: bool = True
    ) -> list[dict[str, str | bool | float]]:
        log = []
        quantise = format.quantise
        if compile:
            quantise = torch.compile(quantise)
        for name, p in self.model.state_dict().items():
            if p.ndim == 2:
                p0 = self.original_params[name].to(p.device)
                p[...] = quantise(p0)
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
                log.append(dict(name=name, quantised=False))
        return log
