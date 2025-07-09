from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from utility import record_memory


class DummyModel(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_layers: int,
        dtype: torch.dtype,
        device: Optional[torch.device] = None,
        recomputation: bool = False,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(
                hidden_size, hidden_size, bias=False, device=device, dtype=dtype
            )
            for _ in range(n_layers)
        )
        self.dtype = dtype
        self.recomputation = recomputation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if self.recomputation:
                x = checkpoint(layer, x, use_reentrant=True)
            else:
                x = layer(x)
        return x


@dataclass
class TrainingSettings:
    n_steps: int
    batch_size: int
    adam: bool = True


@dataclass
class ExecutionSettings:
    dtype: torch.dtype = torch.bfloat16
    compile: str | None = None
    recomputation: bool = False


@dataclass
class Settings:
    hidden_size: int
    n_layers: int
    training: TrainingSettings
    execution: ExecutionSettings
    memory_profile: bool = True

    @classmethod
    def default(cls) -> "Settings":
        return cls(
            hidden_size=2**12,
            n_layers=10,
            training=TrainingSettings(
                n_steps=5,
                batch_size=1,
            ),
            execution=ExecutionSettings(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _compile_model(model: nn.Module, mode: str) -> None:
    for layer_id, layer in model.layers.named_children():
        layer = torch.compile(layer, mode=mode)
        model.layers.register_module(layer_id, layer)


def train(settings: Settings) -> None:
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    with record_memory() if settings.memory_profile else nullcontext():
        model = DummyModel(
            settings.hidden_size,
            settings.n_layers,
            settings.execution.dtype,
            recomputation=settings.execution.recomputation,
        )
        model.train()

        if settings.execution.compile:
            _compile_model(model, mode=settings.execution.compile)

        if settings.training.adam:
            opt = torch.optim.AdamW(model.parameters())
        else:
            opt = torch.optim.SGD(model.parameters())

        for _ in range(settings.training.n_steps):
            x = torch.randn(
                (settings.training.batch_size, settings.hidden_size),
                dtype=settings.execution.dtype,
                requires_grad=True,  # required when using gradient recomp
            )

            opt.zero_grad()
            out = model(x)
            loss = out.square().sum()
            loss.backward()
            opt.step()


if __name__ == "__main__":
    settings = Settings.default()
    settings.training.adam = False
    train(settings)
