from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.distributed.fsdp as fsdp
import torch.multiprocessing as mp
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


@dataclass
class ExecutionSettings:
    world_size: int = torch.cuda.device_count()
    params_dtype: str = "float32"
    compute_dtype: str = "bfloat16"
    teacher_dtype: str = "bfloat16"
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
                batch_size=torch.cuda.device_count(),
            ),
            execution=ExecutionSettings(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _compile_model(model: nn.Module, mode: str) -> None:
    for layer_id, layer in model.layers.named_children():
        layer = torch.compile(layer, mode=mode)
        model.layers.register_module(layer_id, layer)


def _apply_fsdp(model: nn.Module, **kwargs) -> None:
    for layer in model.layers:
        fsdp.fully_shard(layer, **kwargs)

    fsdp.fully_shard(model, **kwargs)


def fsdp_train(rank: int, init_method: str, settings: Settings) -> None:
    dist.init_process_group(
        "nccl",
        init_method=init_method,
        world_size=settings.execution.world_size,
        rank=rank,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    with record_memory() if settings.memory_profile else nullcontext():
        teacher = DummyModel(
            settings.hidden_size,
            settings.n_layers,
            getattr(torch, settings.execution.teacher_dtype),
        )

        teacher.eval()

        for p in teacher.parameters():
            p.requires_grad_(False)

        student = DummyModel(
            settings.hidden_size,
            settings.n_layers,
            getattr(torch, settings.execution.params_dtype),
            device=torch.device("cpu"),
            recomputation=settings.execution.recomputation,
        )
        student.train()

        _apply_fsdp(
            student,
            mp_policy=fsdp.MixedPrecisionPolicy(
                param_dtype=getattr(torch, settings.execution.compute_dtype)
            ),
        )

        if settings.execution.compile:
            teacher = torch.compile(teacher, mode=settings.execution.compile)
            _compile_model(student, mode=settings.execution.compile)

        opt = torch.optim.AdamW(student.parameters())

        for _ in range(settings.training.n_steps):
            local_bs = settings.training.batch_size // world_size
            x = torch.randn(
                (local_bs, settings.hidden_size),
                dtype=getattr(torch, settings.execution.compute_dtype),
                requires_grad=True,
            )

            opt.zero_grad()
            with torch.no_grad():
                teacher_out = teacher(x)
            student_out = student(x)
            loss = (teacher_out - student_out).square().sum()
            loss.backward()
            opt.step()

    dist.destroy_process_group()


if __name__ == "__main__":
    settings = Settings.default()

    mp.spawn(
        fsdp_train,
        args=("tcp://localhost:12345", settings),
        nprocs=settings.execution.world_size,
        join=True,
    )
