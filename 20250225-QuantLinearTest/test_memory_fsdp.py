import contextlib
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

import quantisation as Q
from quantisation.layers import quantise_linear_layers
from utility import record_memory


@dataclass
class Settings:
    fsdp: bool = True
    world_size: int = torch.cuda.device_count()
    quant: bool = False
    n_layers: int = 1
    hidden_size: int = 16 * 2**10
    batch_size: int = 1
    n_steps: int = 5
    memory_profile: bool = True


class Model(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_layers: int,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.dtype = dtype
        self.layers = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
            for _ in range(n_layers)
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def train(rank: int, settings: Settings, init_method: str) -> None:
    dist.init_process_group(
        "nccl", init_method=init_method, world_size=settings.world_size, rank=rank
    )

    device = torch.device("cuda", rank)
    torch.set_default_device(device)  # Put all tensors/modules on device by default
    torch.cuda.set_device(device)

    with record_memory() if settings.memory_profile else contextlib.nullcontext():
        model = Model(
            hidden_size=settings.hidden_size,
            n_layers=settings.n_layers,
            dtype=torch.float32,
        )
        if settings.quant:
            model = quantise_linear_layers(model, weight_fmt=Q.parse("E4M3"))

        if settings.fsdp:
            model = FSDP(
                model,
                auto_wrap_policy=size_based_auto_wrap_policy,
                mixed_precision=MixedPrecision(
                    param_dtype=torch.float32,
                    reduce_dtype=torch.float32,
                ),
                sharding_strategy=(
                    ShardingStrategy.FULL_SHARD
                    if settings.world_size > 1
                    else ShardingStrategy.NO_SHARD
                ),
                device_id=torch.cuda.current_device(),
            )
        else:
            assert rank == 0
            assert settings.world_size == 1

        opt = torch.optim.Adam(params=model.parameters())

        for _ in range(settings.n_steps):
            x = torch.randn(
                (settings.batch_size, settings.hidden_size),
                dtype=model.dtype,
                device=device,
            )

            opt.zero_grad()
            loss = model(x).sum()
            loss.backward()
            opt.step()

    dist.destroy_process_group()


if __name__ == "__main__":
    settings = Settings(fsdp=True, world_size=8, quant=True, n_layers=16, n_steps=3)
    mp.spawn(
        train,
        args=(settings, "tcp://localhost:12355"),
        nprocs=settings.world_size,
        join=True,
    )
