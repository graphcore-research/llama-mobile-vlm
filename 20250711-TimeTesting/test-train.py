import itertools as it
import math
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.fsdp as fsdp
import torch.multiprocessing as mp
import torch.nn.functional as F
import transformers
import weight_formats.quantisation_training as QT
from transformers import MllamaForConditionalGeneration
from weight_formats import quantisation as Q_new

import wandb
from quantisation import quantisation as Q_old
from quantisation.layers import quantise_linear_layers
from training_data import Dataset
from utility import distributed_batches, record_memory

WANDB_PROJECT = "llama-mobile"


@dataclass
class OptimiserSettings:
    lr: float
    betas: tuple[float] = (0.9, 0.999)
    weight_decay: float = 0.0


@dataclass
class LRScheduleSettings:
    type: str = "cosine"
    n_warmup_steps: int = 0


@dataclass
class TrainingSettings:
    n_steps: int
    batch_size: int
    optimiser: OptimiserSettings
    lr_schedule: LRScheduleSettings


@dataclass
class ExecutionSettings:
    world_size: int = torch.cuda.device_count()
    params_dtype: str = "float32"
    compute_dtype: str = "bfloat16"
    teacher_dtype: str = "bfloat16"
    compile: str | None = "default"
    recomputation: bool = True
    wrap_teacher: bool = False


@dataclass
class QuantisationSettings:
    el_fmt: str
    scale_fmt: str
    block_shape: tuple[int]
    implementation: str  # "old"/"new"
    trainable_centroids: bool


@dataclass
class Settings:
    run_name: str
    model_name: str
    train_dataset: str
    quantisation: QuantisationSettings | None
    training: TrainingSettings
    execution: ExecutionSettings
    wandb: bool
    memory_profile: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _compile_model(model: MllamaForConditionalGeneration, mode: str) -> None:
    transformer_modules = [
        model.vision_model.transformer,
        model.vision_model.global_transformer,
        model.language_model.model,
    ]

    for module in transformer_modules:
        for layer_id, layer in module.layers.named_children():
            layer = torch.compile(layer, mode=mode)
            module.layers.register_module(layer_id, layer)


def _apply_fsdp(model: MllamaForConditionalGeneration, **kwargs) -> None:
    transformer_modules = [
        model.vision_model.transformer,
        model.vision_model.global_transformer,
        model.language_model.model,
    ]
    for module in transformer_modules:
        for layer in module.layers:
            fsdp.fully_shard(layer, **kwargs)

    emb_layers = [model.language_model.model.embed_tokens, model.language_model.lm_head]
    for layer in emb_layers:
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

    if settings.wandb and rank == 0:
        config = settings.to_dict()
        config["name"] = settings.run_name
        config["n_devices"] = world_size
        mode = "offline" if settings.wandb == "offline" else "online"
        run = wandb.init(
            config=config,
            mode=mode,
            entity="graphcore",
            project=WANDB_PROJECT,
            reinit=True,
        )

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    processor = transformers.AutoProcessor.from_pretrained(settings.model_name)

    try:
        with record_memory() if settings.memory_profile else nullcontext():
            # Original model
            teacher = transformers.MllamaForConditionalGeneration.from_pretrained(
                settings.model_name,
                torch_dtype=getattr(torch, settings.execution.teacher_dtype),
                device_map="cpu" if settings.execution.wrap_teacher else device,
            )
            for p in teacher.parameters():
                p.requires_grad_(False)

            teacher.eval()

            if settings.execution.wrap_teacher:
                _apply_fsdp(teacher)

            # Quantised model
            student = transformers.MllamaForConditionalGeneration.from_pretrained(
                settings.model_name,
                torch_dtype=getattr(torch, settings.execution.params_dtype),
                device_map="cpu",
            )

            # TODO: Do not enable droupout!
            student.train()

            if settings.quantisation:
                if settings.quantisation.implementation == "new":
                    fmt = Q_new.LinearScalingFormat(
                        Q_new.parse(settings.quantisation.el_fmt),
                        scale_format=Q_new.parse("BFLOAT16"),
                        block_shape=settings.quantisation.block_shape,
                        scaling="absmax",
                    )
                    QT.convert(
                        student,
                        fmt,
                        scaling_mode="dynamic",
                        clip_gradient=False,
                        error_weight=None,
                    )
                    if not settings.quantisation.trainable_centroids:
                        for _, p in QT.get_named_parameters(student, "centroids"):
                            p.requires_grad_(False)
                else:
                    fmt = Q_old.LinearScalingFormat(
                        Q_old.parse(settings.quantisation.el_fmt),
                        group_shapes=[settings.quantisation.block_shape],
                        scale_format=Q_old.parse("BFLOAT16"),
                        scale_combiner=None,
                    )
                    student = quantise_linear_layers(student, fmt)

            if settings.execution.recomputation:
                student.gradient_checkpointing_enable()

            _apply_fsdp(
                student,
                mp_policy=fsdp.MixedPrecisionPolicy(
                    param_dtype=getattr(torch, settings.execution.compute_dtype)
                ),
            )

            # TODO: Move compilation before FSDP
            if settings.execution.compile:
                torch._dynamo.config.cache_size_limit = 64
                teacher = torch.compile(
                    teacher,
                    mode=settings.execution.compile,
                    fullgraph=not settings.execution.wrap_teacher,
                )
                _compile_model(student, mode=settings.execution.compile)

            dir_name = f"data/{settings.train_dataset}-generation/"
            model_name = settings.model_name.replace("meta-llama/", "")
            data = Dataset.load(dir_name + model_name + ".json")

            opt = torch.optim.AdamW(
                student.parameters(),
                lr=settings.training.optimiser.lr,
                betas=settings.training.optimiser.betas,
                weight_decay=settings.training.optimiser.weight_decay,
            )
            lr_scheduler = transformers.get_scheduler(
                settings.training.lr_schedule.type,
                opt,
                num_warmup_steps=settings.training.lr_schedule.n_warmup_steps,
                num_training_steps=settings.training.n_steps,
            )

            batches = it.islice(
                distributed_batches(
                    data.get_datums(),
                    settings.training.batch_size,
                    rank=rank,
                    world_size=world_size,
                ),
                settings.training.n_steps,
            )
            for step, batch in enumerate(batches):
                t0 = time.time()

                imgs = [[x.image] for x in batch]
                texts = [x.out for x in batch]
                inps = processor(imgs, texts, return_tensors="pt").to(device)

                opt.zero_grad()
                with torch.no_grad():
                    teacher_out = teacher(**inps).logits
                student_out = student(**inps, use_cache=False).logits
                loss = F.kl_div(
                    student_out.log_softmax(dim=-1),
                    teacher_out.log_softmax(dim=-1),
                    reduction="sum",
                    log_target=True,
                ) / math.prod(
                    student_out.shape[:-1]
                )  # divide by batch_size * seq_len
                del teacher_out, student_out

                loss.backward()
                opt.step()
                lr_scheduler.step()

                t = time.time() - t0

                # Log step data
                out = {}
                out_loss = loss.detach().clone()
                dist.reduce(out_loss, dst=0, op=dist.ReduceOp.SUM)
                out["loss"] = out_loss.item() / world_size
                out["step_time"] = t
                if settings.wandb and rank == 0:
                    wandb.log(out, step=step)
    except Exception as e:
        error_type = type(e).__name__
        error_message = str(e)
        tb = traceback.format_exc()
        if rank == 0:
            print(error_type, error_message)
            print(tb)
            if settings.wandb:
                run.summary["run_status"] = "Failed"
                run.summary["error_type"] = error_type
                run.summary["error_message"] = error_message
                run.summary["traceback"] = tb

    if settings.wandb and rank == 0:
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    el_fmts = ["E0M2"]
    block_shapes = [(None, None)]
    # block_shapes = [(None, None), (1, 32)]
    torch_compiles = [None, "default"]
    recomputations = [False, True]
    trainable_centroidss = [False, True]
    wrap_teachers = [False, True]
    implementations = ["old", "new"]

    for (
        el_fmt,
        block_shape,
        torch_compile,
        recomp,
        train_centroids,
        wrap_teacher,
        impl,
    ) in it.product(
        el_fmts,
        block_shapes,
        torch_compiles,
        recomputations,
        trainable_centroidss,
        wrap_teachers,
        implementations,
    ):
        if train_centroids and impl == "old":
            continue
        settings = Settings(
            run_name="test-time-v2",
            model_name="meta-llama/Llama-3.2-11B-Vision-Instruct",
            train_dataset="imagenet",
            quantisation=QuantisationSettings(
                el_fmt,
                scale_fmt="BFLOAT16",
                block_shape=block_shape,
                implementation=impl,
                trainable_centroids=train_centroids,
            ),
            training=TrainingSettings(
                n_steps=10,
                batch_size=8,
                optimiser=OptimiserSettings(lr=1e-3),
                lr_schedule=LRScheduleSettings(),
            ),
            execution=ExecutionSettings(
                compile=torch_compile,
                recomputation=recomp,
                wrap_teacher=wrap_teacher,
            ),
            wandb=True,
            memory_profile=False,
        )
        try:
            mp.spawn(
                fsdp_train,
                args=("tcp://localhost:12345", settings),
                nprocs=settings.execution.world_size,
                join=True,
            )
        except Exception:
            continue
