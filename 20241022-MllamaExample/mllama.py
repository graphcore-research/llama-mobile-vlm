import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
import transformers
from torch import Tensor

# Data structures


@dataclass
class Config:
    hidden_size_vision: int
    patch_size: int
    image_size: int
    max_tiles: int
    aspect_ratios: list[tuple[int, int]]  # [(ny, nx)]
    image_mean: list[float]
    image_std: list[float]

    @property
    def n_aspect(self) -> int:
        return len(self.aspect_ratios)

    @property
    def n_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


@dataclass
class LayerNormParams:
    weight: Tensor  # (hidden_size)
    bias: Tensor  # (hidden_size)


@dataclass
class Params:
    patch_embedding: Tensor  # (h_vision, 3, patch_size, patch_size)
    vision_positional_embedding: Tensor  # (n_aspect, max_tiles, n_patches, h_vision)
    vision_class_embedding: Tensor  # (n_aspect, max_tiles, h_vision)
    vision_pre_norm: LayerNormParams  # (h_vision)


@dataclass
class Inputs:
    image: Tensor  # (n_tiles, 3, image_size, image_size)
    aspect_ratio_id: Tensor  # ()


# Converting the model


def config_from_huggingface(
    config: transformers.PretrainedConfig, processor: transformers.BaseImageProcessor
) -> Config:
    return Config(
        hidden_size_vision=config.vision_config.hidden_size,
        patch_size=config.vision_config.patch_size,
        image_size=config.vision_config.image_size,
        max_tiles=config.vision_config.max_num_tiles,
        aspect_ratios=config.vision_config.supported_aspect_ratios,
        image_mean=processor.image_mean,
        image_std=processor.image_std,
    )


def params_from_huggingface(c: Config, model: transformers.PreTrainedModel) -> Params:
    p = dict(model.named_parameters())
    g = "vision_model.gated_positional_embedding"
    g_gate = p[f"{g}.gate"].tanh().reshape(())
    gated_embeddings = (
        p[f"{g}.embedding"] * (1 - g_gate)
        + p[f"{g}.tile_embedding.weight"].reshape(
            c.n_aspect + 1, c.max_tiles, c.n_patches + 1, c.hidden_size_vision
        )[1:]
        * g_gate
    )
    t = "vision_model.pre_tile_positional_embedding"
    t_gate = p[f"{t}.gate"].tanh()
    vision_positional_embedding = (
        p[f"{t}.embedding.weight"].reshape(
            c.n_aspect + 1, c.max_tiles, 1, c.hidden_size_vision
        )[1:]
        * t_gate
        + gated_embeddings[..., 1:, :]
    )
    vision_class_embedding = (
        p["vision_model.class_embedding"] + gated_embeddings[..., 0, :]
    )
    vision_pre_norm = LayerNormParams(
        p["vision_model.layernorm_pre.weight"], p["vision_model.layernorm_pre.bias"]
    )
    return Params(
        patch_embedding=p["vision_model.patch_embedding.weight"],
        vision_positional_embedding=vision_positional_embedding,
        vision_class_embedding=vision_class_embedding,
        vision_pre_norm=vision_pre_norm,
    )


def get_inputs(config: Config, image: Tensor) -> Inputs:
    image_size = config.image_size

    # Minimum upscaling, otherwise maximum downscaling
    # - note that the rescaling ifself is slightly different to the MllamaImageProcessor
    #   rules, which seem a little odd (although the choice of aspect should be identical)
    scales = list(
        enumerate(
            min(ny * image_size / image.shape[0], nx * image_size / image.shape[1])
            for ny, nx in config.aspect_ratios
        )
    )
    if any(s >= 1 for _, s in scales):
        aspect_ratio_id, scale = min(
            (x for x in scales if x[1] >= 1), key=lambda x: (x[1], x[0])
        )
    else:
        aspect_ratio_id, scale = max(scales, key=lambda x: (x[1], -x[0]))
    image = F.interpolate(
        image.movedim(-1, 0)[None], scale_factor=scale, mode="bilinear"
    ).squeeze(0)

    # Padding
    ny = math.ceil(image.shape[-2] / config.image_size)
    nx = math.ceil(image.shape[-1] / config.image_size)
    image = F.pad(
        image,
        (0, nx * image_size - image.shape[-1], 0, ny * image_size - image.shape[-2]),
    )

    # Rescale
    image = (
        (image / 255) - torch.tensor(config.image_mean)[:, None, None]
    ) / torch.tensor(config.image_std)[:, None, None]

    # Split into tiles
    image = (
        image.reshape(3, ny, config.image_size, nx, config.image_size)
        .permute(1, 3, 0, 2, 4)
        .flatten(end_dim=1)
    )

    return Inputs(image=image, aspect_ratio_id=torch.tensor(aspect_ratio_id))


# Model implementation

NORM_EPS = 1e-5


def layer_norm(x: Tensor, p: LayerNormParams) -> Tensor:
    z = x - x.mean(-1, keepdim=True)
    z /= torch.sqrt((z**2).mean(-1, keepdim=True) + NORM_EPS)
    return z * p.weight + p.bias
