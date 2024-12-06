import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
import transformers
from torch import Tensor

# Data structures


@dataclass
class VisionConfig:
    hidden_size: int
    layers: int
    layers_2: int
    heads: int
    head_size: int
    mlp_size: int
    taps: list[int]


@dataclass
class Config:
    vision: VisionConfig
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
class VisionLayerParams:
    attn_norm: LayerNormParams  # (h)
    attn_q: Tensor  # (heads * head_size, h)
    attn_k: Tensor  # (heads * head_size, h)
    attn_v: Tensor  # (heads * head_size, h)
    attn_o: Tensor  # (h, heads * head_size)

    mlp_norm: LayerNormParams  # (h)
    mlp_up: Tensor  # (mlp_size, h)
    mlp_up_bias: Tensor  # (mlp_size)
    mlp_down: Tensor  # (h, mlp_size)
    mlp_down_bias: Tensor  # (h)


@dataclass
class VisionParams:
    patch_embedding: Tensor  # (h, 3, patch_size, patch_size)
    positional_embedding: Tensor  # (n_aspect, max_tiles, n_patches, h)
    class_embedding: Tensor  # (n_aspect, max_tiles, h)
    pre_norm: LayerNormParams  # (h)
    layers: list[VisionLayerParams]  # (h)
    post_norm: LayerNormParams  # (h)
    post_tile_embedding: Tensor  # (n_aspect, max_tiles, h)
    layers_2: list[VisionLayerParams]  # (h)


@dataclass
class Params:
    vision: VisionParams  # (h_vision)


@dataclass
class Inputs:
    image: Tensor  # (n_tiles, 3, image_size, image_size)
    aspect_ratio_id: Tensor  # ()


# Converting the model


def config_from_huggingface(
    config: transformers.PretrainedConfig, processor: transformers.BaseImageProcessor
) -> Config:
    assert config.vision_config.hidden_act == "gelu"
    return Config(
        vision=VisionConfig(
            hidden_size=config.vision_config.hidden_size,
            layers=config.vision_config.num_hidden_layers,
            layers_2=config.vision_config.num_global_layers,
            heads=config.vision_config.attention_heads,
            head_size=config.vision_config.hidden_size
            // config.vision_config.attention_heads,
            mlp_size=config.vision_config.intermediate_size,
            taps=config.vision_config.intermediate_layers_indices,
        ),
        patch_size=config.vision_config.patch_size,
        image_size=config.vision_config.image_size,
        max_tiles=config.vision_config.max_num_tiles,
        aspect_ratios=config.vision_config.supported_aspect_ratios,
        image_mean=processor.image_mean,
        image_std=processor.image_std,
    )


def params_from_huggingface(c: Config, model: transformers.PreTrainedModel) -> Params:
    p = {k: v.detach() for k, v in model.named_parameters()}

    def _layer_norm(name: str) -> LayerNormParams:
        return LayerNormParams(p[f"{name}.weight"], p[f"{name}.bias"])

    # Merge positional embeddings
    g = "vision_model.gated_positional_embedding"
    g_gate = p[f"{g}.gate"].tanh().reshape(())
    gated_embeddings = (
        p[f"{g}.embedding"] * (1 - g_gate)
        + p[f"{g}.tile_embedding.weight"].reshape(
            c.n_aspect + 1, c.max_tiles, c.n_patches + 1, c.vision.hidden_size
        )[1:]
        * g_gate
    )
    t = "vision_model.pre_tile_positional_embedding"
    t_gate = p[f"{t}.gate"].tanh()
    vision_positional_embedding = (
        p[f"{t}.embedding.weight"].reshape(
            c.n_aspect + 1, c.max_tiles, 1, c.vision.hidden_size
        )[1:]
        * t_gate
        + gated_embeddings[..., 1:, :]
    )
    vision_class_embedding = (
        p["vision_model.class_embedding"] + gated_embeddings[..., 0, :]
    )

    def _vision_layer(name: str) -> VisionLayerParams:
        layer = VisionLayerParams(
            attn_norm=_layer_norm(f"{name}.input_layernorm"),
            attn_q=p[f"{name}.self_attn.q_proj.weight"],
            attn_k=p[f"{name}.self_attn.k_proj.weight"],
            attn_v=p[f"{name}.self_attn.v_proj.weight"],
            attn_o=p[f"{name}.self_attn.o_proj.weight"],
            mlp_norm=_layer_norm(f"{name}.post_attention_layernorm"),
            mlp_up=p[f"{name}.mlp.fc1.weight"],
            mlp_up_bias=p[f"{name}.mlp.fc1.bias"],
            mlp_down=p[f"{name}.mlp.fc2.weight"],
            mlp_down_bias=p[f"{name}.mlp.fc2.bias"],
        )
        # Merge the gate into the final projection
        if f"{name}.gate_attn" in p:
            layer.attn_o = layer.attn_o * p[f"{name}.gate_attn"].tanh()
            layer.mlp_down = layer.mlp_down * p[f"{name}.gate_ffn"].tanh()
            layer.mlp_down_bias = layer.mlp_down_bias * p[f"{name}.gate_ffn"].tanh()
        return layer

    vision_pre_norm = _layer_norm("vision_model.layernorm_pre")
    layers = [
        _vision_layer(f"vision_model.transformer.layers.{i}")
        for i in range(c.vision.layers)
    ]
    vision_post_norm = _layer_norm("vision_model.layernorm_post")
    vision_post_tile_embedding = (
        p["vision_model.post_tile_positional_embedding.gate"].tanh()
        * p["vision_model.post_tile_positional_embedding.embedding.weight"]
    ).view(len(c.aspect_ratios) + 1, c.max_tiles, c.vision.hidden_size)[1:]
    layers_2 = [
        _vision_layer(f"vision_model.global_transformer.layers.{i}")
        for i in range(c.vision.layers_2)
    ]

    return Params(
        VisionParams(
            patch_embedding=p["vision_model.patch_embedding.weight"],
            positional_embedding=vision_positional_embedding,
            class_embedding=vision_class_embedding,
            pre_norm=vision_pre_norm,
            layers=layers,
            post_norm=vision_post_norm,
            post_tile_embedding=vision_post_tile_embedding,
            layers_2=layers_2,
        )
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


def vision_transformer(
    config: VisionConfig,
    layers: list[VisionLayerParams],
    hidden: Tensor,
    attn_mask: Tensor,
) -> list[Tensor]:
    """The transformer stack."""
    hiddens = []
    for layer in layers:
        hiddens.append(hidden.clone())
        z = layer_norm(hidden, layer.attn_norm)
        q, k, v = (
            (z @ m.T).view(-1, config.heads, config.head_size).transpose(-2, -3)
            for m in (layer.attn_q, layer.attn_k, layer.attn_v)
        )
        # Add dummy batch axis for torch performance
        mix = F.scaled_dot_product_attention(
            q[None], k[None], v[None], attn_mask=attn_mask
        ).squeeze(0)
        hidden += mix.transpose(-2, -3).flatten(start_dim=-2) @ layer.attn_o.T

        z = layer_norm(hidden, layer.mlp_norm)
        z = F.gelu(z @ layer.mlp_up.T + layer.mlp_up_bias)
        hidden += z @ layer.mlp_down.T + layer.mlp_down_bias
    return hiddens
