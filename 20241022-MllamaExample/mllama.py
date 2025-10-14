import math
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
import transformers
from torch import Tensor

# Data structures


@dataclass
class VisionConfig:
    # Inputs
    patch_size: int
    image_size: int
    max_tiles: int
    aspect_ratios: list[tuple[int, int]]  # [(ny, nx)]
    image_mean: list[float]
    image_std: list[float]

    # Model
    hidden_size: int
    layers: int
    layers_2: int
    heads: int
    head_size: int
    mlp_size: int
    taps: list[int]

    @property
    def n_aspect(self) -> int:
        return len(self.aspect_ratios)

    @property
    def n_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


@dataclass
class TextConfig:
    hidden_size: int
    layers: int
    q_heads_per_kv_head: int
    kv_heads: int
    head_size: int
    rope_angular_frequency: list[float]
    mlp_size: int
    cross_attention_layers: list[int]
    vocab_size: int

    @property
    def q_heads(self) -> int:
        return self.q_heads_per_kv_head * self.kv_heads


@dataclass
class Config:
    vision: VisionConfig
    text: TextConfig


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
    layers_1: list[VisionLayerParams]  # layers x (h)
    post_norm: LayerNormParams  # (h)
    post_tile_embedding: Tensor  # (n_aspect, max_tiles, h)
    layers_2: list[VisionLayerParams]  # layers_2 x (h)


@dataclass
class TextLayerParams:
    attn_norm: Tensor  # (h)
    attn_q: Tensor  # (q_heads_per_kv_head * kv_heads * head_size, h)
    attn_k: Tensor  # (kv_heads * head_size, h)
    attn_v: Tensor  # (kv_heads * head_size, h)
    attn_o: Tensor  # (h, q_heads_per_kv_head * kv_heads * head_size)
    attn_q_norm: Optional[Tensor]  # (head_size)
    attn_k_norm: Optional[Tensor]  # (head_size)

    mlp_norm: Tensor  # (h)
    mlp_up: Tensor  # (mlp_size, h)
    mlp_gate: Tensor  # (mlp_size, h)
    mlp_down: Tensor  # (h, mlp_size)


@dataclass
class TextParams:
    embedding: Tensor  # (vocab_size+, h)
    layers: list[TextLayerParams]  # (h)
    final_norm: Tensor  # (h)
    unembedding: Tensor  # (vocab_size, h)


@dataclass
class Params:
    vision: VisionParams  # (h=h_vision)
    vision_text_projection: Tensor  # (h_text, h_vision)
    vision_text_projection_bias: Tensor  # (h_text)
    text: TextParams  # (h=h_text)


@dataclass
class Inputs:
    image: Tensor  # (n_tiles, 3, image_size, image_size)
    aspect_ratio_id: Tensor  # ()
    text: Tensor  # (seq_len; int64)

    def to(self, device: torch.device) -> "Inputs":
        return type(self)(**{k: v.to(device) for k, v in self.__dict__.items()})


# Converting the model


def _rope_angular_frequency(config: transformers.PretrainedConfig) -> list[float]:
    head_size = config.hidden_size // config.num_attention_heads
    freq = config.rope_theta ** -(
        torch.arange(0, head_size, 2, dtype=torch.float) / head_size
    )
    s = config.rope_scaling
    z = (
        s["original_max_position_embeddings"] * freq / (2 * torch.pi)
        - s["low_freq_factor"]
    ) / (s["high_freq_factor"] - s["low_freq_factor"])
    freq *= torch.lerp(torch.tensor(1 / s["factor"]), torch.tensor(1.0), z.clip(0, 1))
    return freq.tolist()


def config_from_huggingface(
    config: transformers.PretrainedConfig, processor: transformers.BaseImageProcessor
) -> Config:
    assert config.vision_config.hidden_act == "gelu"
    assert config.text_config.hidden_act == "silu"
    assert config.text_config.rms_norm_eps == NORM_EPS
    return Config(
        vision=VisionConfig(
            # Inputs
            patch_size=config.vision_config.patch_size,
            image_size=config.vision_config.image_size,
            max_tiles=config.vision_config.max_num_tiles,
            aspect_ratios=config.vision_config.supported_aspect_ratios,
            image_mean=processor.image_mean,
            image_std=processor.image_std,
            # Model
            hidden_size=config.vision_config.hidden_size,
            layers=config.vision_config.num_hidden_layers,
            layers_2=config.vision_config.num_global_layers,
            heads=config.vision_config.attention_heads,
            head_size=(
                config.vision_config.hidden_size // config.vision_config.attention_heads
            ),
            mlp_size=config.vision_config.intermediate_size,
            taps=config.vision_config.intermediate_layers_indices,
        ),
        text=TextConfig(
            hidden_size=config.text_config.hidden_size,
            layers=config.text_config.num_hidden_layers,
            q_heads_per_kv_head=(
                config.text_config.num_attention_heads
                // config.text_config.num_key_value_heads
            ),
            kv_heads=config.text_config.num_key_value_heads,
            head_size=(
                config.text_config.hidden_size // config.text_config.num_attention_heads
            ),
            rope_angular_frequency=_rope_angular_frequency(config.text_config),
            mlp_size=config.text_config.intermediate_size,
            cross_attention_layers=config.text_config.cross_attention_layers,
            vocab_size=config.text_config.vocab_size,
        ),
    )


def params_from_huggingface(c: Config, model: transformers.PreTrainedModel) -> Params:
    cv = c.vision
    p = {k: v.detach() for k, v in model.named_parameters()}

    ## Vision

    def _layer_norm(name: str) -> LayerNormParams:
        return LayerNormParams(p[f"{name}.weight"], p[f"{name}.bias"])

    # Merge positional embeddings
    g = "vision_model.gated_positional_embedding"
    g_gate = p[f"{g}.gate"].tanh().reshape(())
    gated_embeddings = (
        p[f"{g}.embedding"] * (1 - g_gate)
        + p[f"{g}.tile_embedding.weight"].reshape(
            cv.n_aspect + 1, cv.max_tiles, cv.n_patches + 1, cv.hidden_size
        )[1:]
        * g_gate
    )
    t = "vision_model.pre_tile_positional_embedding"
    t_gate = p[f"{t}.gate"].tanh()
    vision_positional_embedding = (
        p[f"{t}.embedding.weight"].reshape(
            cv.n_aspect + 1, cv.max_tiles, 1, cv.hidden_size
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
    layers_1 = [
        _vision_layer(f"vision_model.transformer.layers.{i}") for i in range(cv.layers)
    ]
    vision_post_norm = _layer_norm("vision_model.layernorm_post")
    vision_post_tile_embedding = (
        p["vision_model.post_tile_positional_embedding.gate"].tanh()
        * p["vision_model.post_tile_positional_embedding.embedding.weight"]
    ).view(len(cv.aspect_ratios) + 1, cv.max_tiles, cv.hidden_size)[1:]
    layers_2 = [
        _vision_layer(f"vision_model.global_transformer.layers.{i}")
        for i in range(cv.layers_2)
    ]

    ## Language

    def _text_layer(n: int) -> TextLayerParams:
        name = f"language_model.model.layers.{n}"
        if n in c.text.cross_attention_layers:
            # Merge the gates into the linear projections
            return TextLayerParams(
                attn_norm=p[f"{name}.input_layernorm.weight"],
                attn_q=p[f"{name}.cross_attn.q_proj.weight"],
                attn_k=p[f"{name}.cross_attn.k_proj.weight"],
                attn_v=p[f"{name}.cross_attn.v_proj.weight"],
                attn_o=(
                    p[f"{name}.cross_attn.o_proj.weight"]
                    * p[f"{name}.cross_attn_attn_gate"].tanh()
                ),
                attn_q_norm=p[f"{name}.cross_attn.q_norm.weight"],
                attn_k_norm=p[f"{name}.cross_attn.k_norm.weight"],
                mlp_norm=p[f"{name}.post_attention_layernorm.weight"],
                mlp_up=p[f"{name}.mlp.up_proj.weight"],
                mlp_gate=p[f"{name}.mlp.gate_proj.weight"],
                mlp_down=(
                    p[f"{name}.mlp.down_proj.weight"]
                    * p[f"{name}.cross_attn_mlp_gate"].tanh()
                ),
            )
        return TextLayerParams(
            attn_norm=p[f"{name}.input_layernorm.weight"],
            attn_q=p[f"{name}.self_attn.q_proj.weight"],
            attn_k=p[f"{name}.self_attn.k_proj.weight"],
            attn_v=p[f"{name}.self_attn.v_proj.weight"],
            attn_o=p[f"{name}.self_attn.o_proj.weight"],
            attn_q_norm=None,
            attn_k_norm=None,
            mlp_norm=p[f"{name}.post_attention_layernorm.weight"],
            mlp_up=p[f"{name}.mlp.up_proj.weight"],
            mlp_gate=p[f"{name}.mlp.gate_proj.weight"],
            mlp_down=p[f"{name}.mlp.down_proj.weight"],
        )

    ## Result

    return Params(
        vision=VisionParams(
            patch_embedding=p["vision_model.patch_embedding.weight"],
            positional_embedding=vision_positional_embedding,
            class_embedding=vision_class_embedding,
            pre_norm=vision_pre_norm,
            layers_1=layers_1,
            post_norm=vision_post_norm,
            post_tile_embedding=vision_post_tile_embedding,
            layers_2=layers_2,
        ),
        vision_text_projection=p["multi_modal_projector.weight"],
        vision_text_projection_bias=p["multi_modal_projector.bias"],
        text=TextParams(
            embedding=p["language_model.model.embed_tokens.weight"],
            layers=[_text_layer(n) for n in range(c.text.layers)],
            final_norm=p["language_model.model.norm.weight"],
            unembedding=p["language_model.lm_head.weight"],
        ),
    )


def get_inputs(
    config: Config,
    tokenizer: transformers.PreTrainedTokenizerBase,
    image: Tensor,
    text: str,
) -> Inputs:
    image_size = config.vision.image_size

    # Minimum upscaling, otherwise maximum downscaling
    # - note that the rescaling ifself is slightly different to the MllamaImageProcessor
    #   rules, which seem a little odd (although the choice of aspect should be identical)
    scales = list(
        enumerate(
            min(ny * image_size / image.shape[0], nx * image_size / image.shape[1])
            for ny, nx in config.vision.aspect_ratios
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
    ny = math.ceil(image.shape[-2] / image_size)
    nx = math.ceil(image.shape[-1] / image_size)
    image = F.pad(
        image,
        (0, nx * image_size - image.shape[-1], 0, ny * image_size - image.shape[-2]),
    )

    # Rescale
    image = (
        (image / 255) - torch.tensor(config.vision.image_mean)[:, None, None]
    ) / torch.tensor(config.vision.image_std)[:, None, None]

    # Split into tiles
    image = (
        image.reshape(3, ny, image_size, nx, image_size)
        .permute(1, 3, 0, 2, 4)
        .flatten(end_dim=1)
    )

    return Inputs(
        image=image,
        aspect_ratio_id=torch.tensor(aspect_ratio_id),
        text=torch.tensor(tokenizer(text).input_ids),
    )


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
    hiddens = [hidden.clone()]
    for layer in layers:
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
        hiddens.append(hidden.clone())
    return hiddens


def vision_model(config: VisionConfig, params: VisionParams, inputs: Inputs) -> Tensor:
    # Image patching
    patches = (  # (n_tiles, n_patches, hidden_size)
        F.conv2d(
            inputs.image.to(params.patch_embedding.dtype),
            params.patch_embedding,
            stride=config.patch_size,
        )
        .flatten(start_dim=-2)
        .transpose(-1, -2)
    )
    patches += params.positional_embedding[inputs.aspect_ratio_id, : patches.shape[0]]
    class_embedding = params.class_embedding[
        inputs.aspect_ratio_id, : patches.shape[0], None
    ]
    patches = torch.cat([class_embedding, patches], dim=-2)
    patches = layer_norm(patches, params.pre_norm)

    # Pad and mask
    npad = -(patches.shape[-2] % -8)
    mask = F.pad(
        torch.ones(patches.shape[:-1], dtype=torch.bool, device=patches.device),
        (0, npad),
    ).flatten()
    mask = (mask[:, None] | mask).to(patches.dtype).log()
    hidden = F.pad(patches, (0, 0, 0, npad)).flatten(end_dim=-2)

    # 'Transformer' stack
    hiddens_1 = vision_transformer(config, params.layers_1, hidden, attn_mask=mask)
    hidden = hiddens_1[-1]

    # Intermediate ops
    hidden = layer_norm(hidden, params.post_norm)
    hidden.view(config.max_tiles, -1, config.hidden_size).add_(
        params.post_tile_embedding[inputs.aspect_ratio_id, : patches.shape[0], None]
    )

    # 'Global Transformer' stack
    hidden = vision_transformer(config, params.layers_2, hidden, attn_mask=mask)[-1]

    # Concat-interleave intermediate hiddens
    hidden_taps = torch.stack([hiddens_1[t] for t in config.taps], dim=-1).flatten(-2)
    hidden = torch.cat([hidden, hidden_taps], dim=-1)

    # Remove padding
    hidden = hidden.unflatten(0, [config.max_tiles, -1])[:, : patches.shape[-2]]

    return hidden


def rms_norm(x: Tensor, weight: Tensor) -> Tensor:
    return weight * x / torch.sqrt((x**2).mean(-1, keepdim=True) + NORM_EPS)


def rotate(z: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    zx, zy = z.unflatten(-1, (2, -1)).movedim(-2, 0)
    return torch.cat([zx * cos - zy * sin, zy * cos + zx * sin], -1)


def text_transformer(
    config: TextConfig,
    layers: list[TextLayerParams],
    hidden: Tensor,
    vision_out: Tensor,
) -> Tensor:
    pos_id = torch.arange(hidden.shape[-2], device=hidden.device)
    rope_angle = pos_id[:, None] * torch.tensor(
        config.rope_angular_frequency, device=hidden.device
    )
    cos, sin = rope_angle.cos().to(hidden.dtype), rope_angle.sin().to(hidden.dtype)
    for layer_idx, layer in enumerate(layers):
        cross_attn = layer_idx in config.cross_attention_layers
        q_input = rms_norm(hidden, layer.attn_norm)
        kv_input = vision_out if cross_attn else q_input
        q, k, v = (
            (z @ m.T).view(-1, nh, config.head_size).transpose(-2, -3)
            for m, nh, z in (
                (layer.attn_q, config.q_heads, q_input),
                (layer.attn_k, config.kv_heads, kv_input),
                (layer.attn_v, config.kv_heads, kv_input),
            )
        )
        if cross_attn:
            q, k = rms_norm(q, layer.attn_q_norm), rms_norm(k, layer.attn_k_norm)
        else:
            q, k = rotate(q, cos, sin), rotate(k, cos, sin)

        # Add dummy batch axis for torch performance
        mix = F.scaled_dot_product_attention(
            q[None], k[None], v[None], is_causal=not cross_attn, enable_gqa=True
        ).squeeze(0)
        hidden += mix.transpose(-2, -3).flatten(start_dim=-2) @ layer.attn_o.T

        z = rms_norm(hidden, layer.mlp_norm)
        z = F.silu(z @ layer.mlp_gate.T) * (z @ layer.mlp_up.T)
        hidden += z @ layer.mlp_down.T
    return hidden


def text_model(
    config: TextConfig, params: TextParams, text: Tensor, vision_out: Tensor
) -> Tensor:
    hidden = params.embedding[text]
    hidden = text_transformer(config, params.layers, hidden, vision_out)
    hidden = rms_norm(hidden, params.final_norm)
    return hidden @ params.unembedding.T


def generate(
    config: Config, params: Params, inputs: Inputs, n_generated_tokens: int
) -> Iterable[int]:
    with torch.no_grad():
        hidden = vision_model(config.vision, params.vision, inputs)
        vision_out = (
            hidden @ params.vision_text_projection.T
            + params.vision_text_projection_bias
        ).flatten(end_dim=-2)
        text = inputs.text
        for _ in range(n_generated_tokens):
            logits = text_model(config.text, params.text, text, vision_out)
            token = logits[-1].argmax(-1)
            yield int(token)
            text = torch.cat([text, token[None]])
