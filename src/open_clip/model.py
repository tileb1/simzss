""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

import open_clip
from DenoisingViT import Denoiser, ViTWrapper
from dinov2.models.vision_transformer import DinoVisionTransformer
from .hf_model import HFTextEncoder
from .modified_resnet import ModifiedResNet
from .timm_model import TimmModel
from .transformer import LayerNormFp32, LayerNorm, QuickGELU, Attention, VisionTransformer, TextTransformer
from .utils import freeze_batch_norm_2d
from .utils import to_2tuple


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    input_patchnorm: bool = False  # whether to use dual patchnorm - would only apply the input layernorm on each patch, as post-layernorm already exist in original clip vit design
    global_average_pool: bool = False  # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer
    n_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    output_tokens: bool = False

    timm_model_name_nooverride: str = None  # Like below, but doesn't do any overriding
    denoiser_type: str = None
    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth

    # Dinov2
    ours: bool = False
    num_heads: Optional[int] = 12,
    checkpoint_path: Optional[str] = None
    n_dense_cls: Optional[int] = None
    init_values: Optional[float] = 1.0
    block_chunks: Optional[int] = 0
    num_register_tokens: Optional[int] = 0
    n_decoder_layers: Optional[int] = 1
    n_decoder_heads: Optional[int] = 1
    use_gap: Optional[bool] = False


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False
    n_dense_cls: int = 0
    mask_type_text: str = 'causal'


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ('bf16', 'pure_bf16'):
        input_dtype = torch.bfloat16
    elif precision in ('fp16', 'pure_fp16'):
        input_dtype = torch.float16
    return input_dtype


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    # Check if ours
    is_ours = False
    if 'ours' in vision_cfg:
        is_ours = True

    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    elif is_ours:
        wrapped = ViTWrapper(model_type=vision_cfg.timm_model_name_nooverride, stride=vision_cfg.patch_size,
                             use_gap=vision_cfg.use_gap)
        visual = Denoiser(vit=wrapped, denoiser_type=vision_cfg.denoiser_type, image_size=vision_cfg.image_size,
                          patch_size=vision_cfg.patch_size)
        if vision_cfg.denoiser_type is not None:
            state_dict = torch.load(vision_cfg.checkpoint_path)
            visual.load_state_dict(state_dict['denoiser'], strict=False)
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            input_patchnorm=vision_cfg.input_patchnorm,
            global_average_pool=vision_cfg.global_average_pool,
            attentional_pool=vision_cfg.attentional_pool,
            n_queries=vision_cfg.n_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    return visual


def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj=text_cfg.proj,
            pooler_type=text_cfg.pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
            output_tokens=text_cfg.output_tokens,
        )
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            output_tokens=text_cfg.output_tokens,
            pad_id=text_cfg.pad_id,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mask_type_text=text_cfg.mask_type_text
        )
    return text


class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.text = text
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()


class CustomTextCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size

        self.init_logit_scale = init_logit_scale
        self.init_logit_bias = init_logit_bias

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False, return_all=False):
        eos, tokens = self.text(text)
        if not return_all:
            return F.normalize(eos, dim=-1) if normalize else eos
        if normalize:
            return F.normalize(eos, dim=-1), F.normalize(tokens, dim=-1)
        else:
            return eos, tokens

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        if isinstance(l, (CLIP, TextTransformer)):
            # convert text nn.Parameter projections
            attr = getattr(l, "text_projection", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

        if isinstance(l, VisionTransformer):
            # convert vision nn.Parameter projections
            attr = getattr(l, "proj", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat


# used to maintain checkpoint compatibility
def convert_to_custom_text_state_dict(state_dict: dict):
    if 'text_projection' in state_dict:
        # old format state_dict, move text tower -> .text
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in (
                    'text_projection',
                    'positional_embedding',
                    'token_embedding',
                    'transformer',
                    'ln_final',
            )):
                k = 'text.' + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict


def build_model_from_openai_state_dict(
        state_dict: dict,
        quick_gelu=True,
        cast_dtype=torch.float16,
):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)
    convert_weights_to_fp16(model)  # OpenAI state dicts are partially converted to float16
    model.load_state_dict(state_dict)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device('cpu')):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros((batch_size, model.context_length), dtype=torch.int, device=device)
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,)
        ))
    model.visual.image_size = image_size
    return model


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', antialias: bool = True):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed


class SimZSS(CustomTextCLIP):

    def __init__(
            self,
            parsed_args,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.args = parsed_args
        self.visual_norm = parsed_args.visual_norm
        self.tokenizer = open_clip.get_tokenizer(self.args.model)

        # Classification layer
        self.n_classes = parsed_args.n_classes
        self.register_buffer('classification_weights_ema', torch.zeros([self.n_classes + 1, kwargs['embed_dim']]))
        self.register_buffer('beta_ema', torch.tensor(parsed_args.beta_ema))

        # Initialize the decoder
        self.register_buffer('logit_scale_local', self.init_logit_scale * torch.ones(1, requires_grad=False))
        if self.args.n_dense_cls > 0:

            if self.args.decoder_type == 'shallow':
                self.decoder = ShallowDecoder(parsed_args, kwargs['embed_dim'])
            else:
                raise NotImplemented

            # Initialize logits scale/bias for local loss
            if self.logit_bias is not None:
                self.logit_bias_local = nn.Parameter(torch.ones([]) * self.init_logit_bias)
            else:
                self.logit_bias_local = None
        else:
            self.logit_bias_local = None

    @torch.no_grad()
    def embed_words(self, words):
        tokens = self.tokenizer(words).to(next(self.parameters()).device)

        # Feed to the text encoder
        chunk_size = 128
        n_chunks = tokens.shape[0] // chunk_size + 1
        words_embeddings = []
        for chunk in torch.chunk(tokens, chunks=n_chunks, dim=0):
            words_embeddings.append(self.encode_text(chunk))
        words_embeddings = torch.cat(words_embeddings, dim=0)
        print('Successfully computed the class embeddings.')
        return words_embeddings

    def encode_image(self, image, normalize: bool = False):
        tokens, cls_tokens = self.visual(image, return_class_token=True)
        B, H, W, C = tokens.shape
        tokens = tokens.reshape(B, H * W, C)
        dict_img = {
            "x_clstoken": cls_tokens,
            "x_patchtokens": tokens,
        }
        return {k: F.normalize(v, dim=-1) for k, v in dict_img.items()} if normalize else dict_img

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
            raw_texts: Optional[torch.Tensor] = None,
            concepts_indices: Optional[torch.Tensor] = None,
            concept_indices: Optional[torch.Tensor] = None,
            nouns: Optional[list] = None,
            concepts_labels: Optional[list] = None,
            just_visual_tokens=False,
            debug=False
    ):
        if self.args.indices_type == 'concept':
            concepts_indices = concept_indices

        dict_img = self.encode_image(image, normalize=True) if image is not None else None
        if just_visual_tokens:
            return dict_img["x_patchtokens"]

        # For zero-shot classification
        if text is None:
            return {"image_features": dict_img["x_clstoken"]}

        eos_txt, tokens_txt = self.encode_text(text, normalize=True, return_all=True) if text is not None else None

        if self.args.n_dense_cls > 0:
            # Identify the indices of the nouns in the caption
            b, n, d = tokens_txt.shape

            # Remove useless data
            temp_bool = concepts_indices != 0
            n_max_tokens = temp_bool.sum(dim=(0, 1)).argmin()
            n_max_nouns = temp_bool.sum(dim=(0, 2)).argmin()

            if n_max_nouns.item() == 0.0 or n_max_tokens.item() == 0.0:
                cls_embeds_txt = None
                cls_embeds_visual = None
                classification_weights_ema = self.classification_weights_ema
            else:
                concepts_indices = concepts_indices[:, :n_max_nouns, :n_max_tokens]
                concepts_indices[concepts_indices == 0] = n

                # indices = self.get_tensor_nouns_indices(nouns_indices, token_padding=n)
                concepts_indices = concepts_indices.to(tokens_txt.device)

                # Pad the sequences with an extra zero token
                tokens_txt = torch.cat(
                    [tokens_txt, torch.zeros([b, 1, d], dtype=tokens_txt.dtype, device=tokens_txt.device)], dim=1)

                # Extract the tokens involved in nouns -> b, max_n_nouns, max_n_tokens, d
                concepts_sequence = torch.gather(
                    input=tokens_txt[:, None].expand(-1, concepts_indices.shape[1], -1, -1),
                    dim=2,
                    index=concepts_indices[..., None].expand(-1, -1, -1, d),
                )

                # Average the tokens representation in each noun
                concepts_sequence = concepts_sequence.sum(dim=2)

                # Prepare the masks for nouns that do not contain any real token
                valid_queries = rearrange(concepts_indices.min(dim=-1).values != n, 'b n -> (b n)')

                # Compute the classificaition weights
                concepts_labels = concepts_labels[:, :concepts_sequence.shape[1]]
                concepts_labels[concepts_labels == -1] = self.n_classes
                classification_weights = torch.zeros([self.n_classes + 1, d], device=concepts_sequence.device,
                                                     dtype=concepts_sequence.dtype)
                classification_weights = torch.scatter_reduce(
                    input=classification_weights,
                    dim=0,
                    index=concepts_labels[..., None].expand(-1, -1, d).reshape(-1, d),
                    src=concepts_sequence.reshape(-1, d),
                    reduce='sum'
                )
                classification_weights = F.normalize(classification_weights, dim=-1, p=2)

                # Get the EMA classification weights
                classification_weights_ema = self.classification_weights_ema
                classification_weights_ema = self.beta_ema * classification_weights_ema.detach() + (
                            1. - self.beta_ema) * classification_weights
                classification_weights_ema = F.normalize(classification_weights_ema, dim=-1, p=2)

                # Update the concepts
                concepts_sequence = torch.gather(
                    input=classification_weights_ema,
                    dim=0,
                    index=concepts_labels[..., None].expand(-1, -1, d).reshape(-1, d),
                )
                concepts_sequence = rearrange(concepts_sequence, '(b n) d -> b n d', b=b)
                concepts_sequence = F.normalize(concepts_sequence, dim=-1)

                # Decode
                queries = concepts_sequence.permute(1, 0, 2)
                key_values = dict_img['x_patchtokens'].permute(1, 0, 2)
                cls_embeds_visual = self.decoder(queries, key_values)
                cls_embeds_visual = F.normalize(rearrange(cls_embeds_visual, 'n b d -> (b n) d'), dim=-1, p=2)
                cls_embeds_txt = rearrange(concepts_sequence, 'b n d -> (b n) d')

                # Discard non-valid representations
                cls_embeds_txt = cls_embeds_txt[valid_queries]
                cls_embeds_visual = cls_embeds_visual[valid_queries]
                concepts_labels = rearrange(concepts_labels, 'b n -> (b n)')[valid_queries]
        else:
            cls_embeds_txt = None
            cls_embeds_visual = None
            classification_weights_ema = self.classification_weights_ema

        # Get image-level features
        if self.args.image_features_type == "cls":
            image_features = dict_img["x_clstoken"]
        else:
            raise NotImplementedError

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": eos_txt,
                "logit_scale": self.logit_scale.exp(),
                "cls_embeds_visual": cls_embeds_visual,
                "cls_embeds_txt": cls_embeds_txt,
                "logit_scale_local": self.logit_scale_local.exp(),
                "classification_weights_ema": classification_weights_ema,
                "concepts_labels": concepts_labels
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
                out_dict['logit_bias_local'] = self.logit_bias_local
            if debug:
                out_dict['cls_embeds_txt_raw'] = cls_embeds_txt
                out_dict['cls_embeds_visual_raw'] = cls_embeds_visual
                out_dict['visual_tokens'] = dict_img["x_patchtokens"]
            return out_dict
        else:
            raise NotImplemented


class OursDinoVisionTransformer(DinoVisionTransformer):
    def __init__(
            self,
            image_size=224,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.image_size = image_size

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert (not unlocked_groups)
        # lock full model
        for param in self.parameters():
            param.requires_grad = False
        if freeze_bn_stats:
            freeze_batch_norm_2d(self)


class ShallowDecoder(nn.Module):
    def __init__(self, parsed_args, d_model):
        super().__init__()
        self.args = parsed_args
        mapping_softmax_dim = {'patch': 2, 'n_dense_cls': 1}
        self.softmax_dim = mapping_softmax_dim[self.args.decoder_softmax_dim]

    def forward(self, text, vision):
        texttopatch = torch.einsum('k b d, n b d -> b k n', text, vision)
        texttopatch = F.softmax(texttopatch / self.args.temp_decoder, dim=self.softmax_dim)
        decoded = torch.einsum('n b d, b k n -> k b d', vision, texttopatch)
        return decoded
