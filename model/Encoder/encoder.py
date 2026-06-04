"""
Encoder module — Pure CLIP Backbone with Soft Prompting + DINOv3 dense encoder.

Primary encoders:
  - CLIP ViT-L/14 (frozen): Provides both the global CLS anchor AND the aligned text embeddings.
  - Supports V-V attention path.
  - DINOv3 ViT-L/16 (frozen): Dense patch features at arbitrary resolution.

Trainable Modules:
  - SoftPromptWrapper: Linguistic context injection.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple

from .CLIP import clip

# Add DINOv3 package root so its absolute imports resolve
_dinov3_root = os.path.join(os.path.dirname(__file__), 'dinov3')
if _dinov3_root not in sys.path:
    sys.path.insert(0, _dinov3_root)

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    pass

class SoftPromptWrapper(nn.Module):
    def __init__(self, base_tp: nn.Module, prompts: nn.ParameterDict):
        super().__init__()
        self.base_tp = [base_tp] 
        self.prompts = prompts
        self.active_key = list(prompts.keys())[0]
        # Calculate token shift caused by soft prompts
        self.token_shift = prompts[self.active_key].shape[1]

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        device = text.device
        if self.base_tp[0].token_embedding.weight.device != device:
             self.base_tp[0].to(device)
        text_tokens = self.base_tp[0].token_embedding(text)
        B, L, D = text_tokens.shape
        ctx_vectors = self.prompts[self.active_key].to(dtype=text_tokens.dtype, device=device)
        if ctx_vectors.shape[0] != B:
            ctx_vectors = ctx_vectors[0:1].expand(B, -1, -1)
        prefix = text_tokens[:, 0:1, :]
        suffix = text_tokens[:, 1:, :]
        combined = torch.cat([prefix, ctx_vectors, suffix], dim=1)
        if combined.shape[1] > 77:
            combined = combined[:, :77, :]
        pos_embed = self.base_tp[0].positional_embedding.to(dtype=combined.dtype, device=device)
        return combined + pos_embed[:combined.shape[1], :]

class _SoftTextPrompts(nn.Module):
    def __init__(self, clip_model: nn.Module, num_tokens: int):
        super().__init__()
        self.num_tokens = num_tokens
        base_tp = clip_model.text_preprocessor
        embed_dim = base_tp.positional_embedding.shape[1]
        prompts = nn.ParameterDict()
        # Separate soft prompts for segmentation (pixel-level) and classification (image-level)
        for cls_type in ["normal_seg", "anomaly_seg", "normal_cls", "anomaly_cls"]:
            param = nn.Parameter(torch.empty(1, self.num_tokens, embed_dim))
            nn.init.normal_(param, std=0.02)
            prompts[cls_type] = param
        self.wrapper = SoftPromptWrapper(base_tp, prompts)
        clip_model.text_preprocessor = self.wrapper

    def use(self, context_type: str):
        if context_type in self.wrapper.prompts:
            self.wrapper.active_key = context_type

class Encoder(nn.Module):
    """
    Pure CLIP Encoder.
    """
    def __init__(self, learnable_tokens: int = 16, use_vv_path: bool = True):
        super().__init__()

        self.clip_model, self.clip_transform, self.clip_out_dim, self.clip_v_dim = self._load_clip(use_vv_path)

        self.soft_text_prompts = None
        if learnable_tokens > 0:
            self.soft_text_prompts = _SoftTextPrompts(self.clip_model, learnable_tokens)

        # Public dimensions
        self.v_dim = self.clip_v_dim # 1024
        self.t_dim = self.clip_out_dim # 768
        self.out_dim = self.clip_out_dim # 768

        self.register_buffer('clip_mean', torch.tensor([0.48145466, 0.4578275,  0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('clip_std',  torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

        self._token_cache: dict = {}

    def _load_clip(self, use_vv_path: bool):
        model, transform = clip.load("ViT-L/14@336px", device="cpu", enable_vv=use_vv_path)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model, transform, model.embed_dim, model.visual.width

    def get_device(self) -> torch.device:
        return next(self.clip_model.parameters()).device

    def encode_images(self, image_tensors: torch.Tensor, out_layers: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.get_device()
        imgs   = image_tensors.to(device)
        
        with torch.no_grad():
            # clip_model returns: x_cls, out_tokens_final_qkv, out_tokens_final_vv
            clip_cls, qkv_stack, vv_stack = self.clip_model.encode_image(imgs, out_layers=out_layers)
            
            # Remove CLS token from patch stacks: [L, B, 1+N, D] -> [L, B, N, D]
            if qkv_stack is not None:
                qkv_stack = qkv_stack[:, :, 1:, :]
            if vv_stack is not None:
                vv_stack = vv_stack[:, :, 1:, :]
            
        return clip_cls, qkv_stack, vv_stack

    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        if not hasattr(self, '_token_cache'):
            self._token_cache = {}
            
        device = self.get_device()
        if isinstance(texts, str): texts = [texts]
        cache_key = tuple(texts)
        if cache_key not in self._token_cache:
            self._token_cache[cache_key] = clip.tokenize(texts).to(device)
        tokens = self._token_cache[cache_key].to(device)
        
        # When text prompt changes, we must allow gradients through the soft prompt parameters
        # But we do not train the base CLIP text encoder
        text_features = self.clip_model.encode_text(tokens)
            
        return text_features

    def use_soft_prompts(self, context_type: str = "global"):
        if self.soft_text_prompts:
            self.soft_text_prompts.use(context_type)


def _remap_hf_dinov3_to_local(sd: dict) -> dict:
    """
    Map HuggingFace DINOv3 safetensors key names to the local DinoVisionTransformer layout.

    HF layout                                   → local layout
    ───────────────────────────────────────────────────────────
    embeddings.cls_token                        → cls_token
    embeddings.mask_token  [1,1,D]              → mask_token  [1,D]  (squeezed)
    embeddings.register_tokens                  → storage_tokens
    embeddings.patch_embeddings.{weight,bias}   → patch_embed.proj.{weight,bias}
    norm.{weight,bias}                          → norm.{weight,bias}
    layer.{i}.norm{1,2}.{weight,bias}           → blocks.{i}.norm{1,2}.{weight,bias}
    layer.{i}.attention.{q,k,v}_proj.weight     → blocks.{i}.attn.qkv.weight  (cat)
    layer.{i}.attention.{q,v}_proj.bias         → blocks.{i}.attn.qkv.bias    (cat, k=zeros)
    layer.{i}.attention.o_proj.{weight,bias}    → blocks.{i}.attn.proj.{weight,bias}
    layer.{i}.layer_scale{1,2}.lambda1          → blocks.{i}.ls{1,2}.gamma
    layer.{i}.mlp.up_proj.{weight,bias}         → blocks.{i}.mlp.fc1.{weight,bias}
    layer.{i}.mlp.down_proj.{weight,bias}       → blocks.{i}.mlp.fc2.{weight,bias}

    Keys absent from HF (architecture buffers, init'd by pretrained=False):
      rope_embed.periods, blocks.*.attn.qkv.bias_mask  → loaded with strict=False
    """
    new = {}

    simple_top = {
        'embeddings.cls_token':               'cls_token',
        'embeddings.register_tokens':         'storage_tokens',
        'embeddings.patch_embeddings.weight': 'patch_embed.proj.weight',
        'embeddings.patch_embeddings.bias':   'patch_embed.proj.bias',
        'norm.weight':                        'norm.weight',
        'norm.bias':                          'norm.bias',
    }
    for hf_k, loc_k in simple_top.items():
        if hf_k in sd:
            new[loc_k] = sd[hf_k]

    # mask_token: HF [1, 1, D] → local [1, D]
    if 'embeddings.mask_token' in sd:
        new['mask_token'] = sd['embeddings.mask_token'].squeeze(0)

    block_simple = {
        'norm1.weight':            'norm1.weight',
        'norm1.bias':              'norm1.bias',
        'norm2.weight':            'norm2.weight',
        'norm2.bias':              'norm2.bias',
        'attention.o_proj.weight': 'attn.proj.weight',
        'attention.o_proj.bias':   'attn.proj.bias',
        'layer_scale1.lambda1':    'ls1.gamma',
        'layer_scale2.lambda1':    'ls2.gamma',
        'mlp.up_proj.weight':      'mlp.fc1.weight',
        'mlp.up_proj.bias':        'mlp.fc1.bias',
        'mlp.down_proj.weight':    'mlp.fc2.weight',
        'mlp.down_proj.bias':      'mlp.fc2.bias',
    }
    for i in range(24):
        hp, lp = f'layer.{i}.', f'blocks.{i}.'
        for hf_s, loc_s in block_simple.items():
            k = hp + hf_s
            if k in sd:
                new[lp + loc_s] = sd[k]

        # Merge separate Q/K/V → single QKV weight [3D, D]
        q_w = sd.get(f'{hp}attention.q_proj.weight')
        k_w = sd.get(f'{hp}attention.k_proj.weight')
        v_w = sd.get(f'{hp}attention.v_proj.weight')
        if q_w is not None and k_w is not None and v_w is not None:
            new[f'{lp}attn.qkv.weight'] = torch.cat([q_w, k_w, v_w], dim=0)

        # QKV bias: Q and V present in HF; K absent → pad with zeros
        q_b = sd.get(f'{hp}attention.q_proj.bias')
        v_b = sd.get(f'{hp}attention.v_proj.bias')
        if q_b is not None and v_b is not None:
            new[f'{lp}attn.qkv.bias'] = torch.cat([q_b, torch.zeros_like(q_b), v_b], dim=0)

    return new


class DINOv3Encoder(nn.Module):
    """
    Frozen DINOv3 ViT-L/16 backbone that loads pretrained weights from HuggingFace.

    Uses the prebuilt dinov3_vitl16 architecture from Model/Encoder/dinov3/ and
    maps HF safetensors keys to the local DinoVisionTransformer layout.

    Args:
        repo_id: HuggingFace model repo (default: facebook/dinov3-vitl16-pretrain-lvd1689m).
        out_layers: block indices for get_intermediate_layers (default: [5,11,17,23]).
    """

    HF_REPO = 'facebook/dinov3-vitl16-pretrain-lvd1689m'

    def __init__(self, repo_id: str = HF_REPO, out_layers: List[int] = None):
        super().__init__()
        self.out_layers = out_layers or [5, 11, 17, 23]

        from dinov3.hub.backbones import dinov3_vitl16
        self.model = dinov3_vitl16(pretrained=False)
        self._load_hf_weights(repo_id)

        # Freeze all parameters — this encoder is a fixed feature extractor
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.embed_dim = 1024  # ViT-L
        self.patch_size = 16

        # ImageNet normalisation (DINOv3 pre-training stats)
        self.register_buffer('dino_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('dino_std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _load_hf_weights(self, repo_id: str):
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file

            weights_path = hf_hub_download(repo_id=repo_id, filename='model.safetensors')
            raw_sd   = load_file(weights_path)
            remapped = _remap_hf_dinov3_to_local(raw_sd)

            missing, unexpected = self.model.load_state_dict(remapped, strict=False)
            # Expected missing: rope_embed.periods, blocks.*.attn.qkv.bias_mask
            real_missing = [k for k in missing if 'rope_embed' not in k and 'bias_mask' not in k]
            if real_missing:
                print(f'[Warning] DINOv3: {len(real_missing)} truly missing keys: {real_missing[:5]}')
            if unexpected:
                print(f'[Warning] DINOv3: {len(unexpected)} unexpected keys ignored.')
            print(f'[INFO] DINOv3 weights loaded from HuggingFace: {repo_id}')
        except Exception as e:
            print(f'[Warning] DINOv3 HF weight load failed ({e}). Using random init.')

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise a [B, 3, H, W] float tensor in [0, 1] with ImageNet stats."""
        return (x - self.dino_mean.to(x.device)) / self.dino_std.to(x.device)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] float in [0, 1] — will be normalised internally.
        Returns:
            List of len(out_layers) tensors, each [B, T, 1024]
            (CLS and register tokens already stripped by get_intermediate_layers).
        """
        x_norm = self.normalise(x)
        with torch.no_grad():
            feats = self.model.get_intermediate_layers(
                x_norm, n=self.out_layers, reshape=False
            )
        return list(feats)
