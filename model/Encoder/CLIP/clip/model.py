from collections import OrderedDict
from typing import Tuple, Union, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# =============================================================================
# 1. HELPERS
# =============================================================================

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class TextPreprocessor(nn.Module):
    def __init__(self, context_length: int, vocab_size: int, width: int, dtype: torch.dtype):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        return x

# =============================================================================
# 2. TRANSFORMER BLOCKS (CLEANED)
# =============================================================================

class ResidualAttentionBlock(nn.Module):
    """
    Residual Attention Block with optional V-V Attention path (AnomalyCLIP style).
    
    When enable_vv=True and input is a list [vv_x, qkv_x]:
    - QKV path: standard attention + MLP
    - V-V path: V-V attention only (NO MLP) for fine-grained features
    """
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, enable_vv: bool = False):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.enable_vv = enable_vv
        self.d_model = d_model
        self.n_head = n_head

    def attention(self, x: torch.Tensor):
        """Standard QKV attention."""
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def attention_dual(self, x: torch.Tensor):
        """
        Compute BOTH QKV and V-V attention in single pass (AnomalyCLIP style).
        Returns: (vv_attn_output, qkv_attn_output)
        """
        # Get the projection weights
        in_proj_weight = self.attn.in_proj_weight  # (3*d, d)
        in_proj_bias = self.attn.in_proj_bias
        out_proj_weight = self.attn.out_proj.weight
        out_proj_bias = self.attn.out_proj.bias
        
        L, N, D = x.shape
        num_heads = self.n_head
        head_dim = D // num_heads
        scale = head_dim ** -0.5
        
        # Project to Q, K, V
        qkv = F.linear(x, in_proj_weight, in_proj_bias)  # (L, N, 3*D)
        qkv = qkv.reshape(L, N, 3, num_heads, head_dim).permute(2, 1, 3, 0, 4)  # (3, N, heads, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (N, heads, L, head_dim)
        
        # --- QKV Attention (original path) ---
        attn_qkv = (q @ k.transpose(-2, -1)) * scale
        if self.attn_mask is not None:
            attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
            attn_qkv = attn_qkv + attn_mask
        attn_qkv = F.softmax(attn_qkv, dim=-1)
        out_qkv = (attn_qkv @ v)  # (N, heads, L, head_dim)
        out_qkv = out_qkv.permute(2, 0, 1, 3).reshape(L, N, D)  # (L, N, D)
        out_qkv = F.linear(out_qkv, out_proj_weight, out_proj_bias)
        
        # --- V-V Attention (DPAM path) ---
        # Replace Q and K with V
        attn_vv = (v @ v.transpose(-2, -1)) * scale
        if self.attn_mask is not None:
            attn_vv = attn_vv + attn_mask
        attn_vv = F.softmax(attn_vv, dim=-1)
        out_vv = (attn_vv @ v)  # (N, heads, L, head_dim)
        out_vv = out_vv.permute(2, 0, 1, 3).reshape(L, N, D)  # (L, N, D)
        out_vv = F.linear(out_vv, out_proj_weight, out_proj_bias)
        
        return out_vv, out_qkv

    def forward(self, x: torch.Tensor, vv_x: torch.Tensor = None):
        """
        Forward pass with optional V-V path.
        
        Args:
            x: QKV path input (L, N, D)
            vv_x: V-V path input (L, N, D) or None
            
        Returns:
            x: QKV path output
            vv_x: V-V path output (or None)
        """
        if self.enable_vv and vv_x is not None:
            # Dual-path mode: compute both attentions
            x_norm = self.ln_1(x)
            vv_attn, qkv_attn = self.attention_dual(x_norm)
            
            # QKV path: attention + MLP
            x = x + qkv_attn
            x = x + self.mlp(self.ln_2(x))
            
            # V-V path: attention ONLY (no MLP) - key difference!
            vv_x = vv_x + vv_attn
            
            return x, vv_x
        else:
            # Standard single-path mode
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x, None

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, context_length: int = 77,
                 enable_vv: bool = False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.enable_vv = enable_vv
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, attn_mask, enable_vv=enable_vv) for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, out_layers: List[int] = None):
        # --- CASE 1: Vision Encoder (Feature Extraction with V-V path) ---
        if out_layers is not None:
            intermediate_outputs = []
            vv_x = x.clone() if self.enable_vv else None
            
            for i, block in enumerate(self.resblocks):
                x, vv_x = block(x, vv_x)
                if i in out_layers:
                    qkv_out = x.permute(1, 0, 2).clone()
                    vv_out = vv_x.permute(1, 0, 2).clone() if (self.enable_vv and vv_x is not None) else None
                    intermediate_outputs.append((qkv_out, vv_out))
            return x, intermediate_outputs

        # --- CASE 2: Text Encoder ---
        for i, block in enumerate(self.resblocks):
            x, _ = block(x, None)
        return x

# =============================================================================
# 3. VISION & CLIP 
# =============================================================================

class VisionTransformer(nn.Module):
    """
    Vision Transformer with optional V-V Attention path.
    
    When enable_vv=True, intermediate layer outputs use accumulated V-V
    attention features instead of standard QKV features for better 
    fine-grained anomaly detection.
    """
    def __init__(self, input_resolution, patch_size, width, layers, heads, output_dim, enable_vv: bool = False):
        super().__init__()
        self.width = width
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.enable_vv = enable_vv
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, enable_vv=enable_vv)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        
        # LoRA parameters for proj
        self.lora_A = nn.Parameter(torch.zeros((width, 16)))
        self.lora_B = nn.Parameter(torch.zeros((16, output_dim)))
        self.lora_alpha = 16.0
        self.lora_r = 16
        nn.init.normal_(self.lora_A, std=1 / self.lora_r)
        nn.init.zeros_(self.lora_B)

    def _interpolate_pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return positional embeddings, bicubically interpolated if grid size changed."""
        pos = self.positional_embedding.to(x.dtype)
        if x.shape[1] == pos.shape[0]:
            return pos
        cls_pos   = pos[:1]                                  # [1, D]
        patch_pos = pos[1:]                                  # [N, D]
        old_g     = int(patch_pos.shape[0] ** 0.5)          # 24 for 336px
        new_n     = x.shape[1] - 1
        new_h     = int(new_n ** 0.5)
        new_w     = new_n // new_h
        patch_pos = patch_pos.reshape(1, old_g, old_g, -1).permute(0, 3, 1, 2).float()
        patch_pos = F.interpolate(patch_pos, size=(new_h, new_w),
                                  mode='bicubic', align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(new_n, -1).to(x.dtype)
        return torch.cat([cls_pos, patch_pos], dim=0)

    def forward(self, x: torch.Tensor, out_layers: List[int] = None):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self._interpolate_pos_embed(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2) # NLD -> LND
        
        if out_layers:
            # Get intermediate features (V-V or QKV based on enable_vv)
            x, out_tokens_tuples = self.transformer(x, out_layers=out_layers)
            
            out_tokens_qkv_raw = []
            out_tokens_vv_raw = []
            
            for (qkv_token, vv_token) in out_tokens_tuples:
                out_tokens_qkv_raw.append(qkv_token)
                if vv_token is not None:
                    out_tokens_vv_raw.append(vv_token)
            
            # Shape: [Layers, Batch, Seq, width (1024)]
            out_tokens_final_qkv = torch.stack(out_tokens_qkv_raw, dim=0)
            
            if out_tokens_vv_raw:
                out_tokens_final_vv = torch.stack(out_tokens_vv_raw, dim=0)
            else:
                out_tokens_final_vv = None
        else:
            x = self.transformer(x)
            out_tokens_final_qkv = None
            out_tokens_final_vv = None
            
        x = x.permute(1, 0, 2) # LND -> NLD

        x_cls_unproj = self.ln_post(x[:, 0, :])
        x_cls = x_cls_unproj
        if self.proj is not None:
            base_out = x_cls_unproj @ self.proj
            lora_out = (x_cls_unproj @ self.lora_A) @ self.lora_B * (self.lora_alpha / self.lora_r)
            x_cls = base_out + lora_out

        return x_cls, out_tokens_final_qkv, out_tokens_final_vv

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int = 1024,
                 image_resolution: int = 336,
                 vision_layers: int = 24,
                 vision_width: int = 1024,
                 vision_patch_size: int = 14,
                 context_length: int = 77,
                 vocab_size: int = 49_408,
                 transformer_width: int = 768,
                 transformer_heads: int = 12,
                 transformer_layers: int = 12,
                 enable_vv: bool = False
        ):
        super().__init__()

        self.context_length = context_length
        self.enable_vv = enable_vv
        self.embed_dim = embed_dim

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution, patch_size=vision_patch_size, width=vision_width, layers=vision_layers,
            heads=vision_heads, output_dim=embed_dim, enable_vv=enable_vv
        )

        self.transformer = Transformer(
            width=transformer_width, layers=transformer_layers, heads=transformer_heads, attn_mask=self.build_attention_mask(),
            enable_vv=enable_vv,
            context_length=self.context_length
        )

        self.vocab_size = vocab_size
        self.text_preprocessor = TextPreprocessor(
            context_length, vocab_size, transformer_width, self.dtype
        )
        
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            if hasattr(block.attn, 'in_proj_weight'): nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None: nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, out_layers):
        # returns cls, qkv_list, vv_list
        return self.visual(image.type(self.dtype), out_layers=out_layers)
    
    def encode_text(self, text):
        x = self.text_preprocessor(text).type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        
        # Adjust EOT token index if tokens were shifted by soft prompts
        eot_indices = text.argmax(dim=-1)
        if hasattr(self.text_preprocessor, 'token_shift'):
            eot_indices = (eot_indices + self.text_preprocessor.token_shift).clamp(max=x.shape[1] - 1)
            
        x = x[torch.arange(x.shape[0]), eot_indices] @ self.text_projection
        return x

    def forward(self, image, text, out_layers=None):
        image_features_cls, image_intermediates_qkv, _ = self.encode_image(image, out_layers=out_layers)
        text_features = self.encode_text(text)

        image_features = image_features_cls / image_features_cls.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

def convert_weights(model: nn.Module):
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None: l.bias.data = l.bias.data.half()
        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None: tensor.data = tensor.data.half()
        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None: attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

def build_model(state_dict: dict, enable_vv: bool = False):
    vit = "visual.proj" in state_dict
    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.transformer.resblocks") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        raise NotImplementedError("Only ViT supported")

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))
    
    model = CLIP(
        embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        enable_vv=enable_vv
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict: del state_dict[key]

    if "token_embedding.weight" in state_dict:
        state_dict["text_preprocessor.token_embedding.weight"] = state_dict.pop("token_embedding.weight")
    if "positional_embedding" in state_dict:
        state_dict["text_preprocessor.positional_embedding"] = state_dict.pop("positional_embedding")
        
    # Clean up old/mismatched keys
    keys_to_remove = [k for k in state_dict if "visual.cm_bottlenecks" in k]
    for k in keys_to_remove: del state_dict[k]
        
    convert_weights(model)
    model.load_state_dict(state_dict, strict=False)
    return model.eval()
