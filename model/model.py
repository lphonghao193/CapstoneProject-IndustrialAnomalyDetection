import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Encoder.encoder import Encoder, DINOv3Encoder
from .losses import FocalLoss, DiceLoss, RelativeDistillationLoss, CrossEntropyLoss
from .prompt import get_prompts, TEMPLATES
from .modules import NeighborAggregator, MapFuser
from utils.utils import gaussian_blur_2d

class ProposeModel(nn.Module):
    def __init__(self, **args):
        super(ProposeModel, self).__init__()
        self.out_layers = [5, 11, 17, 23]
        self.num_layers = len(self.out_layers)
        self.embed_dim = 768

        self.encoder = Encoder(learnable_tokens=12, use_vv_path=True)

        self.qkv_projs = nn.ModuleList([
            nn.Linear(1024, 768, bias=False) for _ in range(self.num_layers)
        ])

        self.dino_encoder = DINOv3Encoder()
        self.dino_layers = self.dino_encoder.out_layers

        self.dino_projs_1 = nn.ModuleList([
            nn.Linear(1024, 1024, bias=True) for _ in range(self.num_layers)
        ])

        self.dino_projs_2 = nn.ModuleList([
            nn.Linear(1024, 768, bias=False) for _ in range(self.num_layers)
        ])

        if hasattr(self.encoder.clip_model.visual, 'proj') and self.encoder.clip_model.visual.proj is not None:
            clip_proj = self.encoder.clip_model.visual.proj.t()
            for proj in self.qkv_projs:
                proj.weight.data.copy_(clip_proj)
            for proj in self.dino_projs_1:
                nn.init.eye_(proj.weight)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)
            for proj in self.dino_projs_2:
                proj.weight.data.copy_(clip_proj)

        self.layer_weights = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)

        self.map_residuals = nn.ModuleList([
            MapFuser() for _ in range(self.num_layers)
        ])

        self.register_buffer('mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

        self.cls_weight   = nn.Parameter(torch.tensor(0.7))
        self.pixel_weight = nn.Parameter(torch.tensor(0.3))

        self.loss_focal   = FocalLoss()
        self.loss_dice    = DiceLoss()
        self.loss_cls     = CrossEntropyLoss()
        self.loss_distill = RelativeDistillationLoss()

        self.sigma      = args.get('sigma', 4.0)
        self.neighbor_agg = NeighborAggregator(embed_dim=768, sigma=self.sigma)

        self.pixel_temp = nn.Parameter(torch.tensor(1.0 / 0.07))
        self.image_temp = nn.Parameter(torch.tensor(1.0 / 0.07))

        self._text_cache: dict = {}
        self._cache_lock = threading.Lock()

        self.setup()

    def get_device(self): return next(self.parameters()).device

    def train(self, mode: bool = True):
        self._text_cache.clear()
        super().train(mode)
        if mode:
            self.encoder.eval()
            self.dino_encoder.eval()
        return self

    def setup(self):
        self.requires_grad_(False)
        self.qkv_projs.train().requires_grad_(True)
        self.dino_projs_1.train().requires_grad_(True)
        self.dino_projs_2.train().requires_grad_(True)
        self.layer_weights.requires_grad = True
        for mod in self.map_residuals:
            mod.train().requires_grad_(True)
        self.cls_weight.requires_grad = True
        self.pixel_weight.requires_grad = True
        self.neighbor_agg.train().requires_grad_(True)
        self.pixel_temp.requires_grad = True
        self.image_temp.requires_grad = True
        if self.encoder.soft_text_prompts:
            self.encoder.soft_text_prompts.train().requires_grad_(True)
        if hasattr(self.encoder.clip_model.visual, 'lora_A'):
            self.encoder.clip_model.visual.lora_A.requires_grad_(True)
            self.encoder.clip_model.visual.lora_B.requires_grad_(True)

    def _get_text_features(self, _type: str, class_name: str = 'object'):
        with self._cache_lock:
            if not self.training and _type in self._text_cache:
                return self._text_cache[_type]

        self.encoder.use_soft_prompts(f"{_type}")
        base_type = _type.split('_')[0] if '_' in _type else _type
        prompts = get_prompts('object')[base_type]

        raw_feats = self.encoder.encode_text(prompts)
        feats = F.normalize(raw_feats, dim=-1)

        num_templates = len(TEMPLATES)

        actual_num_prompts = feats.shape[0]
        if actual_num_prompts % num_templates == 0:
            num_attr = actual_num_prompts // num_templates
            attr_feats = torch.stack([
                F.normalize(feats[j * num_templates:(j + 1) * num_templates].mean(dim=0), dim=0)
                for j in range(num_attr)
            ])
            attr_prompts = [prompts[i * num_templates] for i in range(num_attr)]
        else:
            attr_feats = feats
            attr_prompts = prompts

        mean_feat = F.normalize(attr_feats.mean(dim=0), dim=0)
        result = (mean_feat, attr_feats, attr_prompts)

        with self._cache_lock:
            if not self.training:
                self._text_cache[_type] = result
        return result

    def forward(self, inputs):
        raw_images = inputs['image'].to(self.get_device())
        images     = (raw_images - self.mean) / self.std
        B, C, H_img, W_img = images.shape

        with torch.no_grad():
            clip_cls, qkv_stack, vv_stack = self.encoder.encode_images(images, self.out_layers)

        L, B_local, T, D = qkv_stack.shape

        grid_h = int(T ** 0.5)
        for i in range(grid_h, 0, -1):
            if T % i == 0:
                grid_h = i; break
        grid_w = T // grid_h

        t_norm_seg_proto, t_norm_seg_all, t_norm_seg_prompts = self._get_text_features('normal_seg')
        t_anom_seg_proto, t_anom_seg_all, t_anom_seg_prompts = self._get_text_features('anomaly_seg')

        t_norm_cls_proto, t_norm_cls_all, t_norm_cls_prompts = self._get_text_features('normal_cls')
        t_anom_cls_proto, t_anom_cls_all, t_anom_cls_prompts = self._get_text_features('anomaly_cls')

        w_both = torch.stack([t_norm_seg_proto, t_anom_seg_proto], dim=0).view(2, -1, 1, 1)

        raw_dino = inputs['image_dino'].to(self.get_device())
        dino_feats = self.dino_encoder(raw_dino)

        T_dino = dino_feats[0].shape[1]
        dino_h = int(T_dino ** 0.5)
        for i in range(dino_h, 0, -1):
            if T_dino % i == 0:
                dino_h = i; break
        dino_w = T_dino // dino_h

        raw_diffs       = []
        anomaly_maps_up = []
        align_losses    = []

        if self.training:
            gt_mask_train = inputs['mask'].float().to(self.get_device())
            normal_patch_mask = (F.interpolate(
                gt_mask_train, size=(grid_h, grid_w), mode='bilinear', align_corners=False
            ).reshape(B, T) < 0.1).float()

        current_map = None

        for i in range(self.num_layers):
            dino_distill_feat = self.dino_projs_1[i](dino_feats[i])
            feat_dino = self.dino_projs_2[i](dino_distill_feat)
            feat_s = F.normalize(
                feat_dino.permute(0, 2, 1).reshape(B, 768, dino_h, dino_w), dim=1
            )
            dino_sims = F.conv2d(feat_s.to(w_both.dtype), w_both)
            dino_diff = dino_sims[:, 1:2] - dino_sims[:, 0:1]

            qkv_feat_raw = self.qkv_projs[i](qkv_stack[i])
            qkv_scales = self.neighbor_agg(qkv_feat_raw)

            qkv_sims_scales = []
            for q_scale in qkv_scales:
                q_norm = F.normalize(
                    q_scale.permute(0, 2, 1).reshape(B, 768, grid_h, grid_w), dim=1
                )
                sims = F.conv2d(q_norm.to(w_both.dtype), w_both)
                qkv_sims_scales.append(sims[:, 1:2] - sims[:, 0:1])

            clip_diff = torch.stack(qkv_sims_scales, dim=0).mean(dim=0)
            clip_diff_up = F.interpolate(
                clip_diff, size=(dino_h, dino_w), mode='bilinear', align_corners=False
            )

            current_map = self.map_residuals[i](clip_diff_up, dino_diff)
            raw_diffs.append(current_map)

            if self.training:
                logits = torch.cat([-current_map, current_map], dim=1)
                anomaly_maps_up.append(
                    F.softmax((logits * self.pixel_temp).float(), dim=1)[:, 1:2]
                )

                teacher = F.normalize(qkv_stack[i].detach(), dim=-1)
                dino_distill_spatial = dino_distill_feat.permute(0, 2, 1).reshape(B, 1024, dino_h, dino_w)
                dino_down = F.interpolate(
                    dino_distill_spatial, size=(grid_h, grid_w), mode='bilinear', align_corners=False
                ).permute(0, 2, 3, 1).reshape(B, T, 1024)
                student = F.normalize(dino_down, dim=-1)

                per_patch = (1.0 - (student * teacher).sum(dim=-1))
                n_normal  = normal_patch_mask.sum().clamp(min=1.0)
                align_losses.append((per_patch * normal_patch_mask).sum() / n_normal)

        n_maps   = len(raw_diffs)
        w_layers = F.softmax(self.layer_weights[:n_maps].float(), dim=0).view(n_maps, 1, 1, 1, 1)
        fused    = torch.sum(torch.stack(raw_diffs, dim=0) * w_layers, dim=0).contiguous()

        fused_p  = F.softmax((torch.cat([-fused, fused], dim=1) * self.pixel_temp).float(), dim=1)[:, 1:2]
        final_map = F.interpolate(fused_p, size=(H_img, W_img), mode='bilinear', align_corners=False).contiguous()

        if not self.training and self.sigma > 0:
            final_map = gaussian_blur_2d(final_map, self.sigma)

        text_logits = torch.stack([
            (F.normalize(clip_cls, dim=-1) * t_norm_cls_proto).sum(dim=-1),
            (F.normalize(clip_cls, dim=-1) * t_anom_cls_proto).sum(dim=-1),
        ], dim=1) * self.image_temp

        score_text = F.softmax(text_logits.float(), dim=1)[:, 1]
        k_val     = max(1, int(0.005 * H_img * W_img))
        pixel_max = final_map.flatten(1).topk(k=k_val, dim=1).values.mean(dim=1)

        w_cls, w_pix = torch.softmax(torch.stack([self.cls_weight, self.pixel_weight]), dim=0)
        final_cls = torch.clamp(score_text * w_cls + pixel_max * w_pix, 1e-7, 1.0 - 1e-7)

        if self.training:
            gt_lbl  = inputs['label'].float().to(self.get_device())
            with torch.amp.autocast('cuda', enabled=False):
                loss_seg = 0.0
                for a_map in anomaly_maps_up:
                    a_r = F.interpolate(a_map, size=gt_mask_train.shape[2:], mode='bilinear', align_corners=False)
                    loss_seg += self.loss_focal(a_r, gt_mask_train) + self.loss_dice(a_r, gt_mask_train)
                loss_seg = loss_seg / len(anomaly_maps_up) if anomaly_maps_up else torch.tensor(0.0, device=self.get_device())

                loss_cls = self.loss_cls(final_cls.float(), gt_lbl)

                all_text_feats   = torch.cat([t_norm_seg_all, t_anom_seg_all, t_norm_cls_all, t_anom_cls_all], dim=0)
                all_text_prompts = t_norm_seg_prompts + t_anom_seg_prompts + t_norm_cls_prompts + t_anom_cls_prompts
                loss_distill = self.loss_distill(all_text_feats, all_text_prompts)

                loss_align = torch.stack(align_losses).mean() if align_losses else torch.tensor(0.0, device=self.get_device())

                return loss_seg + loss_cls + loss_distill + loss_align

        return final_cls, final_map.squeeze(1)
