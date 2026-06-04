import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, logging
import transformers.utils.logging as hf_logging

class FocalLoss(nn.Module):  
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = inputs.float() 
        targets = targets.float()
        epsilon = 1e-6
        p = torch.clamp(p, epsilon, 1. - epsilon)
        
        loss_pos = -self.alpha * torch.pow(1. - p, self.gamma) * torch.log(p)
        loss_neg = -(1 - self.alpha) * torch.pow(p, self.gamma) * torch.log(1. - p)
        
        loss = targets * loss_pos + (1 - targets) * loss_neg
        
        if self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum': return loss.sum()
        else: return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = predictions.view(predictions.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        
        intersection = (predictions * targets).sum(1)
        union = predictions.sum(1) + targets.sum(1)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1. - dice.mean()

class RelativeDistillationLoss(nn.Module):
    def __init__(self, deberta_name='microsoft/deberta-v3-base', temp=0.07):
        super().__init__()
        self.temp = temp
        try:
            # omit unnecessary logging
            current_verbosity = logging.get_verbosity()
            logging.set_verbosity_error()
            hf_logging.disable_progress_bar()
            
            # load teacher model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(deberta_name)
            self.deberta = AutoModel.from_pretrained(deberta_name)
            
            # restore original logging verbosity
            logging.set_verbosity(current_verbosity)
            hf_logging.enable_progress_bar()
            
            # frozen teacher
            for p in self.deberta.parameters(): p.requires_grad = False
            self.deberta.eval()
        except Exception:
            self.tokenizer = None
            self.deberta = None

    def forward(self, clip_features: torch.Tensor, texts: list):
        if self.deberta is None: return torch.tensor(0.0, device=clip_features.device)
        
        # move to same device    
        device = clip_features.device
        if not hasattr(self, '_deberta_device') or self._deberta_device != device:
            self.deberta = self.deberta.to(device)
            self._deberta_device = device

        # encoder list of prompts (w/o learnable tokens)
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = self.deberta(**inputs)
            deberta_features = outputs.last_hidden_state[:, 0, :]
        
        # norm    
        deberta_features = F.normalize(deberta_features, dim=-1).to(clip_features.dtype)
        clip_features = F.normalize(clip_features, dim=-1)
        
        # generate similarity matrices
        sim_deb = torch.matmul(deberta_features, deberta_features.t()) / self.temp
        sim_clip = torch.matmul(clip_features, clip_features.t()) / self.temp
        
        
        # kl divergence between similarity distributions/distillation loss
        log_prob_clip = F.log_softmax(sim_clip, dim=-1)
        prob_deb = F.softmax(sim_deb, dim=-1)
        loss = F.kl_div(log_prob_clip, prob_deb, reduction='batchmean')
        return loss

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCELoss()

    def forward(self, inputs, targets):
        return self.criterion(inputs.float(), targets.float())

class AlignmentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, student, teacher, mask):
        # align dinov3 with clip features at patch level
        # learning on normal patches, ignore anomalous ones (mask=0 for anomaly)
        per_patch = (1.0 - (student * teacher).sum(dim=-1))  # [B, T]
        n_normal  = mask.sum().clamp(min=1.0)
        return (per_patch * mask).sum() / n_normal
