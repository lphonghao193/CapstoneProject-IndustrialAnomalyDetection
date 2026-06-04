import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeighborAggregator(nn.Module):
    def __init__(self, embed_dim: int = 768, sigma: float = 4.0):
        super().__init__()
        self.conv3 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False)
        self.conv5 = nn.Conv2d(embed_dim, embed_dim, kernel_size=5, padding=2, groups=embed_dim, bias=False)
        self._init_from_gaussian(embed_dim, sigma)

    @staticmethod
    def _make_gaussian(size: int, sigma: float) -> torch.Tensor:
        ax = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
        y, x = torch.meshgrid(ax, ax, indexing='ij')
        k = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        return k / k.sum()

    def _init_from_gaussian(self, embed_dim: int, sigma: float):
        k3 = self._make_gaussian(3, sigma)
        k5 = self._make_gaussian(5, sigma)
        self.conv3.weight.data.copy_(k3.unsqueeze(0).unsqueeze(0).expand(embed_dim, 1, 3, 3))
        self.conv5.weight.data.copy_(k5.unsqueeze(0).unsqueeze(0).expand(embed_dim, 1, 5, 5))

    def forward(self, patches: torch.Tensor):
        B, T, D = patches.shape
        h = w = int(math.sqrt(T))
        x = patches.permute(0, 2, 1).reshape(B, D, h, w)
        scale1 = patches
        scale3 = self.conv3(x.to(self.conv3.weight.dtype)).reshape(B, D, T).permute(0, 2, 1)
        scale5 = self.conv5(x.to(self.conv5.weight.dtype)).reshape(B, D, T).permute(0, 2, 1)
        return [scale1, scale3, scale5]

class MapFuser(nn.Module):
    def __init__(self):
        super().__init__()
        self.balancer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2, 8, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(8, 2, kernel_size=1)
        )
        
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        
        nn.init.constant_(self.conv2.weight, 0.0)
        nn.init.constant_(self.conv2.bias, 0.0)

    def forward(self, clip_map, dino_map):
        x = torch.cat([clip_map, dino_map], dim=1)
        
        weights = F.softmax(self.balancer(x), dim=1)
        base_map = (weights[:, 0:1] * clip_map) + (weights[:, 1:2] * dino_map)
        
        res = self.conv2(self.act(self.conv1(x)))
        return base_map + res
