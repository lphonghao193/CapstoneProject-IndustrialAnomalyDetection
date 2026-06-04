import torch 
import torch.distributed as dist
import os
import random
import numpy as np
import torch.nn.functional as F
import math

def load_config(args):
    return vars(args)

def setup_distributed(gpus):
    if torch.cuda.is_available() and 'RANK' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        device_str = gpus[local_rank] if local_rank < len(gpus) else f"cuda:{local_rank}"
        torch.cuda.set_device(device_str)
        dist.init_process_group(backend='nccl', init_method='env://')
        return rank, world_size, local_rank, torch.device(device_str)
    
    device = torch.device(gpus[0] if gpus else "cuda")
    return 0, 1, 0, device

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def gaussian_blur_2d(input_tensor: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    if sigma == 0: return input_tensor
    kernel_size = 2 * int(4 * sigma + 0.5) + 1
    
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # calculate kernel
    gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance)
    )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).to(input_tensor.device)
    
    padding = kernel_size // 2
    channels = input_tensor.shape[1]
    kernel = kernel.repeat(channels, 1, 1, 1)
    
    return F.conv2d(input_tensor, kernel, padding=padding, groups=channels)
