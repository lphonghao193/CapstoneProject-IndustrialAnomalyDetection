import os
import json
import random
import logging
import argparse
import warnings
import sys
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler
from rich.progress import Progress, BarColumn, TextColumn
import wandb

from model.model import ProposeModel
from data.data.dataset import AnomalyDetectionDataset
from utils.utils import load_config, setup_distributed, seed_everything

def parser_args():
    parser = argparse.ArgumentParser(description='Configuration')
    parser.add_argument('--dataset', type=str, default='visa', choices=['mvtec', 'visa', 'btad', 'dagm', 'mpdd', 'dtd'], help='Dataset name')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test', 'all'], help='Dataset split')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size per GPU')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--gpus', type=str, nargs='+', default=['cuda:0'], help='GPU ranks (e.g. --gpus cuda:0 cuda:1)')
    return parser.parse_args()

def save_checkpoint(model: torch.nn.Module, epoch: int, loss: float, config: dict, filename: str = None):
    seed = config.get('seed', 'unknown')
    if filename is None:
        filename = f"model_{seed}.pth"
    raw_model = model.module if hasattr(model, "module") else model
    trainable_keys = {n for n, p in raw_model.named_parameters() if p.requires_grad}
    trainable_state = {k: v.cpu() for k, v in raw_model.state_dict().items() if k in trainable_keys}
    if not trainable_state:
        logging.warning("No trainable parameters found to save!")
    ckpt = {"epoch": epoch, "loss": loss, "model_state": trainable_state, "config": config, "seed": config.get('seed')}
    
    save_dir = f'./checkpoints/train_{config["dataset"].lower()}'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(ckpt, os.path.join(save_dir, filename))
    logging.info(f"Saved: {filename} (Loss: {loss:.4f})")

def train(model: torch.nn.Module, train_dataset: torch.utils.data.Dataset, config: dict, device: torch.device, wandb_run=None):
    rank = config.get("rank", 0)
    world_size = config.get("world_size", 1)
    is_main = rank == 0

    sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=(sampler is None),
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
        persistent_workers=(config.get('num_workers', 4) > 0),
        prefetch_factor=2 if config.get('num_workers', 4) > 0 else None,
    )

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=config.get('lr', 1e-4), weight_decay=1e-5)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=config.get('lr', 1e-4), epochs=config.get('epochs', 10), steps_per_epoch=len(loader))
    scaler = GradScaler('cuda')
    best_loss = float('inf')
    log_every = max(1, len(loader) // 10)

    try:
        for epoch in range(config.get('epochs', 10)):
            if sampler:
                sampler.set_epoch(epoch)
            model.train()
            total_loss = 0.0
            if is_main:
                progress = Progress(
                    TextColumn(f"Ep {epoch+1}"), BarColumn(),
                    TextColumn("{task.completed}/{task.total} Loss: {task.fields[loss]:.4f}"),
                    transient=True,
                )
                progress.start()
                task = progress.add_task("train", total=len(loader), loss=0.0)

            for step, batch in enumerate(loader):
                optimizer.zero_grad(set_to_none=True)
                imgs = batch["image"].to(device, non_blocking=True)
                imgs_dino = batch["image_dino"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                with autocast('cuda'):
                    loss = model({"image": imgs, "image_dino": imgs_dino, "label": labels, "mask": masks, "class_name": batch["class_name"]})

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                scaler.step(optimizer)
                scale_before = scaler.get_scale()
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()

                loss_val = loss.item()
                total_loss += loss_val
                if is_main:
                    progress.update(task, advance=1, loss=loss_val)
                    if wandb_run and step % log_every == 0:
                        wandb_run.log({"batch_loss": loss_val, "lr": scheduler.get_last_lr()[0]})

            if is_main:
                progress.stop()
                avg_loss = total_loss / len(loader)
                print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
                if wandb_run:
                    wandb_run.log({"epoch_loss": avg_loss, "epoch": epoch + 1})
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_checkpoint(model, epoch + 1, best_loss, config)

            if world_size > 1:
                dist.barrier()
    finally:
        del loader

def main():
    args = parser_args()
    
    # setup variables and configuration
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'data'))
    dino_size, img_res, num_workers, sigma, vv_path = 672, 336, 4, 4.0, True
    config = load_config(args)
    rank, world_size, local_rank, device = setup_distributed(args.gpus)
    config.update({'rank': rank, 'world_size': world_size, 'local_rank': local_rank, 'device': str(device), 'data_path': data_path, 'dino_size': dino_size, 'vv_path': vv_path, 'img_res': img_res, 'num_workers': num_workers, 'sigma': sigma})
    
    if config.get('seed') is None:
        config['seed'] = random.randint(0, 1000)
        if rank == 0:
            print(f"[INFO] No seed provided. Generated random seed: {config['seed']}")
    else:
        if rank == 0:
            print(f"[INFO] Using explicit seed: {config['seed']}")
            
    seed_everything(config['seed'])

    model = ProposeModel(**config).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    train_ds = AnomalyDetectionDataset(
        root=data_path,
        dataset=args.dataset,
        category='all',
        split=args.split,
        img_res=img_res,
        dino_size=dino_size,
    )

    run = None
    if rank == 0:
        run = wandb.init(project="Specialized Project", config=config,
                         name=f"train_{args.dataset.lower()}")

    train(model, train_ds, config, device, wandb_run=run)
    if run: run.finish()
    if dist.is_initialized(): dist.destroy_process_group()

if __name__ == "__main__":
    main()
