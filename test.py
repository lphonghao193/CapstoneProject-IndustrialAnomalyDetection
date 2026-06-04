import os
import json
import random
import argparse
import warnings
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import wandb

from model.model import ProposeModel
from data.data.dataset import AnomalyDetectionDataset
from utils.metrics import (
    compute_image_auroc, compute_image_ap, compute_image_f1_max,
    compute_pixel_auroc, compute_pixel_ap, compute_pixel_f1_metrics, compute_pixel_pro
)
from utils.utils import load_config, setup_distributed

warnings.filterwarnings("ignore", message=".*SHA256 checksum.*")

def parser_args():
    parser = argparse.ArgumentParser(description='Configuration')
    parser.add_argument('--dataset', type=str, default='visa', choices=[
        'mvtec', 'visa', 'btad', 'dagm', 'mpdd', 'dtd',
    ], help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size per GPU')
    parser.add_argument('--sigma', type=float, default=4.0, help='Gaussian blur sigma for predictions (0 to disable)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--gpus', type=str, nargs='+', default=['cuda:0'], help='GPU ranks (e.g. --gpus cuda:0 cuda:1)')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/best_model.pth', help='Path to the checkpoint file to load')
    parser.add_argument('--n_vis', type=int, default=50, help='Number of images to visualize per category')
    return parser.parse_args()

def load_weights(model, ckpt_path, rank=0):
    if not os.path.exists(ckpt_path):
        if rank == 0: print(f"[Warning] Path not found: {ckpt_path}")
        return False, None
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    raw = model.module if hasattr(model, 'module') else model
    ckpt_seed = ckpt.get('seed') or (ckpt.get('config') or {}).get('seed')
    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        model_dict = raw.state_dict()
        v_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        raw.load_state_dict(v_dict, strict=False)
        if rank == 0: print(f"[INFO] Loaded weights.")
    return True, ckpt_seed

def save_vis(samples, cat_name, run=None):
    if not run: return
    # Log images in chunks to avoid overwhelming the WandB backend
    chunk_size = 100
    for chunk_idx in range(0, len(samples), chunk_size):
        chunk = samples[chunk_idx : chunk_idx + chunk_size]
        wandb_images = []
        for i, s in enumerate(chunk):
            # raw image
            orig = (s["img"].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            H, W = orig.shape[:2]
            
            # ground truth mask
            gt_raw = s["mask"].squeeze().cpu().numpy()
            gt = (gt_raw > 0).astype(np.uint8) * 255
            gt_rgb = cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)
            
            # overlay heatmap
            amap_raw = s["amap"].cpu().squeeze().numpy()
            amap_norm = (amap_raw - amap_raw.min()) / (amap_raw.max() - amap_raw.min() + 1e-8)
            hmap = cv2.applyColorMap((amap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            hmap = cv2.resize(hmap, (W, H))
            hmap_rgb = cv2.cvtColor(hmap, cv2.COLOR_BGR2RGB)
            
            blend = cv2.addWeighted(orig, 0.7, hmap_rgb, 0.3, 0)
            
            combined = np.concatenate([orig, hmap_rgb, gt_rgb, blend], axis=1)
            wandb_images.append(wandb.Image(combined, caption=f"{cat_name}: {s.get('image_name')}; GT Anomaly: {'Yes' if s['label'] else 'No'}"))
        
        if wandb_images:
            run.log({f"visuals/{cat_name}": wandb_images})

def test(model: torch.nn.Module, test_loader: DataLoader, device: torch.device, config: dict, run=None, n_vis: int = 0):
    model.eval()
    img_preds, img_lbls, pix_preds, pix_lbls, vis_samples = [], [], [], [], []
    flip_fns = [lambda x: x, lambda x: x.flip(-1), lambda x: x.flip(-2)]
    with torch.no_grad():
        for batch in test_loader:
            img = batch["image"].to(device, non_blocking=True)
            img_dino = batch["image_dino"].to(device, non_blocking=True)
            tta_scores, tta_maps = [], []
            for flip_fn in flip_fns:
                s, m = model({"image": flip_fn(img), "image_dino": flip_fn(img_dino), "class_name": batch["class_name"]})
                tta_scores.append(s)
                tta_maps.append(flip_fn(m))
            score_cls = torch.stack(tta_scores).mean(0)
            map_anom = torch.stack(tta_maps).mean(0)
            img_preds.extend(score_cls.cpu().tolist())
            img_lbls.extend(batch["label"].cpu().tolist())
            maps_np = map_anom.cpu().numpy()
            for i in range(score_cls.shape[0]):
                gt = (batch["mask"][i].squeeze().numpy() > 0.5).astype(np.uint8)
                pred = maps_np[i]
                if pred.shape != gt.shape: pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
                pix_preds.append(pred)
                pix_lbls.append(gt)
                if n_vis > 0 and len(vis_samples) < n_vis:
                    img_paths = batch.get("image_path")
                    img_name = os.path.basename(img_paths[i]) if img_paths else f"res_{i}.png"
                    vis_samples.append({"img": batch["image"][i], "mask": batch["mask"][i], "amap": map_anom[i], "image_name": img_name, "label": batch["label"][i]})
    return img_preds, img_lbls, pix_preds, pix_lbls, vis_samples

def main():
    args = parser_args()
    
    # setup variables and configuration
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'data'))
    dino_size, img_res, num_workers, vv_path = 672, 336, 4, True
    config = load_config(args)
    rank, world_size, local_rank, device = setup_distributed(args.gpus)
    config.update({'rank': rank, 'world_size': world_size, 'device': str(device), 'data_path': data_path, 'dino_size': dino_size, 'vv_path': vv_path, 'img_res': img_res, 'num_workers': num_workers})
    
    # load model
    model = ProposeModel(**config).to(device)
    if world_size > 1: model = DDP(model, device_ids=[local_rank])
    
    # load visa checkpoint for mvtec/mvtec checkpoint for others
    source = "visa" if args.dataset.lower() == "mvtec" else "mvtec"
    ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), config["ckpt_path"]))
    loaded, ckpt_seed = load_weights(model, ckpt_path, rank)
    if config.get('seed') is None: config['seed'] = ckpt_seed if ckpt_seed is not None else random.randint(0, 1000)
    
    # setup wandb and metrics storage
    run = None
    if rank == 0: run = wandb.init(project="EVALUATION", config=config, name=f"test_{source}_to_{args.dataset.lower()}")
    
    # dataset
    all_metrics = {k: [] for k in ["i_auc", "i_ap", "i_f1", "p_auc", "p_ap", "p_f1", "p_pro"]}
    cat_results = {}
    categories = AnomalyDetectionDataset(root=data_path, dataset=args.dataset, category='all', split='test', img_res=img_res, dino_size=dino_size).categories
    
    if rank == 0:
        print(f"\nEvaluating {args.dataset.upper()} (Source: {source.upper()})")
        print("-" * 155)
        print(f"| {'Category':<15} | {'Img AUC':<10} | {'Img AP':<10} | {'Img F1':<10} | {'Pix AUC':<10} | {'Pix AP':<10} | {'Pix F1':<10} | {'Pix PRO':<10} |")
        print("-" * 155)
        
    for cat in categories:
        ds = AnomalyDetectionDataset(root=data_path, dataset=args.dataset, category=cat, split='test', img_res=img_res, dino_size=dino_size)
        loader = DataLoader(ds, batch_size=config.get('batch_size', 1), shuffle=False, num_workers=num_workers, pin_memory=True, sampler=DistributedSampler(ds, shuffle=False) if world_size > 1 else None)
        
        # Collect local predictions
        img_preds_local, img_lbls_local, pix_preds_local, pix_lbls_local, vis = test(model, loader, device, config, run, n_vis=(args.n_vis if rank == 0 else 0))
        del loader
        
        # Gather results from all ranks in DDP mode
        if world_size > 1:
            # We use a list to gather objects from all ranks
            gather_list = [None for _ in range(world_size)]
            local_data = {
                "img_preds": img_preds_local,
                "img_lbls": img_lbls_local,
                "pix_preds": pix_preds_local,
                "pix_lbls": pix_lbls_local
            }
            dist.all_gather_object(gather_list, local_data)
            
            if rank == 0:
                img_preds, img_lbls, pix_preds, pix_lbls = [], [], [], []
                for data in gather_list:
                    img_preds.extend(data["img_preds"])
                    img_lbls.extend(data["img_lbls"])
                    pix_preds.extend(data["pix_preds"])
                    pix_lbls.extend(data["pix_lbls"])
            else:
                img_preds, img_lbls, pix_preds, pix_lbls = [], [], [], []
        else:
            img_preds, img_lbls, pix_preds, pix_lbls = img_preds_local, img_lbls_local, pix_preds_local, pix_lbls_local

        if rank == 0:
            if not img_preds: continue
            
            img_p, img_l = torch.tensor(img_preds), torch.tensor(img_lbls)
            pix_p, pix_l = torch.tensor(np.array(pix_preds)), torch.tensor(np.array(pix_lbls))
            f1, _, _ = compute_pixel_f1_metrics(pix_p, pix_l)
            
            met = {
                "i_auc": compute_image_auroc(img_p, img_l) * 100, "i_ap": compute_image_ap(img_p, img_l) * 100, "i_f1": compute_image_f1_max(img_p, img_l) * 100,
                "p_auc": compute_pixel_auroc(pix_p, pix_l) * 100, "p_ap": compute_pixel_ap(pix_p, pix_l) * 100, "p_f1": f1 * 100, "p_pro": compute_pixel_pro(pix_p, pix_l) * 100
            }
            
            if vis: save_vis(vis, cat, run=run)
            print(f"| {cat:<15} | {met['i_auc']:10.1f} | {met['i_ap']:10.1f} | {met['i_f1']:10.1f} | {met['p_auc']:10.1f} | {met['p_ap']:10.1f} | {met['p_f1']:10.1f} | {met['p_pro']:10.1f} |")
            for k in all_metrics.keys(): all_metrics[k].append(met[k])
            cat_results[cat] = met
        
        # Barrier to keep all ranks synchronized per category
        if world_size > 1:
            dist.barrier()
            
    if rank == 0:
        print("-" * 155)
        avgs = {k: np.mean(v) for k, v in all_metrics.items()}
        cat_results["AVERAGE"] = avgs
        print(f"| {'AVERAGE':<15} | {avgs['i_auc']:10.1f} | {avgs['i_ap']:10.1f} | {avgs['i_f1']:10.1f} | {avgs['p_auc']:10.1f} | {avgs['p_ap']:10.1f} | {avgs['p_f1']:10.1f} | {avgs['p_pro']:10.1f} |")
        print("=" * 155)
        results_dir = os.path.join("results", args.dataset)
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, f"{config['seed']}.json"), "w") as f: json.dump(cat_results, f, indent=4)
        if run: run.finish()
    
    # Final barrier to ensure Rank 0 finishes serial tasks before other ranks exit
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
