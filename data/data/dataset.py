import random
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class AnomalyDetectionDataset(Dataset):
    """
    split options:
      'train' — only training split samples (all normal in MVTec)
      'test'  — only test split samples (normal + anomaly in MVTec)
      'all'   — all samples regardless of split
    """
    def __init__(self, root, dataset, category, split, img_res, dino_size=672):
        super().__init__()

        if split not in ('train', 'test', 'all'):
            raise ValueError(f"split must be 'train', 'test', or 'all', got '{split}'")

        self.dataset = dataset.lower()
        self.category = category.lower()
        self.split = split
        self.img_res = img_res
        self.dino_size = dino_size

        self.root = Path(root)
        json_path = self.root / 'data.json'
        if not json_path.exists():
            raise FileNotFoundError(f"Metadata JSON not found at: {json_path}")

        with open(json_path, 'r') as f:
            self.full_metadata = json.load(f)

        if self.dataset not in self.full_metadata:
            raise ValueError(
                f"Dataset '{self.dataset}' not found in JSON. Available: {list(self.full_metadata.keys())}"
            )

        self.dataset_metadata = self.full_metadata[self.dataset]
        self.categories = list(self.dataset_metadata.keys())

        if category == 'all':
            self.categories_to_load = self.categories
        elif category in self.categories:
            self.categories_to_load = [category]
        else:
            raise ValueError(
                f"Invalid category: {category}. Must be 'all' or one of {self.categories}"
            )

        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []

        for cat in self.categories_to_load:
            cat_samples = self.dataset_metadata[cat]

            for s in cat_samples:
                if self.split == 'train' and not s['is_train']:
                    continue
                if self.split == 'test' and s['is_train']:
                    continue

                image_path = self.root / s['image_path']
                mask_path = s['mask_path']
                label = 0 if s['is_normal'] else 1

                if mask_path and mask_path != "None":
                    mask_full_path = self.root / mask_path
                else:
                    mask_full_path = None

                samples.append((image_path, mask_full_path, label, cat))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label, category_name = self.samples[idx]

        # 1. Load image
        image = cv2.imread(str(img_path))
        if image is None:
            image = np.zeros((self.img_res, self.img_res, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        H, W = image.shape[:2]

        # 2. Load mask at original resolution
        if mask_path is not None:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((H, W), dtype=np.uint8)
        else:
            mask = np.zeros((H, W), dtype=np.uint8)

        # 3. Center crop both image and mask to square
        shorter_side = min(H, W)
        h_start = (H - shorter_side) // 2
        w_start = (W - shorter_side) // 2
        crop      = image[h_start:h_start+shorter_side, w_start:w_start+shorter_side]
        mask_crop = mask[h_start:h_start+shorter_side, w_start:w_start+shorter_side]

        # augmentation
        if self.split == 'train':
            crop, mask_crop = self._augment(crop, mask_crop)

        # resize image for CLIP/DINO
        dino_img      = cv2.resize(crop,      (self.dino_size, self.dino_size), interpolation=cv2.INTER_CUBIC)
        image_resized = cv2.resize(crop,      (self.img_res,   self.img_res),   interpolation=cv2.INTER_CUBIC)
        mask_resized  = cv2.resize(mask_crop, (self.img_res,   self.img_res),   interpolation=cv2.INTER_NEAREST)
        mask_resized  = (mask_resized > 0).astype(np.uint8) * 255

        image_t      = torch.from_numpy(image_resized.transpose(2, 0, 1)).float() / 255.0
        image_dino_t = torch.from_numpy(dino_img.transpose(2, 0, 1)).float() / 255.0
        mask_t       = torch.from_numpy(mask_resized).float().unsqueeze(0) / 255.0

        return {
            'image':      image_t,
            'image_dino': image_dino_t,
            'mask':       mask_t,
            'image_path': str(img_path),
            'label':      label,
            'class_name': category_name,
        }

    def _augment(self, image: np.ndarray, mask: np.ndarray):
        H, W = image.shape[:2]

        # Random horizontal flip
        if random.random() < 0.5:
            image = np.fliplr(image).copy()
            mask  = np.fliplr(mask).copy()

        # Random vertical flip
        if random.random() < 0.3:
            image = np.flipud(image).copy()
            mask  = np.flipud(mask).copy()

        # Random rotation ±30° — BORDER_REFLECT for image avoids black edges;
        # BORDER_CONSTANT=0 for mask so introduced pixels are treated as normal.
        if random.random() < 0.5:
            angle = random.uniform(-30, 30)
            M = cv2.getRotationMatrix2D((W // 2, H // 2), angle, 1.0)
            image = cv2.warpAffine(image, M, (W, H),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT_101)
            mask  = cv2.warpAffine(mask,  M, (W, H),
                                   flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)

        # Random zoom in / zoom out (0.8x to 1.25x): allowing the model to learn 
        # both small and large defect variations.
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.25)
            new_size = int(H * scale)
            image_res = cv2.resize(image, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
            mask_res  = cv2.resize(mask,  (new_size, new_size), interpolation=cv2.INTER_NEAREST)
            
            if scale >= 1.0:
                # Zoom in: crop back to original size
                max_off = new_size - H
                ox = random.randint(0, max_off)
                oy = random.randint(0, max_off)
                image = image_res[oy:oy + H, ox:ox + W]
                mask  = mask_res[oy:oy + H, ox:ox + W]
            else:
                # Zoom out: pad back to original size
                pad_h = H - new_size
                pad_w = W - new_size
                top = random.randint(0, pad_h)
                bottom = pad_h - top
                left = random.randint(0, pad_w)
                right = pad_w - left
                
                image = cv2.copyMakeBorder(image_res, top, bottom, left, right, cv2.BORDER_REFLECT_101)
                mask  = cv2.copyMakeBorder(mask_res, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        return image, mask

def get_dataloader(root, dataset, category, split, batch_size, num_workers, img_res, dino_size, shuffle):
    ds = AnomalyDetectionDataset(root=root, dataset=dataset, category=category, split=split, img_res=img_res, dino_size=dino_size)
    return DataLoader(
        ds, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=shuffle, 
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )