import os
import sys
import json
import shutil
import time
import subprocess
from pathlib import Path
import requests
from tqdm import tqdm
import concurrent.futures

# Constants
DATASET_NAME = "mvtec"
FOLDER_NAME = "mvtec_anomaly_detection"
URL = "https://www.mydrive.ch/shares/150996/b52ecdcbf521176e9db9c731f2304b27/download/420938113-1629960298/mvtec_anomaly_detection.tar.xz"

def get_project_root():
    return Path(__file__).resolve().parent.parent.parent

def download_file(url, dest, threads=16):
    print(f"Downloading {url} to {dest} with {threads} threads...")
    r = requests.head(url, allow_redirects=True)
    try: file_size = int(r.headers.get('content-length', 0))
    except: file_size = 0
    if file_size == 0:
        response = requests.get(url, stream=True)
        with open(dest, 'wb') as f:
            for data in response.iter_content(1024*1024): f.write(data)
        return
    with open(dest, "wb") as f: f.truncate(file_size)
    chunk_size = file_size // threads
    def download_range(start, end, dest_path, pbar):
        headers = {'Range': f'bytes={start}-{end}'}
        with requests.Session() as s:
            response = s.get(url, headers=headers, stream=True)
            with open(dest_path, "r+b") as f:
                f.seek(start)
                for chunk in response.iter_content(256*1024):
                    if chunk: f.write(chunk); pbar.update(len(chunk))
    with tqdm(total=file_size, unit='iB', unit_scale=True, desc=dest.name) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(download_range, i*chunk_size, ((i+1)*chunk_size-1 if i<threads-1 else file_size-1), dest, pbar) for i in range(threads)]
            for future in concurrent.futures.as_completed(futures):
                future.result()

def download_and_extract(base_dir):
    dataset_dir = Path(base_dir) / FOLDER_NAME
    dataset_dir.mkdir(parents=True, exist_ok=True)
    tar_path = Path(base_dir) / "mvtec_anomaly_detection.tar.xz"
    if (dataset_dir / "bottle").exists(): return
    download_file(URL, tar_path)
    print(f"Extracting {DATASET_NAME}...")
    try: subprocess.run(['tar', '-xf', str(tar_path), '-C', str(dataset_dir)], check=True)
    except: pass
    if tar_path.exists(): tar_path.unlink()
    # Handle Nesting
    nested_dir = dataset_dir / "mvtec_anomaly_detection"
    if nested_dir.exists():
        for item in nested_dir.iterdir():
            target = dataset_dir / item.name
            if target.exists():
                if target.is_dir(): shutil.rmtree(target)
                else: target.unlink()
            shutil.move(str(item), str(target))
        nested_dir.rmdir()

def build_index(base_dir):
    dataset_dir = Path(base_dir) / FOLDER_NAME
    if not dataset_dir.exists(): return {}
    categories = [d for d in os.listdir(dataset_dir) if os.path.isdir(dataset_dir / d)]
    index = {}
    for cat in categories:
        cat_dir = dataset_dir / cat
        cat_samples = []
        train_good = cat_dir / "train" / "good"
        if train_good.exists():
            for f in os.listdir(train_good):
                if f.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_path = Path(f"{FOLDER_NAME}/{cat}/train/good/{f}").as_posix()
                    cat_samples.append({"image_path": image_path, "mask_path": None, "is_normal": True, "is_train": True})
        test_dir = cat_dir / "test"
        if test_dir.exists():
            for label in os.listdir(test_dir):
                label_dir = test_dir / label
                if not os.path.isdir(label_dir): continue
                is_normal = (label == "good")
                for f in os.listdir(label_dir):
                    if f.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        mask_path = None
                        if not is_normal:
                            gt_dir = cat_dir / "ground_truth" / label
                            mask_file = f.replace(".png", "_mask.png")
                            if not (gt_dir / mask_file).exists(): mask_file = f
                            if (gt_dir / mask_file).exists():
                                mask_path = Path(f"{FOLDER_NAME}/{cat}/ground_truth/{label}/{mask_file}").as_posix()
                        image_path = Path(f"{FOLDER_NAME}/{cat}/test/{label}/{f}").as_posix()
                        cat_samples.append({"image_path": image_path, "mask_path": mask_path, "is_normal": is_normal, "is_train": False})
        if cat_samples: index[cat] = cat_samples
    return index

def run(base_dir, data_json_path):
    download_and_extract(base_dir)
    new_index = build_index(base_dir)
    if not new_index: return
    
    data_json_path = Path(data_json_path).resolve()
    lock_path = data_json_path.with_suffix(".json.lock")
    
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            time.sleep(0.1)

    try:
        data = {}
        if data_json_path.exists() and data_json_path.stat().st_size > 0:
            with open(data_json_path, 'r') as f:
                try: data = json.load(f)
                except: data = {}
        data[DATASET_NAME] = new_index
        with open(data_json_path, 'w') as f:
            json.dump(data, f, indent=2)
    finally:
        if os.path.exists(lock_path):
            os.remove(lock_path)
    print(f"Updated {DATASET_NAME} in {data_json_path}.")

if __name__ == "__main__":
    root = get_project_root()
    default_base = root / "data" / "data"
    default_json = root / "data" / "data" / "data.json"
    base = sys.argv[1] if len(sys.argv) > 1 else default_base
    json_p = sys.argv[2] if len(sys.argv) > 2 else default_json
    run(base, json_p)
