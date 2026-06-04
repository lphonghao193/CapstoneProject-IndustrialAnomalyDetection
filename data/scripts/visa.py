import os
import sys
import json
import shutil
import csv
import time
import subprocess
from pathlib import Path
import requests
from tqdm import tqdm
import concurrent.futures

# Add project root for utils
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.append(str(_project_root))

# Constants
DATASET_NAME = "visa"
URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"

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
    dataset_dir = Path(base_dir) / DATASET_NAME
    dataset_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dataset_dir / "VisA_20220922.tar"
    if (dataset_dir / "candle").exists(): return
    download_file(URL, tar_path)
    print(f"Extracting {DATASET_NAME}...")
    try: subprocess.run(['tar', '-xf', str(tar_path), '-C', str(dataset_dir)], check=True)
    except: pass
    if tar_path.exists(): tar_path.unlink()
    nested_dir = dataset_dir / "VisA_20220922"
    if nested_dir.exists():
        for item in nested_dir.iterdir():
            target = dataset_dir / item.name
            if target.exists():
                if target.is_dir(): shutil.rmtree(target)
                else: target.unlink()
            shutil.move(str(item), str(target))
        nested_dir.rmdir()

def build_index(base_dir):
    dataset_dir = Path(base_dir) / DATASET_NAME
    if not dataset_dir.exists(): return {}
    split_file = dataset_dir / "split_csv" / "1cls.csv"
    if not split_file.exists(): return {}
    index = {}
    with open(split_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row['object']
            if cat not in index: index[cat] = []
            is_normal = (row['label'] == 'normal')
            is_train = (row['split'] == 'train')
            image_path = Path(f"{DATASET_NAME}/{row['image']}").as_posix()
            mask_path = Path(f"{DATASET_NAME}/{row['mask']}").as_posix() if row['mask'] else None
            index[cat].append({"image_path": image_path, "mask_path": mask_path, "is_normal": is_normal, "is_train": is_train})
    return index

def run(base_dir, data_json_path):
    download_and_extract(base_dir)
    new_index = build_index(base_dir)
    if not new_index: return
    
    data_json_path = Path(data_json_path).resolve()
    lock_path = data_json_path.with_suffix(".json.lock")
    
    # Simple File Lock for concurrent updates
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
    base = sys.argv[1] if len(sys.argv) > 1 else root / "data" / "data"
    json_p = sys.argv[2] if len(sys.argv) > 2 else root / "data" / "data" / "data.json"
    run(base, json_p)
