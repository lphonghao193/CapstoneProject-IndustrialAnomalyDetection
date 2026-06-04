import os
import sys
import json
import shutil
import time
from pathlib import Path
import opendatasets as od
from dotenv import load_dotenv

load_dotenv()

# Constants
DATASET_NAME = "dtd"
KAGGLE_URL = "https://www.kaggle.com/datasets/lephonghao/dtd-synthetic"

def get_project_root():
    return Path(__file__).resolve().parent.parent.parent

def setup_kaggle_creds():
    username = os.getenv('KAGGLE_USERNAME')
    key = os.getenv('KAGGLE_KEY')
    if username and key:
        creds = {"username": username, "key": key}
        with open('kaggle.json', 'w') as f:
            json.dump(creds, f)

def download_and_extract(base_dir):
    base_dir = Path(base_dir).resolve()
    dataset_dir = base_dir / DATASET_NAME
    
    # We check if a known object directory exists to verify if already downloaded
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        print(f"{DATASET_NAME} already exists.")
    else:
        setup_kaggle_creds()
        print(f"Downloading {DATASET_NAME} from Kaggle to {base_dir}...")
        od.download(KAGGLE_URL, data_dir=str(base_dir))
        
        # opendatasets folder name
        od_folder = base_dir / "dtd-synthetic"
        if od_folder.exists():
            print(f"Moving contents from {od_folder} to {dataset_dir}...")
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # The structure from kaggle might have a "DTD-Synthetic" folder inside
            items = list(od_folder.iterdir())
            if len(items) == 1 and items[0].is_dir() and items[0].name.lower() == "dtd-synthetic":
                source_dir = items[0]
            else:
                source_dir = od_folder
                
            for item in source_dir.iterdir():
                target = dataset_dir / item.name
                if target.exists():
                    if target.is_dir(): shutil.rmtree(target)
                    else: target.unlink()
                shutil.move(str(item), str(target))
            shutil.rmtree(od_folder)

    # Remove training sets as requested
    if dataset_dir.exists():
        for cat in os.listdir(dataset_dir):
            cat_path = dataset_dir / cat
            if cat_path.is_dir():
                train_dir = cat_path / "train"
                if train_dir.exists():
                    print(f"Removing training set for {DATASET_NAME}/{cat}...")
                    shutil.rmtree(train_dir)

def build_index(base_dir):
    dataset_dir = Path(base_dir) / DATASET_NAME
    if not dataset_dir.exists(): return {}
    
    categories = [d for d in os.listdir(dataset_dir) if os.path.isdir(dataset_dir / d)]
    index = {}
    for cat in categories:
        cat_dir = dataset_dir / cat
        cat_samples = []
        
        test_dir = cat_dir / "test"
        if test_dir.exists():
            for label in os.listdir(test_dir):
                label_dir = test_dir / label
                if not os.path.isdir(label_dir): continue
                is_normal = (label == "good")
                for f in os.listdir(label_dir):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        mask_path = None
                        if not is_normal:
                            gt_dir = cat_dir / "ground_truth" / label
                            # e.g., 000_mask.png
                            mask_file = f.replace(Path(f).suffix, f"_mask{Path(f).suffix}")
                            if not (gt_dir / mask_file).exists():
                                mask_file = f # Fallback
                            if (gt_dir / mask_file).exists():
                                mask_path = Path(f"{DATASET_NAME}/{cat}/ground_truth/{label}/{mask_file}").as_posix()
                        
                        image_path = Path(f"{DATASET_NAME}/{cat}/test/{label}/{f}").as_posix()
                        cat_samples.append({
                            "image_path": image_path,
                            "mask_path": mask_path,
                            "is_normal": is_normal,
                            "is_train": False
                        })
        if cat_samples: index[cat] = cat_samples
    return index

def run(base_dir, data_json_path):
    download_and_extract(base_dir)
    new_index = build_index(base_dir)
    if not new_index:
        print(f"No samples found for {DATASET_NAME}. Index not updated.")
        return
    
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
