import os
import sys
import json
import shutil
import zipfile
import time
from pathlib import Path
import opendatasets as od
import numpy as np
from PIL import Image

# Constants
DATASET_NAME = "dagm"
KAGGLE_URL = "https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection"

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
    
    if (dataset_dir / "Class1").exists():
        print(f"{DATASET_NAME} already exists and is aligned.")
    else:
        setup_kaggle_creds()
        print(f"Downloading {DATASET_NAME} from Kaggle to {base_dir}...")
        od.download(KAGGLE_URL, data_dir=str(base_dir))
        
        # opendatasets folder name
        od_folder = base_dir / "dagm-2007-competition-dataset-optical-inspection"
        if od_folder.exists():
            print(f"Moving contents from {od_folder} to {dataset_dir}...")
            dataset_dir.mkdir(parents=True, exist_ok=True)
            for item in od_folder.iterdir():
                target = dataset_dir / item.name
                if target.exists():
                    if target.is_dir(): shutil.rmtree(target)
                    else: target.unlink()
                shutil.move(str(item), str(target))
            shutil.rmtree(od_folder)

    # Manual Unzip fallback
    zips = list(dataset_dir.glob("**/Class*_def.zip")) + list(dataset_dir.glob("*.zip"))
    if zips:
        for z in zips:
            print(f"Manually unzipping {z}...")
            with zipfile.ZipFile(z, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            z.unlink()

    # Robust Aggressive Flattening
    # Search for Class1 anywhere and move its parent's contents to root
    found_class1 = list(dataset_dir.glob("**/Class1"))
    if found_class1:
        source_dir = found_class1[0].parent
        if source_dir.resolve() != dataset_dir.resolve():
            print(f"Flattening redundant directory structure from {source_dir}...")
            # Move everything from source_dir to dataset_dir
            for item in list(source_dir.iterdir()):
                target = dataset_dir / item.name
                if target.resolve() == item.resolve(): continue # Skip if already at destination
                if target.exists():
                    if target.is_dir(): shutil.rmtree(target)
                    else: target.unlink()
                shutil.move(str(item), str(target))
    
    # Cleanup: Remove any non-category folders at the root
    valid_categories = [f"Class{i}" for i in range(1, 11)]
    for item in list(dataset_dir.iterdir()):
        if item.is_dir() and item.name not in valid_categories:
            print(f"Removing redundant folder: {item.name}")
            shutil.rmtree(item)
        elif item.is_file() and not item.name.lower().endswith(('.png', '.jpg', '.jpeg', '.txt')):
             # Remove auxiliary files like Thumbs.db
             if item.name != "how_to_cite.txt": item.unlink()

    # Remove training sets as requested
    for cat in os.listdir(dataset_dir):
        cat_path = dataset_dir / cat
        if cat_path.is_dir() and cat.startswith("Class"):
            for train_name in ["Train", "train"]:
                train_dir = cat_path / train_name
                if train_dir.exists():
                    print(f"Removing training set for {DATASET_NAME}/{cat}...")
                    shutil.rmtree(train_dir)

def build_index(base_dir):
    dataset_dir = Path(base_dir) / DATASET_NAME
    if not dataset_dir.exists(): return {}
    
    categories = [d for d in os.listdir(dataset_dir) if os.path.isdir(dataset_dir / d) and d.startswith("Class")]
    index = {}
    for cat in categories:
        cat_dir = dataset_dir / cat
        cat_samples = []
        for test_name in ["Test", "test"]:
            test_dir = cat_dir / test_name
            if test_dir.exists():
                for root, dirs, files in os.walk(test_dir):
                    if "Label" in root: continue
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_abs = Path(root) / f
                            img_rel = img_abs.relative_to(dataset_dir)
                            mask_path = None
                            is_normal = True
                            label_dir = Path(root) / "Label"
                            if not label_dir.exists():
                                label_dir = Path(root).parent / "Label"
                            if label_dir.exists():
                                found_mask_file = None
                                if (label_dir / f).exists(): 
                                    found_mask_file = label_dir / f
                                else:
                                    # Try _label suffix with correct extension
                                    ext = Path(f).suffix
                                    label_file = f.replace(ext, f"_label{ext}")
                                    if (label_dir / label_file).exists(): 
                                        found_mask_file = label_dir / label_file
                                
                                if found_mask_file:
                                    mask_rel = found_mask_file.relative_to(dataset_dir)
                                    mask_path = Path(f"{DATASET_NAME}/{mask_rel.as_posix()}").as_posix()
                                    # In DAGM, only abnormal images have a label file
                                    is_normal = False
                            
                            cat_samples.append({
                                "image_path": f"{DATASET_NAME}/{img_rel.as_posix()}",
                                "mask_path": mask_path,
                                "is_normal": is_normal,
                                "is_train": False
                            })
                break
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
        except FileExistsError: time.sleep(0.1)
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
        if os.path.exists(lock_path): os.remove(lock_path)
    print(f"Updated {DATASET_NAME} in {data_json_path}.")

if __name__ == "__main__":
    root = get_project_root()
    default_base = root / "data" / "data"
    default_json = root / "data" / "data" / "data.json"
    base = sys.argv[1] if len(sys.argv) > 1 else default_base
    json_p = sys.argv[2] if len(sys.argv) > 2 else default_json
    run(base, json_p)
