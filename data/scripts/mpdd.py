import os
import sys
import json
import shutil
import zipfile
import time
from pathlib import Path
import opendatasets as od

# Constants
DATASET_NAME = "mpdd"
KAGGLE_URL = "https://www.kaggle.com/datasets/lephonghao/metal-parts-defect-detection-dataset-mpdd"

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
    if dataset_dir.exists() and any(d.lower() in ["bracket_black", "connector"] for d in os.listdir(dataset_dir)):
        print(f"{DATASET_NAME} already exists and is aligned.")
    else:
        setup_kaggle_creds()
        print(f"Downloading {DATASET_NAME} from Kaggle to {base_dir}...")
        od.download(KAGGLE_URL, data_dir=str(base_dir))
        od_folder = base_dir / "metal-parts-defect-detection-dataset-mpdd"
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
    zips = list(dataset_dir.glob("*.zip"))
    if zips:
        for z in zips:
            print(f"Manually unzipping {z}...")
            with zipfile.ZipFile(z, 'r') as zip_ref: zip_ref.extractall(dataset_dir)
            z.unlink()

    # Robust Aggressive Flattening
    # Search for "bracket_black" category anywhere and move its parent's contents to root
    found_indicator = list(dataset_dir.glob("**/bracket_black"))
    if found_indicator:
        source_dir = found_indicator[0].parent
        if source_dir.resolve() != dataset_dir.resolve():
            print(f"Flattening redundant directory structure from {source_dir}...")
            for item in list(source_dir.iterdir()):
                target = dataset_dir / item.name
                if target.resolve() == item.resolve(): continue
                if target.exists():
                    if target.is_dir(): shutil.rmtree(target)
                    else: target.unlink()
                shutil.move(str(item), str(target))

    # Cleanup: Remove any non-category folders at root
    valid_indicators = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
    for item in list(dataset_dir.iterdir()):
        if item.is_dir() and item.name not in valid_indicators:
            print(f"Removing redundant folder: {item.name}")
            shutil.rmtree(item)
    
    # Remove training set
    for cat in os.listdir(dataset_dir):
        cat_path = dataset_dir / cat
        if cat_path.is_dir() and cat in valid_indicators:
            for train_name in ["Train", "train"]:
                train_dir = cat_path / train_name
                if train_dir.exists():
                    print(f"Removing training set for {DATASET_NAME}/{cat}...")
                    shutil.rmtree(train_dir)

def build_index(base_dir):
    dataset_dir = Path(base_dir) / DATASET_NAME
    if not dataset_dir.exists(): return {}
    categories = [d for d in os.listdir(dataset_dir) if os.path.isdir(dataset_dir / d) and not d.startswith('.')]
    index = {}
    for cat in categories:
        cat_dir = dataset_dir / cat
        cat_samples = []
        for test_name in ["Test", "test"]:
            test_dir = cat_dir / test_name
            if test_dir.exists():
                for label in os.listdir(test_dir):
                    label_dir = test_dir / label
                    if not os.path.isdir(label_dir): continue
                    is_normal = (label.lower() in ["good", "ok"])
                    for f in os.listdir(label_dir):
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            mask_path = None
                            if not is_normal:
                                for gt_sub in ["ground_truth", "gt", "GroundTruth"]:
                                    gt_dir = cat_dir / gt_sub / label
                                    if (gt_dir / f).exists(): mask_path = Path(f"{DATASET_NAME}/{cat}/{gt_sub}/{label}/{f}").as_posix(); break
                                    m_f = f.replace(".png", "_mask.png")
                                    if (gt_dir / m_f).exists(): mask_path = Path(f"{DATASET_NAME}/{cat}/{gt_sub}/{label}/{m_f}").as_posix(); break
                            image_path = Path(f"{DATASET_NAME}/{cat}/test/{label}/{f}").as_posix()
                            cat_samples.append({"image_path": image_path, "mask_path": mask_path, "is_normal": is_normal, "is_train": False})
                break
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
        except FileExistsError: time.sleep(0.1)
    try:
        data = {}
        if data_json_path.exists() and data_json_path.stat().st_size > 0:
            with open(data_json_path, 'r') as f:
                try: data = json.load(f)
                except: data = {}
        data[DATASET_NAME] = new_index
        with open(data_json_path, 'w') as f: json.dump(data, f, indent=2)
    finally:
        if os.path.exists(lock_path): os.remove(lock_path)
    print(f"Updated {DATASET_NAME} in {data_json_path}.")

if __name__ == "__main__":
    root = get_project_root()
    base = sys.argv[1] if len(sys.argv) > 1 else root / "data" / "data"
    json_p = sys.argv[2] if len(sys.argv) > 2 else root / "data" / "data" / "data.json"
    run(base, json_p)
