import sys
import os
import concurrent.futures
from pathlib import Path

current_dir = Path(__file__).resolve().parent
scripts_dir = current_dir / "scripts"
sys.path.append(str(scripts_dir))

from scripts import mvtec, visa, btad, dagm, mpdd, dtd
dataset_modules = [mvtec, visa, btad, dagm, mpdd, dtd]

def run_module(module):
    project_root = current_dir.parent
    base_dir = project_root / "data" / "data"
    data_json_path = project_root / "data" / "data" / "data.json"
    
    try:
        module.run(str(base_dir), str(data_json_path))
        print(f"[+] {module.DATASET_NAME.upper()} preparation completed.")
    except Exception as e:        
        print(f"[!] Error preparing {module.DATASET_NAME.upper()}: {e}")

def main(selected: list[str] | None = None):
    requested = [m for m in dataset_modules
                 if selected is None or m.DATASET_NAME in selected]
    if not requested:
        print(f"[!] No matching datasets found for: {selected}")
        return
    print(f"[-] Starting Prep for: {[m.DATASET_NAME for m in requested]}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(requested)) as executor:
        executor.map(run_module, requested)
    print("[-] All Prep Finished.")

if __name__ == "__main__":
    names = sys.argv[1:] if len(sys.argv) > 1 else None
    main(names)
