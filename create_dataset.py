#!/usr/bin/env python3
"""
Create the training dataset for the NLP Challenge (High-Storage Version).

Downloads the IGC-2024 (Icelandic Gigaword Corpus) from HuggingFace
and EXTRACTS the JSONL files to `data/igc_full`.
"""

import os
import zipfile
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

def prepare_data():
    """Download and extract IGC-2024 to data/igc_full as JSONL files."""
    
    # 1. Download
    print("Downloading IGC-2024 from HuggingFace...")
    snapshot_dir = "IGC-2024-snapshot"
    snapshot_download(
        repo_id="arnastofnun/IGC-2024",
        repo_type="dataset",
        local_dir=snapshot_dir,
        allow_patterns=["*.zip"]
    )

    # 2. Extract
    target_dir = Path("data/igc_full")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting files to {target_dir}...")
    
    for root, dirs, files in os.walk(snapshot_dir):
        for file in files:
            if file.endswith(".zip"):
                zip_path = os.path.join(root, file)
                print(f"  Extracting {file}...")
                try:
                    with zipfile.ZipFile(zip_path, "r") as z:
                        z.extractall(target_dir)
                except Exception as e:
                    print(f"  Error extracting {file}: {e}")

    # 3. Cleanup Snapshot (Optional, saves space if volume is tight, but 170GB is plenty)
    # shutil.rmtree(snapshot_dir)

    print("\n" + "="*60)
    print(f"Done! JSONL files are ready in: {target_dir}")
    print(f"Found {len(list(target_dir.rglob('*.jsonl')))} JSONL files.")
    print("="*60)

if __name__ == "__main__":
    prepare_data()
