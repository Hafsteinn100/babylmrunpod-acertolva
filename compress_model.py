#!/usr/bin/env python3
"""
Convert counts.json.gz to counts.json.xz for better compression.
"""
import gzip
import lzma
import json
import shutil
from pathlib import Path
import os

def recompress():
    input_path = Path("submission/counts.json.gz")
    output_path = Path("submission/counts.json.xz")
    
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return

    print(f"Reading {input_path}...")
    with gzip.open(input_path, 'rt') as f_in:
        data = f_in.read()
        
    print(f"Writing {output_path} (LZMA)...")
    with lzma.open(output_path, 'wt', preset=9) as f_out:
        f_out.write(data)
        
    size_gz = input_path.stat().st_size
    size_xz = output_path.stat().st_size
    
    print(f"GZIP size: {size_gz/1024:.1f} KB")
    print(f"LZMA size: {size_xz/1024:.1f} KB")
    print(f"Reduction: {100 * (size_gz - size_xz) / size_gz:.1f}%")
    
    # Remove old file to avoid including both everyone
    print("Removing old file...")
    input_path.unlink()

if __name__ == "__main__":
    recompress()
