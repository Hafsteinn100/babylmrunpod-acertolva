#!/usr/bin/env python3
"""
Prune an existing counts.json.xz model to reduce size.
"""
import lzma
import json
import argparse
from pathlib import Path

def prune_model(min_count):
    input_path = Path("submission/counts.json.xz")
    output_path = Path("submission/counts.json.xz") # Overwrite
    
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return

    print(f"Reading {input_path}...")
    with lzma.open(input_path, 'rt') as f:
        counts = json.load(f)
        
    print(f"Original contexts: {len(counts)}")
    
    pruned_counts = {}
    total_ngrams = 0
    removed_contexts = 0
    
    for context, byte_counts in counts.items():
        # byte_counts is list of [byte, count]
        # Calculate total count for this context
        total = sum(c for b, c in byte_counts)
        
        # Prune if total count for context is small (unreliable context)
        # OR prune individual bytes if they are small (handled by train_ngram already but we can raise bar)
        
        # Let's simple apply higher min_count
        new_byte_counts = [[b, c] for b, c in byte_counts if c >= min_count]
        
        if new_byte_counts:
            pruned_counts[context] = new_byte_counts
        else:
            removed_contexts += 1

    print(f"Pruned contexts: {len(pruned_counts)}")
    print(f"Removed: {removed_contexts}")
    
    print(f"Writing {output_path}...")
    with lzma.open(output_path, 'wt', preset=9) as f:
        json.dump(pruned_counts, f, separators=(',', ':'))
        
    size_xz = output_path.stat().st_size
    print(f"New size: {size_xz/1024:.1f} KB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-count", type=int, default=60)
    args = parser.parse_args()
    prune_model(args.min_count)
