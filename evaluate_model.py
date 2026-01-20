#!/usr/bin/env python3
"""
Evaluate the model on held-out data to estimate the Bits-Per-Byte (BPB) score.

Usage:
    python evaluate_model.py --data data/igc_full --submission submission
"""

import argparse
import math
import sys
import time
import random
from pathlib import Path
import importlib.util

# Set random seed for reproducibility
random.seed(42)

def softmax(logits):
    """Compute softmax over a list of numbers."""
    # Stability fix: subtract max
    max_logit = max(logits)
    exp_logits = [math.exp(x - max_logit) for x in logits]
    sum_exp = sum(exp_logits)
    return [x / sum_exp for x in exp_logits]

def load_model(submission_dir):
    """Load the Model class from submission/model.py"""
    model_path = submission_dir / "model.py"
    if not model_path.exists():
        print(f"Error: {model_path} not found.")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("submission_model", model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["submission_model"] = module
    spec.loader.exec_module(module)
    
    return module.Model(submission_dir)

import json
import glob

def load_data(data_path, start_idx=50000, num_docs=100):
    """Load a subset of the dataset from JSONL files."""
    data_path = Path(data_path)
    if data_path.is_dir():
        files = sorted(list(data_path.glob("*.jsonl")))
    else:
        files = [data_path]

    print(f"Loading data from {len(files)} files...")
    texts = []
    
    # Simple iterator over files to pick docs in range
    current_idx = 0
    end_idx = start_idx + num_docs
    
    for file_path in files:
        if current_idx >= end_idx:
            break
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if current_idx >= end_idx:
                        break
                        
                    if current_idx >= start_idx:
                        try:
                            item = json.loads(line)
                            start_byte_len = len(texts)
                            
                            if "text" in item:
                                texts.append(item["text"].encode("utf-8"))
                            elif "document" in item:
                                texts.append(item["document"].encode("utf-8"))
                                
                            # If we added a text, increment
                            if len(texts) > start_byte_len:
                                pass # successfully added
                            else:
                                pass # skipped
                                
                        except Exception as e:
                            print(f"JSON decode error at line {current_idx}: {e}")
                            pass
                            
                    current_idx += 1
                    if current_idx % 1000 == 0:
                        print(f"Scanned {current_idx} docs...", end="\r")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    print(f"Loaded {len(texts)} documents.")
    return texts

def evaluate(model, texts, batch_size=1024):
    """Calculate average Bits-Per-Byte (BPB)."""
    total_log_prob = 0.0
    total_bytes = 0
    
    # We will process byte-by-byte for each text
    # To use the batching capabilities of the model, we collect contexts into batches
    
    contexts = []
    targets = []
    
    # Pre-generate all contexts (memory intensive but simple for now)
    # Optimally we would stream this
    print("Preparing contexts...")
    for text in texts:
        # Evaluate on the first 1024 bytes of each document to save time
        # or evaluate full documents. Let's do partial documents for speed.
        limit = min(len(text), 2048) 
        
        for i in range(limit):
            # Context is previous 512 bytes
            start = max(0, i - 512)
            context = list(text[start:i])
            target_byte = text[i]
            
            contexts.append(context)
            targets.append(target_byte)
            
    num_samples = len(contexts)
    print(f"Evaluating on {num_samples:,} bytes...")
    
    start_time = time.time()
    
    for i in range(0, num_samples, batch_size):
        batch_contexts = contexts[i : i + batch_size]
        batch_targets = targets[i : i + batch_size]
        
        # Get logits
        batch_logits = model.predict(batch_contexts)
        
        for logits, target in zip(batch_logits, batch_targets):
            probs = softmax(logits)
            prob = probs[target]
            
            # Avoid log(0)
            if prob < 1e-10:
                prob = 1e-10
                
            total_log_prob -= math.log2(prob)
            total_bytes += 1
            
        if (i + batch_size) % 10000 < batch_size:
            print(f"Processed {i + len(batch_contexts)} / {num_samples} bytes...", end="\r")
            
    print()
    elapsed = time.time() - start_time
    bpb = total_log_prob / total_bytes
    
    print("-" * 40)
    print(f"Total Bytes: {total_bytes:,}")
    print(f"Time: {elapsed:.2f}s ({total_bytes/elapsed:.0f} bytes/s)")
    print(f"Score: {bpb:.4f} BPB")
    print("-" * 40)
    
    return bpb

def main():
    parser = argparse.ArgumentParser(description="Evaluate model BPB")
    parser.add_argument("--data", type=Path, required=True, help="Path to evaluation data")
    parser.add_argument("--submission", type=Path, default=Path("submission"), help="Path to submission directory")
    parser.add_argument("--start-idx", type=int, default=100000, help="Start index in dataset (default: 100000 to skip training set)")
    parser.add_argument("--num-docs", type=int, default=50, help="Number of documents to evaluate")
    args = parser.parse_args()
    
    print("Loading model...")
    model = load_model(args.submission)
    
    print("Loading data...")
    texts = load_data(args.data, start_idx=args.start_idx, num_docs=args.num_docs)
    
    if not texts:
        print("No texts found!")
        return
        
    evaluate(model, texts)

if __name__ == "__main__":
    main()
