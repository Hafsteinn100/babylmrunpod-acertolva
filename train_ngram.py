#!/usr/bin/env python3
"""
Train an n-gram model for next-byte prediction.

This creates a simple but effective baseline model that:
1. Counts byte sequences in training data
2. Saves the counts as a compressed JSON file
3. At inference, predicts based on observed frequencies

Usage:
    # From HuggingFace dataset (arrow format):
    python train_ngram.py --data /path/to/igc_full --n 3

    # From text files:
    python train_ngram.py --data /path/to/texts --n 3 --text-mode

The output goes to submission/counts.json.gz
"""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def load_from_hf_dataset(data_path: Path, max_docs: int = None):
    """Yield training data from JSONL files."""
    import json as json_mod
    
    data_path = Path(data_path)
    if data_path.is_dir():
        files = sorted(list(data_path.glob("*.jsonl")))
    else:
        files = [data_path]
        
    print(f"Found {len(files)} JSONL files...")
    
    docs_processed = 0
    
    for file_path in files:
        if max_docs and docs_processed >= max_docs:
            break
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if max_docs and docs_processed >= max_docs:
                        break
                        
                    try:
                        item = json_mod.loads(line)
                        text = None
                        if "text" in item:
                            text = item["text"]
                        elif "document" in item:
                            text = item["document"]
                            
                        if text:
                            yield text.encode("utf-8")
                            docs_processed += 1
                            
                            if docs_processed % 10000 == 0:
                                print(f"Processed {docs_processed:,} docs...", end="\r")
                    except:
                        continue
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    print(f"\nTotal documents processed: {docs_processed:,}")


def load_from_text_files(data_path: Path):
    """Yield training data from text files."""
    if data_path.is_file():
        yield data_path.read_bytes()
    elif data_path.is_dir():
        for f in data_path.glob("**/*.txt"):
            yield f.read_bytes()
    else:
        raise ValueError(f"Data path not found: {data_path}")


def train_ngram(texts_loader, n: int, min_count: int = 2) -> dict:
    """
    Train n-gram counts from training data stream.
    
    Args:
        texts_loader: Generator yielding byte sequences
        n: N-gram order
        min_count: Minimum count to keep
    """
    # Count n-grams: context (n-1 bytes) -> next byte -> count
    counts = defaultdict(lambda: defaultdict(int))
    
    total_ngrams = 0
    doc_count = 0
    
    print(f"Training...")
    
    for text in texts_loader:
        doc_count += 1
        for i in range(len(text)):
            # Get context (previous n-1 bytes, or less at start)
            start = max(0, i - n + 1)
            context = tuple(text[start:i])
            next_byte = text[i]
            counts[context][next_byte] += 1
            total_ngrams += 1
            
        if doc_count % 10000 == 0:
            print(f"Docs: {doc_count:,} | N-grams: {total_ngrams:,} | Contexts: {len(counts):,}", end="\r")

    print()
    print(f"Counted {total_ngrams:,} n-grams")
    print(f"Unique contexts: {len(counts):,}")

    # Prune rare n-grams to save space
    print(f"Pruning (min_count={min_count})...")
    pruned_counts = {}
    for context, byte_counts in counts.items():
        total = sum(byte_counts.values())
        if total >= min_count:
            # Convert to list format: [[byte, count], ...]
            # Only keep bytes that also meet the min_count threshold
            filtered_bytes = [
                [b, c] for b, c in byte_counts.items() if c >= min_count
            ]
            if filtered_bytes:
                pruned_counts[str(list(context))] = filtered_bytes
            
    print(f"After pruning: {len(pruned_counts):,} contexts")
    return pruned_counts


def save_counts(counts: dict, output_path: Path):
    """Save counts to gzipped JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure .gz extension
    if not str(output_path).endswith('.gz'):
        output_path = output_path.with_suffix('.json.gz')

    json_str = json.dumps(counts, separators=(',', ':'))

    with gzip.open(output_path, 'wt') as f:
        f.write(json_str)

    size_kb = output_path.stat().st_size / 1024
    print(f"Saved to {output_path} ({size_kb:.1f} KB)")

    if size_kb > 900:
        print("\nWARNING: File is close to 1 MB limit!")
        print("Try increasing --min-count or decreasing --n")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train n-gram model for byte prediction")
    parser.add_argument("--data", type=Path, required=True,
                        help="Path to training data (HuggingFace dataset dir or text files)")
    parser.add_argument("--n", type=int, default=5,
                        help="N-gram order (default: 5)")
    parser.add_argument("--min-count", type=int, default=5,
                        help="Minimum count to keep (default: 5)")
    parser.add_argument("--output", type=Path, default=Path("submission/counts.json.gz"),
                        help="Output file path")
    parser.add_argument("--text-mode", action="store_true",
                        help="Load from .txt files instead of HuggingFace dataset")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Maximum documents to load (default: all)")
    args = parser.parse_args()

    print(f"Training {args.n}-gram model")
    print(f"Data: {args.data}")
    print(f"Min count: {args.min_count}")
    print()

    # Load data generator
    if args.text_mode:
        texts_loader = load_from_text_files(args.data)
    else:
        texts_loader = load_from_hf_dataset(args.data, max_docs=args.max_docs)

    print()

    # Train
    counts = train_ngram(texts_loader, args.n, args.min_count)

    # Save
    output_path = save_counts(counts, args.output)

    print()
    print("Done! Now run:")
    print("  python create_submission.py")
    print()
    print("Then upload submission.zip to the competition website.")


if __name__ == "__main__":
    main()
