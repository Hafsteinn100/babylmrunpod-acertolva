#!/usr/bin/env python3
"""
Train on a SMALL subset of IGC data - fits in 20GB disk.
Downloads only 3 ZIP files (~3GB) instead of all 15GB.
"""
import argparse
import os
import json
import zipfile
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

# Configuration
EMBED_DIM = 64
HIDDEN_DIM = 400  # Fits under 1MB when quantized
VOCAB_SIZE = 256
SEQ_LEN = 128
BATCH_SIZE = 64

# Only download these files (small subset, ~3GB total)
SUBSET_FILES = [
    "igc_news1.zip",
    "igc_wiki.zip",
    "igc_journals.zip",
]

class ByteLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, num_layers=1, batch_first=True)
        self.head = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        
    def forward(self, x, hidden=None):
        emb = self.embed(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.head(out)
        return logits, hidden

def download_subset():
    """Download only a subset of IGC data."""
    from huggingface_hub import hf_hub_download
    
    if os.path.exists("IGC-data") and len(list(Path("IGC-data").glob("*.jsonl"))) > 0:
        print("Data already exists, skipping download...")
        return "IGC-data"
    
    os.makedirs("IGC-data", exist_ok=True)
    
    for filename in SUBSET_FILES:
        print(f"Downloading {filename}...")
        try:
            local_path = hf_hub_download(
                repo_id="arnastofnun/IGC-2024",
                repo_type="dataset",
                filename=filename,
            )
            
            print(f"  Extracting {filename}...")
            with zipfile.ZipFile(local_path, 'r') as z:
                z.extractall("IGC-data")
            
            # Delete the zip after extraction to save space
            os.remove(local_path)
            print(f"  Done with {filename}")
        except Exception as e:
            print(f"  Skipping {filename}: {e}")
    
    return "IGC-data"

def load_data_stream(data_path, max_docs=None):
    """Stream data from local JSONL files."""
    data_path = Path(data_path)
    files = sorted(list(data_path.glob("*.jsonl")))
    
    print(f"Found {len(files)} JSONL files.")
    random.shuffle(files)
    
    batch_x, batch_y = [], []
    docs_processed = 0
    stride = SEQ_LEN // 2
    
    for file_path in files:
        if max_docs and docs_processed >= max_docs:
            break
        print(f"Processing {file_path.name}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if max_docs and docs_processed >= max_docs:
                        break
                    try:
                        item = json.loads(line)
                        text_content = item.get("document") or item.get("text")
                        if not text_content:
                            continue
                        
                        text = text_content.encode("utf-8")
                        if len(text) < 2:
                            continue
                        
                        for i in range(0, len(text) - 1, stride):
                            chunk = text[i : i + SEQ_LEN + 1]
                            if len(chunk) < 2:
                                continue
                            input_seq = list(chunk[:-1])
                            target_seq = list(chunk[1:])
                            
                            if len(input_seq) < SEQ_LEN:
                                continue
                            
                            batch_x.append(input_seq[:SEQ_LEN])
                            batch_y.append(target_seq[:SEQ_LEN])
                            
                            if len(batch_x) >= BATCH_SIZE:
                                yield torch.tensor(batch_x, dtype=torch.long), torch.tensor(batch_y, dtype=torch.long)
                                batch_x, batch_y = [], []
                        
                        docs_processed += 1
                        if docs_processed % 1000 == 0:
                            print(f"Processed {docs_processed} docs...", end="\r")
                    except:
                        continue
        except:
            continue

def train(output_dir, epochs=1, max_docs=100000):
    # Download subset first
    data_path = download_subset()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = ByteLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    step = 0
    try:
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            batch_iter = load_data_stream(data_path, max_docs=max_docs)
            
            for x, y in batch_iter:
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                logits, _ = model(x)
                loss = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
                loss.backward()
                optimizer.step()
                
                step += 1
                if step % 50 == 0:
                    print(f"Step {step}, Loss: {loss.item():.4f}", end="\r")
                
                if step % 10000 == 0:
                    ckpt_path = output_dir / "checkpoint_fp32.pt"
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"\nCheckpoint saved at step {step}")
                    
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
    
    print("\n\nTraining Complete.")
    print("Quantizing model to INT8...")
    
    model.cpu()
    model.eval()
    
    torch.backends.quantized.engine = 'qnnpack'
    model_int8 = torch.quantization.quantize_dynamic(
        model,
        {nn.LSTM, nn.Linear},
        dtype=torch.qint8
    )
    
    int8_path = output_dir / "model_int8.pt"
    torch.save(model_int8.state_dict(), int8_path)
    int8_size = int8_path.stat().st_size / 1024 / 1024
    print(f"INT8 model saved: {int8_size:.2f} MB")
    
    if int8_size > 1.0:
        print("WARNING: Model exceeds 1MB!")
    else:
        print("SUCCESS: Model fits in 1MB limit!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default="submission")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-docs", type=int, default=100000)
    args = parser.parse_args()
    
    train(args.output_dir, epochs=args.epochs, max_docs=args.max_docs)
