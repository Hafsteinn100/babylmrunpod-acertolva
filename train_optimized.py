#!/usr/bin/env python3
"""
Train a 4-Layer Transformer on the FULL IGC Dataset with optimizations for 1 BPB.
"""
import argparse
import random
import json
import math
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

# --- Configuration (Must Match submission/model.py) ---
VOCAB_SIZE = 256
D_MODEL = 160
N_LAYERS = 4      
N_HEADS = 4
D_FF = 320
DROPOUT = 0.1
SEQ_LEN = 512
BATCH_SIZE = 128

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Parameter(torch.zeros(1, SEQ_LEN, D_MODEL))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_FF,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        b, t = x.size()
        pos = torch.arange(0, t, dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.tok_emb(x) + self.pos_emb[:, :t, :]
        mask = nn.Transformer.generate_square_subsequent_mask(t).to(x.device)
        x = self.blocks(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, None

def load_data_stream(data_path, max_docs=None, validation=False):
    """Stream data from local JSONL files."""
    data_path = Path(data_path)
    if not data_path.exists():
        print(f"Error: {data_path} does not exist.")
        return
        
    files = sorted(list(data_path.rglob("*.jsonl")))
    
    # Simple validation split: last 5% of files
    val_split_idx = int(len(files) * 0.95)
    if validation:
        files = files[val_split_idx:]
        print(f"Validation mode: {len(files)} files.")
    else:
        files = files[:val_split_idx]
        print(f"Training mode: {len(files)} files.")
        random.shuffle(files)
    
    batch_x, batch_y = [], []
    docs_processed = 0
    stride = SEQ_LEN // 2
    
    for file_path in files:
        if max_docs and docs_processed >= max_docs: break
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if max_docs and docs_processed >= max_docs: break
                    try:
                        item = json.loads(line)
                        text_content = item.get("document") or item.get("text")
                        if not text_content: continue
                        text = text_content.encode("utf-8")
                        if len(text) < 2: continue
                        
                        for i in range(0, len(text) - 1, stride):
                            chunk = text[i : i + SEQ_LEN + 1]
                            if len(chunk) < SEQ_LEN + 1: continue
                            
                            batch_x.append(list(chunk[:-1]))
                            batch_y.append(list(chunk[1:]))
                            
                            if len(batch_x) >= BATCH_SIZE:
                                yield torch.tensor(batch_x, dtype=torch.long), torch.tensor(batch_y, dtype=torch.long)
                                batch_x, batch_y = [], []
                        
                        docs_processed += 1
                        if not validation and docs_processed % 5000 == 0:
                            print(f"Processed {docs_processed} docs...", end="\r")
                    except: continue
        except: continue

def validate(model, data_path, device, max_docs=1000):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_batches = 0
    
    with torch.no_grad():
        for x, y in load_data_stream(data_path, max_docs=max_docs, validation=True):
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            total_loss += loss.item()
            total_batches += 1
            if total_batches >= 100: break # Limit validation time
            
    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    bpb = avg_loss / 0.693147
    return bpb, avg_loss

def train(data_path, output_dir, epochs=5, max_docs=None):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS acceleration!")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    model = TinyTransformer().to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Cosine Annealing Scheduler
    # Assuming approx 5000 steps per epoch for partial data, but for full data it's much more.
    # We'll set T_max to a large number or estimate based on data size if possible, 
    # but for simplicity we'll just set it to a fixed large number of steps per epoch * epochs.
    # Let's estimate: 2M docs / 32 batch size / ~10 chunks per doc = very rough.
    # Safe bet: T_max = 50000 * epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50000*epochs, eta_min=1e-5)
    
    step = 0
    best_bpb = float('inf')
    
    try:
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            model.train()
            batch_iter = load_data_stream(data_path, max_docs=max_docs, validation=False)
            
            for x, y in batch_iter:
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                logits, _ = model(x)
                loss = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                step += 1
                if step % 50 == 0:
                    bpb = loss.item() / 0.693147
                    lr = optimizer.param_groups[0]['lr']
                    print(f"Step {step} | BPB: {bpb:.4f} | Loss: {loss.item():.4f} | LR: {lr:.5f}", end="\r")
                
                if step % 5000 == 0:
                    save_checkpoint(model, output_dir)
                    # Run quick validation
                    val_bpb, val_loss = validate(model, data_path, device)
                    print(f"\n[Validation] Step {step} | Val BPB: {val_bpb:.4f} | Val Loss: {val_loss:.4f}")
                    if val_bpb < best_bpb:
                        best_bpb = val_bpb
                        save_best_model(model, output_dir)
                        
    except KeyboardInterrupt:
        print("\nInterrupted. Saving model...")
        
    save_quantized(model, output_dir)

def save_checkpoint(model, output_dir):
    ckpt_path = Path(output_dir) / "checkpoint_fp32.pt"
    torch.save(model.state_dict(), ckpt_path)

def save_best_model(model, output_dir):
    ckpt_path = Path(output_dir) / "best_model_fp32.pt"
    torch.save(model.state_dict(), ckpt_path)
    print("Saved best model.")
    
def save_quantized(model, output_dir):
    print("\nQuantizing to INT8...")
    model.cpu().eval()
    torch.backends.quantized.engine = 'fbgemm'
    model_int8 = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    path = Path(output_dir) / "model_int8.pt"
    torch.save(model_int8.state_dict(), path)
    size_mb = path.stat().st_size / (1024*1024)
    print(f"Saved INT8 model: {size_mb:.2f} MB")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default="data/igc_full")
    parser.add_argument("--output_dir", type=Path, default="submission")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    
    # Override global batch size
    global BATCH_SIZE
    BATCH_SIZE = args.batch_size
    
    train(args.data, args.output_dir, epochs=args.epochs, max_docs=args.max_docs)
