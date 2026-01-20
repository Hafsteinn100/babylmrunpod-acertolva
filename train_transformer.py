#!/usr/bin/env python3
"""
Train a Tiny Transformer for the Golden Plate NLP challenge ("Hail Mary" attempt).
Features:
- GPT-style Decoder
- Tied weights (Embedding == Output Head)
- Float16 saving
- AdamW + Cosine Scheduler
- Checkpointing
"""
import argparse
import random
import json
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import save_file

# Configuration
VOCAB_SIZE = 256
SEQ_LEN = 512  # INCREASED from 128
BATCH_SIZE = 32 # Decreased safely to avoid memory issues with 4x length

# Architecture (Fits in 1MB with FP16)
# Embed: 256*192 = 49k
# Block: 
#   Attn: 4*192*192 = 147k
#   MLP: 2*192*(192*2) = 147k (Expand 2x)
#   Total per block: ~300k
# 2 Layers = 600k 
# + Embed ~ 650k params * 2 bytes = 1.3 MB (Too big?)
# Let's tune down slightly.
# D_MODEL = 160
# Embed: 40k
# Block: 
#   Attn: 4*160*160 = 102k
#   MLP: 2*160*320 = 102k
#   Total: ~210k
# 2 Layers = 420k + 40k = 460k params.
# 460k * 2 bytes = 0.92 MB. PERFECT.

D_MODEL = 160
N_LAYERS = 2
N_HEADS = 4
D_FF = 320 # 2x expansion
DROPOUT = 0.1

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
            norm_first=True # Pre-Norm is better for stability
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        
        # Tie weights
        self.head.weight = self.tok_emb.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # x: [batch, seq]
        b, t = x.size()
        
        # Positional encoding
        pos = torch.arange(0, t, dtype=torch.long, device=x.device).unsqueeze(0) # [1, t]
        
        x = self.tok_emb(x) + self.pos_emb[:, :t, :]
        
        # Causal Mask
        mask = nn.Transformer.generate_square_subsequent_mask(t).to(x.device)
        
        x = self.blocks(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, None # Dummy hidden state for compatibility

def load_data_stream(data_path, max_docs=None):
    data_path = Path(data_path)
    if data_path.is_dir():
        files = sorted(list(data_path.glob("*.jsonl")))
    else:
        files = [data_path]
        
    random.shuffle(files)
    batch_x, batch_y = [], []
    docs_processed = 0
    
    for file_path in files:
        if max_docs and docs_processed >= max_docs: break
        print(f"Processing {file_path.name}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if max_docs and docs_processed >= max_docs: break
                    try:
                        item = json.loads(line)
                        text_content = item.get("text") or item.get("document")
                        if not text_content: continue
                        
                        text = text_content.encode("utf-8")
                        if len(text) < 2: continue
                        
                        # Chunking with stride
                        stride = SEQ_LEN // 2
                        for i in range(0, len(text) - 1, stride):
                            chunk = text[i : i + SEQ_LEN + 1]
                            if len(chunk) < 2: continue
                            input_seq = list(chunk[:-1])
                            target_seq = list(chunk[1:])
                            
                            # For long context, we accept shorter sequences at the end of docs too, but for simplicity let's require full context for stability
                            if len(input_seq) < SEQ_LEN: continue 
                            
                            batch_x.append(input_seq[:SEQ_LEN])
                            batch_y.append(target_seq[:SEQ_LEN])
                            
                            if len(batch_x) >= BATCH_SIZE:
                                yield torch.tensor(batch_x, dtype=torch.long), torch.tensor(batch_y, dtype=torch.long)
                                batch_x, batch_y = [], []
                        
                        docs_processed += 1
                        if docs_processed % 1000 == 0:
                            print(f"Processed {docs_processed} docs...", end="\r")
                    except: continue
        except: continue

def train(data_path, output_dir, epochs=5, max_docs=100000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = TinyTransformer().to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # We can't know total steps easily with streaming, so we estimate or use plateau scheduler
    # For simplicity in this script, accurate Cosine is hard without knowing steps.
    # We'll use StepLR as a robust fallback.
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)
    
    step = 0
    best_loss = float('inf')
    
    try:
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            batch_iter = load_data_stream(data_path, max_docs=max_docs)
            
            for x, y in batch_iter:
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                logits, _ = model(x)
                loss = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                step += 1
                if step % 50 == 0:
                    lr = optimizer.param_groups[0]['lr']
                    print(f"Step {step}, Loss: {loss.item():.4f}, LR: {lr:.5f}", end="\r")
                
                # Checkpoint every 5000 steps
                if step % 5000 == 0:
                    # Save as float16
                    ckpt_path = output_dir / "model.safetensors"
                    # Convert to half for saving
                    state_dict = {k: v.half() for k, v in model.state_dict().items()}
                    save_file(state_dict, ckpt_path)
                    print(f"\nSaved checkpoint to {ckpt_path}")
            
            print(f"\nEpoch {epoch+1} complete.")
            
    except KeyboardInterrupt:
        print("\nManually interrupted. Saving current state...")
        
    # Final Save
    final_path = output_dir / "model.safetensors"
    state_dict = {k: v.half() for k, v in model.state_dict().items()}
    save_file(state_dict, final_path)
    
    size_mb = final_path.stat().st_size / (1024*1024)
    print(f"Final model saved to {final_path} ({size_mb:.3f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default="submission")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-docs", type=int, default=100000)
    args = parser.parse_args()
    
    train(args.data, args.output_dir, epochs=args.epochs, max_docs=args.max_docs)
