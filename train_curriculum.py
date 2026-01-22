#!/usr/bin/env python3
"""
Curriculum Training Script for Golden Plate NLP Challenge (RunPod Version).

STRATEGY:
- Phase 1 (80%): Train on ALL data (Learn everything).
- Phase 2 (20%): Train on HIGH QUALITY data only (Unlearn noise).

High Quality = Wiki, News, Law, Journals.
Low Quality = Social (Bland, Hugi, etc.)
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
BATCH_SIZE = 32

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

def get_files(data_path, phase="all"):
    """Filter files based on curriculum phase."""
    all_files = sorted(list(Path(data_path).rglob("*.jsonl")))
    
    if phase == "hq":
        # High Quality: Wiki, News, Law, Journals, Adjud
        hq_files = [f for f in all_files if any(x in f.name for x in [
            "igc_wiki", "igc_news", "igc_law", "igc_journals", "igc_adjud"
        ])]
        print(f"[{phase.upper()}] Selected {len(hq_files)} High-Quality files.")
        return hq_files
    else:
        # All files
        print(f"[{phase.upper()}] Selected {len(all_files)} files (Full Dataset).")
        return all_files

def load_data_stream(files):
    """Stream data from selected files."""
    random.shuffle(files)
    
    batch_x, batch_y = [], []
    stride = SEQ_LEN // 2
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
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
                    except: continue
        except: continue

def train_curriculum(data_path, output_dir, total_epochs=5):
    # Setup Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    model = TinyTransformer().to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.85)
    
    step = 0
    
    try:
        for epoch in range(total_epochs):
            # CURRICULUM LOGIC
            # If last epoch, switch to High Quality Only
            is_final_phase = (epoch == total_epochs - 1)
            phase_name = "hq" if is_final_phase else "all"
            
            print(f"\n=== Epoch {epoch+1}/{total_epochs} (Phase: {phase_name.upper()}) ===")
            
            files = get_files(data_path, phase_name)
            batch_iter = load_data_stream(files)
            
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
                    
    except KeyboardInterrupt:
        print("\nInterrupted. Saving model...")
        
    save_quantized(model, output_dir)

def save_checkpoint(model, output_dir):
    ckpt_path = Path(output_dir) / "checkpoint_fp32.pt"
    torch.save(model.state_dict(), ckpt_path)
    
def save_quantized(model, output_dir):
    print("\nQuantizing to INT8...")
    model.cpu().eval()
    torch.backends.quantized.engine = 'qnnpack'
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
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    
    train_curriculum(args.data, args.output_dir, total_epochs=args.epochs)
