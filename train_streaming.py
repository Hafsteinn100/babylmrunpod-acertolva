#!/usr/bin/env python3
"""
Train a LARGER byte-level LSTM model, then quantize to INT8.
This version STREAMS from HuggingFace - no disk save required!
"""
import argparse
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

# Configuration - LARGER model for INT8 quantization
EMBED_DIM = 64
HIDDEN_DIM = 440  # Reduced from 480 to fit 1MB limit
VOCAB_SIZE = 256
SEQ_LEN = 128
BATCH_SIZE = 64

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

def load_data_stream_hf(max_docs=None):
    """Stream data directly from HuggingFace - no disk save needed!"""
    from datasets import load_dataset
    
    print("Streaming from HuggingFace (no disk save)...")
    ds = load_dataset("arnastofnun/IGC-2024", split="train", streaming=True)
    
    batch_x, batch_y = [], []
    docs_processed = 0
    stride = SEQ_LEN // 2
    
    for item in ds:
        if max_docs and docs_processed >= max_docs:
            break
            
        text_content = item.get("text") or item.get("document")
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

def train(output_dir, epochs=1, max_docs=100000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = ByteLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Estimated INT8 size: {param_count * 1 / 1024 / 1024:.2f} MB (approx)")
    
    step = 0
    try:
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            batch_iter = load_data_stream_hf(max_docs=max_docs)
            
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
                
                # Checkpoint every 10000 steps
                if step % 10000 == 0:
                    ckpt_path = output_dir / "checkpoint_fp32.pt"
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"\nCheckpoint saved at step {step}")
                    
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
    
    print("\n\nTraining Complete.")
    print("Quantizing model to INT8...")
    
    # Move to CPU for quantization
    model.cpu()
    model.eval()
    
    # Dynamic quantization
    torch.backends.quantized.engine = 'qnnpack'
    model_int8 = torch.quantization.quantize_dynamic(
        model,
        {nn.LSTM, nn.Linear},
        dtype=torch.qint8
    )
    
    # Save quantized model
    int8_path = output_dir / "model_int8.pt"
    torch.save(model_int8.state_dict(), int8_path)
    int8_size = int8_path.stat().st_size / 1024 / 1024
    print(f"INT8 model saved: {int8_size:.2f} MB")
    
    if int8_size > 1.0:
        print("WARNING: INT8 model exceeds 1MB!")
    else:
        print("SUCCESS: INT8 model fits in 1MB limit!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default="submission")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-docs", type=int, default=100000)
    args = parser.parse_args()
    
    train(args.output_dir, epochs=args.epochs, max_docs=args.max_docs)
