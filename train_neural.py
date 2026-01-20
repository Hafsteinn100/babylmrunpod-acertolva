#!/usr/bin/env python3
"""
Train a byte-level LSTM model for the Golden Plate NLP challenge.
"""
import argparse
import random
import sys
import json
import glob
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import save_file

# Constants - sized to fit in 1MB with Int8 quantization
EMBED_DIM = 64
HIDDEN_DIM = 180  # Proven size (0.91MB)
VOCAB_SIZE = 256
SEQ_LEN = 128
BATCH_SIZE = 64

# ... (rest of imports/classes)

def train(data_path, output_path, epochs=1, max_docs=20000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ByteLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
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
                
    print()
    print("Training Complete.")
    
    # Quantize and Save
    print("Quantizing model to Int8...")
    model.cpu()
    model.eval()
    
    # Set engine for quantization
    torch.backends.quantized.engine = 'qnnpack'
    
    # Dynamic quantization
    model_int8 = torch.quantization.quantize_dynamic(
        model, 
        {nn.LSTM, nn.Linear},
        dtype=torch.qint8
    )
    
    state_dict = model_int8.state_dict()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use safetensors for compact storage
    save_file(state_dict, output_path)
    
    size_mb = output_path.stat().st_size / (1024*1024)
    print(f"Saved quantized model to {output_path} ({size_mb:.3f} MB)")
    
    if size_mb > 1.0:
        print("WARNING: Model exceeds 1MB limit!")


class ByteLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        # Using 1 layer to keep size small. Could try 2 if size permits.
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, num_layers=1, batch_first=True)
        self.head = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        
    def forward(self, x, hidden=None):
        # x: [batch, seq_len]
        emb = self.embed(x)  # [batch, seq_len, embed_dim]
        out, hidden = self.lstm(emb, hidden)  # [batch, seq_len, hidden_dim]
        logits = self.head(out)  # [batch, seq_len, vocab_size]
        return logits, hidden

def load_data_stream(data_path, max_docs=None):
    """Yield batches of (input, target) tensors from JSONL files."""
    data_path = Path(data_path)
    if data_path.is_dir():
        files = sorted(list(data_path.glob("*.jsonl")))
    else:
        files = [data_path]
        
    print(f"Found {len(files)} files.")
    
    # We want to stream through files without loading everything
    # But we also want some randomness if possible. 
    # For a simple implementation, we'll shuffle the file order,
    # and then shuffle chunks within a buffer.
    
    random.shuffle(files)
    
    batch_x = []
    batch_y = []
    
    docs_processed = 0
    
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
                        if "text" in item:
                            text_content = item["text"]
                        elif "document" in item:
                            text_content = item["document"]
                        else:
                            continue
                        
                        text = text_content.encode("utf-8") # Byte level
                        if len(text) < 2:
                            continue
                            
                        # Chunking
                        stride = SEQ_LEN // 2
                        for i in range(0, len(text) - 1, stride):
                            chunk = text[i : i + SEQ_LEN + 1]
                            if len(chunk) < 2: 
                                continue
                                
                            input_seq = list(chunk[:-1])
                            target_seq = list(chunk[1:])
                            
                            # Skip if too short (padding complex in simple stream)
                            if len(input_seq) < SEQ_LEN:
                                continue
                                
                            batch_x.append(input_seq[:SEQ_LEN])
                            batch_y.append(target_seq[:SEQ_LEN])
                            
                            if len(batch_x) >= BATCH_SIZE:
                                yield torch.tensor(batch_x, dtype=torch.long), torch.tensor(batch_y, dtype=torch.long)
                                batch_x = []
                                batch_y = []
                        
                        docs_processed += 1
                        if docs_processed % 1000 == 0:
                            print(f"Processed {docs_processed} docs...", end="\r")
                            
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

def train(data_path, output_path, epochs=1, max_docs=20000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ByteLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        batch_iter = load_data_stream(data_path, max_docs=max_docs)
        
        for x, y in batch_iter:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(x)
            
            # Flatten for loss
            # logits: [batch, seq, vocab] -> [batch*seq, vocab]
            # y: [batch, seq] -> [batch*seq]
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            step += 1
            if step % 50 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}", end="\r")
            
            # Checkpointing
            if step % 5000 == 0:
                ckpt_path = output_path.with_name("model_checkpoint.safetensors")
                save_file(model.state_dict(), ckpt_path)

    print()
    print("Training Complete.")
    
    # Save weights - keep float32 for CPU compatibility
    # FP16 causes issues on some CPU implementations
    model = model.cpu()
    state_dict = model.state_dict()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use safetensors for compact storage
    save_file(state_dict, output_path)
    
    size_mb = output_path.stat().st_size / (1024*1024)
    print(f"Saved to {output_path} ({size_mb:.3f} MB)")
    
    if size_mb > 1.0:
        print("WARNING: Model exceeds 1MB limit! Consider reducing architecture size.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default="submission/model.safetensors")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-docs", type=int, default=20000)
    args = parser.parse_args()
    
    try:
        train(args.data, args.output, epochs=args.epochs, max_docs=args.max_docs)
    except KeyboardInterrupt:
        print("\nTraining interrupted! Model state is preserved in the last checkpoint (if any).")
        # In a real scenario we'd pass model out to save here, but for now relying on checkpoint is safer than saving broken state.
        # Actually, let's just let the user know.
