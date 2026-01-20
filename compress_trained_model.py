#!/usr/bin/env python3
"""
Load float32 weights from safetensors, quantize to int8, save as torch pickle.
"""
import torch
import torch.nn as nn
from pathlib import Path
from safetensors.torch import load_file
import os

# Constants
EMBED_DIM = 64
HIDDEN_DIM = 425 
VOCAB_SIZE = 256

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

def compress():
    input_path = "submission/model.pt"  # Actually safetensors format
    output_path = "submission/model_quantized.pt"
    
    print(f"Loading {input_path} (safetensors format)...")
    state_dict = load_file(input_path)
    
    # Load into model
    model = ByteLSTM()
    model.load_state_dict(state_dict)
    model.eval()
    
    # Quantize
    print("Quantizing to Int8...")
    torch.backends.quantized.engine = 'qnnpack'
    model_int8 = torch.quantization.quantize_dynamic(
        model, 
        {nn.LSTM, nn.Linear},
        dtype=torch.qint8
    )
    
    # Verify it works
    print("Verifying inference...")
    dummy_input = torch.zeros((1, 10), dtype=torch.long)
    with torch.no_grad():
        out, _ = model_int8(dummy_input)
        print(f"Output shape: {out.shape}")
    
    # Save quantized
    print(f"Saving to {output_path}...")
    torch.save(model_int8.state_dict(), output_path)
    
    size_mb = os.path.getsize(output_path) / (1024*1024)
    print(f"Final Size: {size_mb:.3f} MB")
    
    if size_mb <= 1.0:
        print("✅ SUCCESS: Model fits in 1MB!")
        # Rename to final name
        os.rename(output_path, "submission/model.pt")
        print("Renamed to submission/model.pt")
    else:
        print("❌ WARNING: Model is still too big.")

if __name__ == "__main__":
    compress()
