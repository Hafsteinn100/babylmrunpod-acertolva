
import torch
import torch.nn as nn
from pathlib import Path
import os
import argparse

# Constants needed for model definition
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

def compress_model(input_path, output_path):
    print(f"Loading {input_path}...")
    device = torch.device('cpu')
    
    # Load float32 model
    model = ByteLSTM()
    state_dict = torch.load(input_path, map_location=device, weights_only=False)
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
        model_int8(dummy_input)
    
    # Save
    print(f"Saving to {output_path}...")
    torch.save(model_int8.state_dict(), output_path)
    
    # Check size
    size_mb = os.path.getsize(output_path) / (1024*1024)
    print(f"Final Size: {size_mb:.3f} MB")
    
    if size_mb <= 1.0:
        print("✅ SUCCESS: Model fits in 1MB!")
    else:
        print("❌ WARNING: Model is still too big.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="submission/model.pt")
    parser.add_argument("--output", default="submission/model_quantized.pt")
    args = parser.parse_args()
    
    compress_model(args.input, args.output)
