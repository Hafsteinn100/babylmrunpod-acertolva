
import torch
import torch.nn as nn
import time
import os
from pathlib import Path

# Define the model structure (same as current)
class ByteLSTM(nn.Module):
    def __init__(self, hidden_dim=180):
        super().__init__()
        self.embed = nn.Embedding(256, 64)
        self.lstm = nn.LSTM(64, hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_dim, 256)
        
    def forward(self, x, hidden=None):
        emb = self.embed(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.head(out)
        return logits, hidden

def test_quantization():
    print("Testing quantization...")
    
    # Set quantization engine for ARM64/Mac
    torch.backends.quantized.engine = 'qnnpack'
    
    # 1. Create standard float32 model
    model_fp32 = ByteLSTM(hidden_dim=180)
    model_fp32.eval()
    
    # Save fp32 size
    torch.save(model_fp32.state_dict(), "test_fp32.pt")
    size_fp32 = os.path.getsize("test_fp32.pt") / (1024*1024)
    print(f"FP32 Size: {size_fp32:.3f} MB")
    
    # 2. Quantize to Int8
    print("Quantizing to Int8...")
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32, 
        {nn.LSTM, nn.Linear}, 
        dtype=torch.qint8
    )
    
    # Save int8 size
    # Note: simple state_dict save might not preserve quantization wrapper structure easily for loading
    # usually we save the whole quantized model or trace it. 
    # For size check, torch.save works on the object.
    torch.save(model_int8.state_dict(), "test_int8.pt")
    size_int8 = os.path.getsize("test_int8.pt") / (1024*1024)
    print(f"Int8 Size: {size_int8:.3f} MB")
    print(f"Reduction: {size_fp32 / size_int8:.1f}x")
    
    # 3. Test Inference Speed
    input_data = torch.randint(0, 256, (1, 128)) # Batch 1, Seq 128
    
    # Warmup
    for _ in range(10):
        model_fp32(input_data)
        model_int8(input_data)
        
    # Timing FP32
    start = time.time()
    for _ in range(100):
        model_fp32(input_data)
    time_fp32 = time.time() - start
    print(f"FP32 Time (100 runs): {time_fp32:.4f}s")
    
    # Timing Int8
    start = time.time()
    for _ in range(100):
        model_int8(input_data)
    time_int8 = time.time() - start
    print(f"Int8 Time (100 runs): {time_int8:.4f}s")
    
    # 4. Try LARGER model
    print("\nTesting LARGER model capability...")
    # Try 512 hidden units (approx 3x larger params than 180)
    large_dim = 450
    model_large = ByteLSTM(hidden_dim=large_dim)
    model_large.eval()
    model_large_int8 = torch.quantization.quantize_dynamic(
        model_large, 
        {nn.LSTM, nn.Linear},
        dtype=torch.qint8
    )
    torch.save(model_large_int8.state_dict(), "test_large_int8.pt")
    size_large = os.path.getsize("test_large_int8.pt") / (1024*1024)
    print(f"Hidden Dim {large_dim} Int8 Size: {size_large:.3f} MB")
    
    # Cleanup
    for f in ["test_fp32.pt", "test_int8.pt", "test_large_int8.pt"]:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    test_quantization()
