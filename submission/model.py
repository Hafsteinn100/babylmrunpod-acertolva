"""
Byte-level LSTM Model for Golden Plate NLP Challenge.
INT8 Quantized version (larger model, 400 hidden units).
"""
import torch
import torch.nn as nn
from pathlib import Path

# Configuration (Must match training)
EMBED_DIM = 64
HIDDEN_DIM = 440  # Reduced to fit 1MB limit
VOCAB_SIZE = 256
SEQ_LEN = 128

class ByteLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, num_layers=1, batch_first=True)
        self.head = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
    
    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.head(out)
        return logits, hidden

class Model:
    def __init__(self, submission_dir: Path):
        self.device = torch.device("cpu")
        
        # Create base model
        base_model = ByteLSTM()
        base_model.eval()
        
        # Apply dynamic quantization (must match training)
        torch.backends.quantized.engine = 'qnnpack'
        self.model = torch.quantization.quantize_dynamic(
            base_model,
            {nn.LSTM, nn.Linear},
            dtype=torch.qint8
        )
        self.model.to(self.device)
        
        # Load weights (INT8 format, saved with torch.save)
        weights_path = submission_dir / "model_int8.pt"
        if weights_path.exists():
            print(f"Loading INT8 weights from {weights_path}...")
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state_dict)
        else:
            print("Warning: No weights found at model_int8.pt")
            
    def predict(self, contexts: list[list[int]]) -> list[list[float]]:
        batch_size = len(contexts)
        if batch_size == 0:
            return []
            
        max_len = max(len(c) for c in contexts)
        if max_len == 0:
            return [[0.0] * 256] * batch_size
            
        # Pad sequences
        padded = torch.zeros((batch_size, max_len), dtype=torch.long)
        last_indices = []
        
        for i, ctx in enumerate(contexts):
            l = len(ctx)
            if l > 0:
                padded[i, :l] = torch.tensor(ctx, dtype=torch.long)
                last_indices.append(l - 1)
            else:
                last_indices.append(0)

        padded = padded.to(self.device)
        
        with torch.no_grad():
            logits, _ = self.model(padded)
            logits_np = logits.cpu().numpy()
            
            final_logits = []
            for i, idx in enumerate(last_indices):
                final_logits.append(logits_np[i, idx].tolist())
                    
        return final_logits
