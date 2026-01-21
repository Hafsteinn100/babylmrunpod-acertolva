"""
4-Layer INT8 Transformer Model for Golden Plate NLP Challenge.
"""
import torch
import torch.nn as nn
from pathlib import Path

# --- Configuration (MUST MATCH TRAINING) ---
VOCAB_SIZE = 256
D_MODEL = 160
N_LAYERS = 4
N_HEADS = 4
D_FF = 320
DROPOUT = 0.0 # No dropout in inference
SEQ_LEN = 512

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
        
        # Tie weights
        self.head.weight = self.tok_emb.weight

    def forward(self, x):
        b, t = x.size()
        pos = torch.arange(0, t, dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.tok_emb(x) + self.pos_emb[:, :t, :]
        mask = nn.Transformer.generate_square_subsequent_mask(t).to(x.device)
        x = self.blocks(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, None

class Model:
    def __init__(self, submission_dir: Path):
        self.device = torch.device("cpu")
        
        # Create base model structure
        base_model = TinyTransformer()
        base_model.eval()
        
        # Apply quantization structure
        torch.backends.quantized.engine = 'qnnpack'
        self.model = torch.quantization.quantize_dynamic(
            base_model,
            {nn.Linear},
            dtype=torch.qint8
        )
        self.model.to(self.device)
        
        # Load weights
        weights_path = submission_dir / "model_int8.pt"
        if weights_path.exists():
            print(f"Loading INT8 Transformer weights form {weights_path}...")
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(f"Warning: {weights_path} not found.")

    def predict(self, contexts: list[list[int]]) -> list[list[float]]:
        batch_size = len(contexts)
        if batch_size == 0: return []
        
        # Truncate context to max training length if needed
        max_limit = SEQ_LEN
        processed_contexts = []
        for c in contexts:
            if len(c) > max_limit:
                processed_contexts.append(c[-max_limit:])
            else:
                processed_contexts.append(c)
        
        # Pad
        max_len = max(len(c) for c in processed_contexts)
        if max_len == 0: return [[0.0]*256] * batch_size
        
        padded = torch.zeros((batch_size, max_len), dtype=torch.long)
        last_indices = []
        
        for i, ctx in enumerate(processed_contexts):
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
