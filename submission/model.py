"""
Transformer Model for Golden Plate NLP Challenge.
(Hail Mary Attempt: FP16, Tied Weights, Transformer)
"""
import torch
import torch.nn as nn
from pathlib import Path
# Configuration (Must match training)
VOCAB_SIZE = 256
D_MODEL = 160
N_LAYERS = 2
N_HEADS = 4
D_FF = 320
SEQ_LEN = 512  # INCREASED

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Parameter(torch.zeros(1, SEQ_LEN, D_MODEL))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_FF,
            dropout=0.0, # No dropout during inference
            batch_first=True,
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok_emb.weight # Tie weights logic

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
        self.model = TinyTransformer()
        self.model.eval()
        self.model.to(self.device)
        
        weights_path = submission_dir / "model.safetensors"
        if weights_path.exists():
            print(f"Loading weights from {weights_path}...")
            # Load weights (saved as FP16, so we load, then likely convert model to float32 for CPU inference safety)
            # Actually, most CPUs support FP32 best. 
            # We load the half weights, but let's cast them to float32 for inference stability on all CPUS.
            state_dict = load_file(weights_path)
            
            # Cast to float32 explicitly
            new_state_dict = {k: v.float() for k, v in state_dict.items()}
            
            self.model.load_state_dict(new_state_dict)
        else:
            print("Warning: No weights found at model.safetensors")
            
    def predict(self, contexts: list[list[int]]) -> list[list[float]]:
        batch_size = len(contexts)
        if batch_size == 0: return []
            
        max_len = max(len(c) for c in contexts)
        if max_len == 0: return [[0.0] * 256] * batch_size
        
        # Limit to SEQ_LEN for Transformer (it has fixed position embeddings)
        actual_len = min(max_len, SEQ_LEN)
            
        padded = torch.zeros((batch_size, actual_len), dtype=torch.long)
        last_indices = []
        
        for i, ctx in enumerate(contexts):
            # Take last SEQ_LEN tokens
            ctx_trunc = ctx[-SEQ_LEN:]
            l = len(ctx_trunc)
            if l > 0:
                padded[i, :l] = torch.tensor(ctx_trunc, dtype=torch.long)
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
