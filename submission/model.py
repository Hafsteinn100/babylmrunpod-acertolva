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
        
    def _init_weights(self, module):
        pass

    def forward(self, x):
        b, t = x.size()
        pos = torch.arange(0, t, dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.tok_emb(x) + self.pos_emb[:, :t, :]
        mask = nn.Transformer.generate_square_subsequent_mask(t).to(x.device)
        x = self.blocks(x, mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, None

class Model:
    def __init__(self, submission_dir: Path):
        self.device = torch.device("cpu")
            
        # Create base model structure
        base_model = TinyTransformer()
        base_model.eval()
        
        # Prepare for quantization (Dynamic INT8)
        # MUST MATCH TRAINING ENGINE
        torch.backends.quantized.engine = 'fbgemm'
        
        self.model = torch.quantization.quantize_dynamic(
            base_model,
            {nn.Linear},
            dtype=torch.qint8
        )
        self.model.to(self.device)
        
        # Load weights
        weights_path = submission_dir / "model_int8.pt"
        if weights_path.exists():
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                # print("Successfully loaded model.")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"Weights not found at {weights_path}")

    def predict(self, contexts: list[list[int]]) -> list[list[float]]:
        # Process in batches
        mini_batch_size = 64 
        final_results = []
        
        for i in range(0, len(contexts), mini_batch_size):
            batch_contexts = contexts[i : i + mini_batch_size]
            batch_size = len(batch_contexts)
            if batch_size == 0: continue
            
            # Pad
            max_len = max(len(c) for c in batch_contexts)
            if max_len == 0: 
                final_results.extend([[0.0]*256] * batch_size)
                continue
            
            # Truncate to SEQ_LEN if inputs are too long (unlikely in this challenge but safe)
            max_len = min(max_len, SEQ_LEN)

            padded = torch.zeros((batch_size, max_len), dtype=torch.long)
            last_indices = []
            
            for idx, ctx in enumerate(batch_contexts):
                ctx_len = len(ctx)
                if ctx_len > max_len:
                    # Keep last max_len content
                    padded[idx, :] = torch.tensor(ctx[-max_len:], dtype=torch.long)
                    last_indices.append(max_len - 1)
                elif ctx_len > 0:
                    padded[idx, :ctx_len] = torch.tensor(ctx, dtype=torch.long)
                    last_indices.append(ctx_len - 1)
                else:
                    last_indices.append(0)
                    
            padded = padded.to(self.device)
            
            with torch.no_grad():
                logits, _ = self.model(padded)
                
                # Extract only LAST token
                batch_indices = torch.arange(batch_size, device=self.device)
                last_indices_tensor = torch.tensor(last_indices, device=self.device)
                final_token_logits = logits[batch_indices, last_indices_tensor]
                
                final_results.extend(final_token_logits.tolist())
                
        return final_results
