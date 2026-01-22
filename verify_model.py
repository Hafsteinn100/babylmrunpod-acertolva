import torch
import torch.nn as nn
from pathlib import Path
import sys

# Must match training config
VOCAB_SIZE = 256
D_MODEL = 160
N_LAYERS = 4      
N_HEADS = 4
D_FF = 320
DROPOUT = 0.1
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

def verify_model(model_path):
    print(f"Verifying {model_path}...")
    
    path = Path(model_path)
    if not path.exists():
        print(f"Error: {path} not found.")
        return False
        
    # 1. Check Size
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Size: {size_mb:.4f} MB")
    if size_mb > 1.0:
        print("FAIL: Model > 1MB")
    else:
        print("PASS: Model <= 1MB")

    # 2. Check Loadability
    try:
        # Load the QUANTIZED model
        model = TinyTransformer()
        model.eval()
        
        # Prepare for quantization to match saved state
        torch.backends.quantized.engine = 'qnnpack'
        model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        
        model.load_state_dict(torch.load(path))
        print("PASS: Model loaded successfully")
    except Exception as e:
        print(f"FAIL: Could not load model: {e}")
        return False

    # 3. Check Inference
    try:
        dummy_input = torch.randint(0, 256, (1, 128))
        with torch.no_grad():
            logits, _ = model(dummy_input)
        
        if logits.shape != (1, 128, 256):
            print(f"FAIL: Output shape mismatch. Expected (1, 128, 256), got {logits.shape}")
            return False
        
        print(f"PASS: Inference successful. Output shape: {logits.shape}")
        return True
    except Exception as e:
        print(f"FAIL: Inference failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        model_path = "submission/model_int8.pt"
    else:
        model_path = sys.argv[1]
        
    success = verify_model(model_path)
    if not success:
        sys.exit(1)
