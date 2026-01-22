import torch
import torch.nn as nn
import zipfile
import os
import sys

# --- Re-define Model Structure to Load ---
VOCAB_SIZE = 256
D_MODEL = 160
N_LAYERS = 4
N_HEADS = 4
D_FF = 320
DROPOUT = 0.0
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

def quantize_attn():
    print("Loading partially quantized model...")
    # We need to construct the "Partially Quantized" structure to load the state_dict
    # This is tricky because the saved model has DynamicQuantizedLinear modules.
    
    # 1. Create Base
    base_model = TinyTransformer()
    base_model.eval()
    
    # 2. Apply same quantization as before (Linear Only)
    torch.backends.quantized.engine = 'qnnpack'
    model_partially_quant = torch.quantization.quantize_dynamic(
        base_model, {nn.Linear}, dtype=torch.qint8
    )
    
    # 3. Load Weights
    try:
        state_dict = torch.load('submission/model_int8.pt', map_location='cpu')
        model_partially_quant.load_state_dict(state_dict)
        print("Loaded successfully.")
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    # 4. Now Quantize Attention!
    print("Quantizing Attention Layers...")
    model_fully_quant = torch.quantization.quantize_dynamic(
        model_partially_quant, {nn.MultiheadAttention}, dtype=torch.qint8
    )
    
    # 5. Save
    output_pt = 'submission/model_int8_full.pt'
    torch.save(model_fully_quant.state_dict(), output_pt)
    
    # 6. Zip and Check
    output_zip = 'submission_full_quant.zip'
    with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write('submission/model.py', arcname='model.py')
        zf.write(output_pt, arcname='model_int8.pt')
        
    size = os.path.getsize(output_zip) / (1024*1024)
    print(f"\nFinal Fully Quantized Zip Size: {size:.3f} MB")
    
    # Verify Attention Weights are gone (checking keys)
    sd = torch.load(output_pt)
    print("\nChecking for large FP32 chunks:")
    for k, v in sd.items():
        if torch.is_tensor(v) and v.dtype == torch.float32 and v.numel() > 10000:
             print(f"  Warning: {k} is still FP32 ({v.numel()*4/1024:.2f} KB)")

if __name__ == "__main__":
    quantize_attn()
