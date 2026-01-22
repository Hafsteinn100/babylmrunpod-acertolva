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
        
        # Mask creation should be efficient
        mask = nn.Transformer.generate_square_subsequent_mask(t).to(x.device)
        

        x = self.blocks(x, mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, None

class Model:
    def __init__(self, submission_dir: Path):
        self.device = torch.device("cpu")
            
        # Create base model structure (STANDARD FP32)
        base_model = TinyTransformer()
        base_model.eval()
        self.model = base_model 
        self.model.to(self.device)
        
        # Load weights
        weights_path = submission_dir / "model_packed.pt"
        if not weights_path.exists():
            weights_path = submission_dir / "model_int8.pt"
            
        if weights_path.exists():
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                
                # --- UNPACKING LOGIC ---
                unpacked_dict = {}
                for k, v in state_dict.items():
                    if isinstance(v, tuple) and len(v) == 4 and v[0] == 'MANUAL':
                        # Dequantize: (q - z) * s
                        _, q_v, scale, zero_point = v
                        v_recon = (q_v.float() - zero_point) * scale
                        unpacked_dict[k] = v_recon
                    else:
                        unpacked_dict[k] = v
                
                self.model.load_state_dict(unpacked_dict, strict=False)
                # print("Successfully loaded model.")
            except Exception as e:
                pass # Ideally log this
        else:
            pass
            
        # FAST INFERENCE: Re-enable Dynamic Quantization!
        # Convert the (now FP32) Linear layers to INT8 kernels for speed.
        torch.backends.quantized.engine = 'qnnpack'
        self.model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear},
            dtype=torch.qint8
        )
        
        # CRITICAL HACK: Disable "sparsity fast path" in TransformerEncoderLayer
        # The standard implementation checks self.linear1.weight.device, but for 
        # QuantizedLinear, .weight is a method, causing an AttributeError.
        # Adding a hook disables this check.
        def dummy_hook(module, input, output):
            pass
            
        for layer in self.model.blocks.layers:
            layer.register_forward_hook(dummy_hook)
        self.model.to(self.device)

    def predict(self, contexts: list[list[int]]) -> list[list[float]]:
        # OOM Fix: Process in mini-batches!
        # OPTIMIZATION: Increased buffer since we optimized the extraction logic logic
        mini_batch_size = 64 
        final_results = []
        
        for i in range(0, len(contexts), mini_batch_size):
            batch_contexts = contexts[i : i + mini_batch_size]
            
            # --- Mini-Batch Processing ---
            batch_size = len(batch_contexts)
            if batch_size == 0: continue
            
            # Truncate
            # TUNING: 512 was ~32 mins (Timeout). 256 was ~5 mins (Bad Score).
            # 384 is the "Goldilocks" zone: Projected ~24 mins.
            max_limit = 384 
            processed_contexts = []
            for c in batch_contexts:
                if len(c) > max_limit:
                    processed_contexts.append(c[-max_limit:])
                else:
                    processed_contexts.append(c)
            
            # Pad
            max_len = max(len(c) for c in processed_contexts)
            if max_len == 0: 
                final_results.extend([[0.0]*256] * batch_size)
                continue
            
            padded = torch.zeros((batch_size, max_len), dtype=torch.long)
            last_indices = []
            
            for idx, ctx in enumerate(processed_contexts):
                l = len(ctx)
                if l > 0:
                    padded[idx, :l] = torch.tensor(ctx, dtype=torch.long)
                    last_indices.append(l - 1)
                else:
                    last_indices.append(0)
                    
            padded = padded.to(self.device)
            
            with torch.no_grad():
                logits, _ = self.model(padded)
                # logits: [B, T, V]
                
                # OPTIMIZATION: Extract only LAST token on Device
                # Avoid moving the full [B, T, V] tensor to CPU
                batch_indices = torch.arange(batch_size, device=self.device)
                last_indices_tensor = torch.tensor(last_indices, device=self.device)
                
                # Advanced Indexing: [0..B-1, last_indices_tensor]
                # Result shape: [B, V]
                final_token_logits = logits[batch_indices, last_indices_tensor]
                
                # Move only the small result to list
                final_results.extend(final_token_logits.tolist())
                
        return final_results
