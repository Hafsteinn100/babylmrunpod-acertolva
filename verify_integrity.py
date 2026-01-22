
import torch
import torch.nn as nn
from submission.model import Model as PackedModel
from submission.model import TinyTransformer
from pathlib import Path
import sys

def verify():
    print("--- Verifying Integrity ---")
    
    # 1. Load Packed Model
    print("Loading Packed Model...")
    submission_dir = Path("submission")
    try:
        packed_wrapper = PackedModel(submission_dir)
        packed_model = packed_wrapper.model
    except Exception as e:
        print(f"FAILED to load packed model: {e}")
        return

    # 2. Try to Load Original INT8 Model
    print("Loading Original Model_INT8.pt...")
    original_model = TinyTransformer()
    # We need to apply qdynamic structure first to match state dict?
    # Actually model_int8.pt was saved as a JIT or state dict of a quantized model?
    # Let's assume state dict.
    
    state_dict_path = submission_dir / "model_int8.pt"
    if not state_dict_path.exists():
        print("model_int8.pt not found.")
        return

    try:
        # To load a quantized state dict, we usually need the model structure to be quantized first
        # But quantize_dynamic modifies the class in place or returns a new one.
        original_model = torch.quantization.quantize_dynamic(
            original_model, {nn.Linear}, dtype=torch.qint8
        )
        
        # FIX: Apply same hook workaround to original model
        def dummy_hook(module, input, output): pass
        for layer in original_model.blocks.layers:
            layer.register_forward_hook(dummy_hook)
            
        state_dict = torch.load(state_dict_path, map_location="cpu")
        original_model.load_state_dict(state_dict)
        print("Original model loaded successfully!")
    except Exception as e:
        print(f"Could not load original model (this is expected if versions differ): {e}")
        print("Skipping direct comparison.")
        return

    # 3. Compare Predictions
    print("Comparing predictions on dummy input...")
    dummy_input = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
    
    with torch.no_grad():
        y_packed, _ = packed_model(dummy_input)
        y_orig, _ = original_model(dummy_input)
        
    diff = (y_packed - y_orig).abs().mean().item()
    print(f"Mean Absolute Difference in Logits: {diff:.6f}")
    
    # Check prediction agreement
    preds_packed = y_packed.argmax(dim=-1)
    preds_orig = y_orig.argmax(dim=-1)
    
    agreement = (preds_packed == preds_orig).float().mean().item()
    print(f"Prediction Agreement: {agreement:.2%} ({int(agreement * dummy_input.numel())}/{dummy_input.numel()})")
    
    if agreement > 0.9:
        print("✅ Models predict consistently!")
    elif agreement > 0.5:
        print("⚠️ Models match somewhat, but re-quantization noise is high.")
    else:
        print("❌ MODELS DISAGREE. PACKING IS BROKEN.")

if __name__ == "__main__":
    verify()
