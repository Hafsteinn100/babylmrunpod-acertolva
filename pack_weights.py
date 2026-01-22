import torch
import torch.nn as nn
import zipfile
import os

def pack_weights_fake_quant():
    print("Loading model_int8.pt...")
    # Map location CPU is important
    state_dict = torch.load('submission/model_int8.pt', map_location='cpu')
    
    packed_dict = {}
    
    print("\nState Dict Keys Found:")
    for k, v in state_dict.items():
        print(f"  {k}: {type(v)}")

    print("\nConverting ALL weights to Manual INT8 Storage...")
    
    for k, v in state_dict.items():
        # Case A: Already Manual Tuple (from previous runs? No, starting fresh usually)
        # Case B: Dynamic Quantized Linear (_packed_params)
        if '_packed_params' in k:
            # Format: 'blocks.layers.0.linear1._packed_params'
            # We want to convert this to 'blocks.layers.0.linear1.weight' and '...bias'
            
            # Key for the linear layer
            prefix = k.replace('._packed_params', '')
            
            if isinstance(v, tuple):
                # v is (weight_packed, bias)
                w_packed, bias = v
                
                # Extract Weight info
                # w_packed is torch.ops.quantized.LinearPackedParams (or similar)
                # We need to unpack it.
                # Usually w_packed.unpack() returns (weight_q_tensor, bias)
                try:
                    w_q, b_check = torch.ops.quantized.linear_unpack(w_packed)
                except:
                    # Fallback for newer PyTorch
                    try: 
                        w_q, b_check = w_packed.unpack()
                    except:
                        if isinstance(w_packed, torch.Tensor) and w_packed.is_quantized:
                            # It is the quantized weight tensor itself!
                            w_q = w_packed
                            w_int8 = w_q.int_repr()
                            w_scale = w_q.q_scale()
                            w_zp = w_q.q_zero_point()
                            
                            packed_dict[f"{prefix}.weight"] = ('MANUAL', w_int8, w_scale, w_zp)
                            # Bias lost/fused? Assume None.
                            print(f"  Extracted {k} -> {prefix}.weight [INT8 Tensor]")
                            continue
                        else:
                            print(f"FAILED to unpack {k}. Type: {type(w_packed)}")
                            continue
                
                # w_q is a Quantized Tensor (qint8)
                # Extract int8 values and scale/zp
                w_int8 = w_q.int_repr()
                w_scale = w_q.q_scale()
                w_zp = w_q.q_zero_point()
                
                # Store Weight as Manual Tuple
                packed_dict[f"{prefix}.weight"] = ('MANUAL', w_int8, w_scale, w_zp)
                
                # Store Bias (usually Float)
                if bias is not None:
                    packed_dict[f"{prefix}.bias"] = bias
                elif b_check is not None:
                    packed_dict[f"{prefix}.bias"] = b_check
                    
                print(f"  Unpacked {k} -> {prefix}.weight [INT8]")
                
        # Case C: Method 2 (suffix ._packed_params.dtype etc - ignore these metadata keys)
        elif k.endswith('._packed_params.dtype') or k.endswith('._packed_params._packed_params'):
            continue # Skip these metadata keys
            
        # Case D: FP32 Tensor (Embeddings, Attention)
        elif torch.is_tensor(v) and v.dtype == torch.float32:
             # Manual Quantization (Same as before)
            min_val = v.min().item()
            max_val = v.max().item()
            scale = (max_val - min_val) / 255.0
            if scale == 0: scale = 1e-6
            zero_point = -128 - min_val / scale
            q_v = torch.clamp((v / scale + zero_point), -128, 127).to(torch.int8)
            
            # Verify MSE
            # v_recon = (q_v.float() - zero_point) * scale
            # mse = torch.mean((v - v_recon)**2).item()
            
            packed_dict[k] = ('MANUAL', q_v, scale, zero_point)
            if v.numel() > 1000:
                print(f"  Quantized {k} -> INT8")
        
        # Case E: Other (e.g. buffers, int64 params)
        else:
            packed_dict[k] = v

    print("\nSaving packed model...")
    torch.save(packed_dict, 'submission/model_packed.pt')
    
    # Zip it
    with zipfile.ZipFile('submission_packed.zip', 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write('submission/model.py', arcname='model.py')
        zf.write('submission/model_packed.pt', arcname='model_packed.pt')
        
    size = os.path.getsize('submission_packed.zip') / (1024*1024)
    print(f"\nPacked Zip Size: {size:.3f} MB")

if __name__ == "__main__":
    pack_weights_fake_quant()
