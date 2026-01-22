import torch
import os
import zipfile

def shrink():
    print("Loading model_int8.pt...")
    state_dict = torch.load('submission/model_int8.pt', map_location='cpu')
    
    new_state_dict = {}
    
    print("\nAnalyzing Tensor Sizes:")
    total_bytes = 0
    fp32_bytes = 0
    
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            size = v.numel() * v.element_size()
            total_bytes += size
            dtype = str(v.dtype)
            
            # Print big tensors
            if size > 10000:
                print(f"  {k}: {dtype} - {size/1024:.2f} KB")
            
            # Cast FP32 to FP16
            if v.dtype == torch.float32:
                fp32_bytes += size
                new_state_dict[k] = v.half()
            else:
                new_state_dict[k] = v
        else:
            # PackedParams or other quantized objects
            new_state_dict[k] = v
            print(f"  {k}: {type(v)} (Keeping as is)")

    print(f"\nTotal Raw Bytes: {total_bytes/1024:.2f} KB")
    print(f"FP32 Bytes (Candidate for FP16): {fp32_bytes/1024:.2f} KB")
    
    # Save FP16 version
    torch.save(new_state_dict, 'submission/model_int8_fp16.pt')
    
    # Standard Zip
    with zipfile.ZipFile('submission_fp16.zip', 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write('submission/model.py', arcname='model.py')
        zf.write('submission/model_int8_fp16.pt', arcname='model_int8.pt')
        
    # LZMA Zip
    with zipfile.ZipFile('submission_fp16_lzma.zip', 'w', compression=zipfile.ZIP_LZMA) as zf:
        zf.write('submission/model.py', arcname='model.py')
        zf.write('submission/model_int8_fp16.pt', arcname='model_int8.pt')

    print("\n--- Results ---")
    s_orig = os.path.getsize('submission.zip') / (1024*1024)
    s_new = os.path.getsize('submission_fp16.zip') / (1024*1024)
    s_lzma = os.path.getsize('submission_fp16_lzma.zip') / (1024*1024)
    
    print(f"Original Zip: {s_orig:.3f} MB")
    print(f"FP16 Zip:     {s_new:.3f} MB")
    print(f"FP16 LZMA:    {s_lzma:.3f} MB")

if __name__ == "__main__":
    shrink()
