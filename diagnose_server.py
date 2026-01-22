
import sys
import os
import torch
import glob

def check():
    print("=== SERVER DIAGNOSTICS ===")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    try:
        print(f"CPUs: {os.cpu_count()}")
    except:
        print("CPUs: Unknown")
    
    print("\n--- Directory Content ---")
    files = glob.glob("*")
    print(files)
    
    print("\n--- Model Load Test ---")
    try:
        from submission.model import Model
        from pathlib import Path
        m = Model(Path("submission"))
        print("Model instantiated successfully.")
        
        input_data = [[10, 20, 30]]
        print("Running prediction...")
        out = m.predict(input_data)
        print(f"Prediction successful. Output len: {len(out)}")
    except Exception as e:
        print(f"CRASH: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check()
