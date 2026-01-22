
import torch
import sys
from pathlib import Path
from submission.model import Model
import traceback

def test():
    try:
        print("Initializing Model...")
        submission_dir = Path("submission")
        if not submission_dir.exists():
            print("Submission dir not found, creating dummy")
            submission_dir.mkdir(exist_ok=True)
            
        m = Model(submission_dir)
        print("Model initialized.")
        
        contexts = [
            [1, 2, 3],
            [4, 5, 6, 7],
            [8] * 64, 
            [] # Empty context
        ]
        
        # DEBUG: Inspect the quantized model structure
        print("Debugging Quantized Model Structure...")
        layer0 = m.model.blocks.layers[0]
        print(f"Layer 0 type: {type(layer0)}")
        print(f"Linear1 type: {type(layer0.linear1)}")
        
        if hasattr(layer0.linear1, 'weight'):
            w = layer0.linear1.weight
            print(f"Linear1.weight type: {type(w)}")
            if hasattr(w, 'device'):
                print(f"Linear1.weight device: {w.device}")
            else:
                print("Linear1.weight HAS NO DEVICE")
        else:
            print("Linear1 HAS NO WEIGHT ATTRIBUTE")
            
        print("Running predict...")
        logits = m.predict(contexts)
        print(f"Predict returned {len(logits)} items.")
        print("Success!")
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    test()
