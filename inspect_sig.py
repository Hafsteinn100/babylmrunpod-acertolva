
import torch
import torch.nn as nn
import inspect

def test():
    sig = inspect.signature(nn.TransformerEncoder.forward)
    print(f"TransformerEncoder.forward signature: {sig}")
    
    layer = nn.TransformerEncoderLayer(d_model=160, nhead=4)
    sig_layer = inspect.signature(layer.forward)
    print(f"TransformerEncoderLayer.forward signature: {sig_layer}")

if __name__ == "__main__":
    test()
