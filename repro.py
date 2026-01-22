
import torch
import torch.nn as nn

def test():
    try:
        t = 10
        mask = nn.Transformer.generate_square_subsequent_mask(t)
        print(f"Mask type: {type(mask)}")
        print(f"Mask device: {mask.device}")
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=160, nhead=4)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        src = torch.randn(10, 32, 160)
        out = transformer_encoder(src, mask) 
        print("Success")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
