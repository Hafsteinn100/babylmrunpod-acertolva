
import torch
import torch.nn.functional as F
from submission.model import Model
from pathlib import Path

def generate():
    print("Loading packed model...")
    # Initialize properly
    submission_dir = Path("submission")
    wrapper = Model(submission_dir)
    model = wrapper.model
    device = wrapper.device
    
    # Text Prompt
    # "The weather in Iceland is"
    prompt_str = "The weather in Iceland is"
    prompt = list(prompt_str.encode("utf-8"))
    
    print(f"Prompt: {prompt_str}")
    
    # Generation Config
    num_generate = 100
    temperature = 1.0
    top_k = 10
    
    model.eval()
    generated = list(prompt)
    
    print("Generating...", end="", flush=True)
    
    with torch.no_grad():
        for i in range(num_generate):
            # Context window
            ctx = generated[-512:] 
            
            # Prepare input
            x = torch.tensor([ctx], dtype=torch.long, device=device)
            
            # Predict
            logits, _ = model(x)
            
            # Get last token logits
            last_logits = logits[0, -1, :] / temperature
            
            # Top-K Sampling
            v, _ = torch.topk(last_logits, top_k)
            last_logits[last_logits < v[[-1]]] = -float('Inf')
            
            probs = F.softmax(last_logits, dim=-1)
            
            # Sample
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            print(".", end="", flush=True)

    print("\n\n--- GENERATED TEXT ---")
    full_bytes = bytes(generated)
    print(full_bytes.decode("utf-8", errors="replace"))
    print("----------------------")

if __name__ == "__main__":
    generate()
