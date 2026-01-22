
import time
import torch
import random
from pathlib import Path
from submission.model import Model
import os

def benchmark_threads():
    print("Benchmarking Threading Scaling...")
    
    # Generate data once
    vocab_size = 256
    seq_len = 512
    batch_size = 1024
    contexts = []
    for _ in range(batch_size):
        l = random.randint(10, seq_len)
        ctx = [random.randint(0, vocab_size-1) for _ in range(l)]
        contexts.append(ctx)
        
    thread_counts = [1, 2, 4, 8]
    
    for t in thread_counts:
        print(f"\n--- Threads: {t} ---")
        torch.set_num_threads(t)
        
        # Reload model to ensure clean state if needed (though set_num_threads should work runtime)
        submission_dir = Path("submission")
        model = Model(submission_dir)
        
        # Warmup
        model.predict(contexts[:32])
        
        start = time.time()
        model.predict(contexts)
        end = time.time()
        
        duration = end - start
        speed = batch_size / duration
        print(f"Time: {duration:.2f}s")
        print(f"Speed: {speed:.1f} items/s")
        print(f"Projected 88k Time: {88000/speed/60:.2f} min")

if __name__ == "__main__":
    benchmark_threads()
