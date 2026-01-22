
import time
import torch
import random
from pathlib import Path
from submission.model import Model

def benchmark():
    print("Initializing Model...")
    submission_dir = Path("submission")
    model = Model(submission_dir)
    print("Model loaded.")
    
    # Configuration
    vocab_size = 256
    seq_len = 512
    batch_size = 1024
    num_batches = 5
    
    print(f"Benchmarking Inference Speed...")
    print(f"Batch Size: {batch_size}")
    print(f"Seq Len:    {seq_len}")
    print(f"Batches:    {num_batches}")
    
    total_time = 0
    total_items = 0
    
    # Warmup
    print("Warming up...")
    dummy_ctx = [[random.randint(0, vocab_size-1) for _ in range(random.randint(10, seq_len))] for _ in range(10)]
    model.predict(dummy_ctx)
    
    for i in range(num_batches):
        # Generate random contexts of variable length
        contexts = []
        for _ in range(batch_size):
            l = random.randint(10, seq_len)
            ctx = [random.randint(0, vocab_size-1) for _ in range(l)]
            contexts.append(ctx)
            
        start = time.time()
        _ = model.predict(contexts)
        end = time.time()
        
        duration = end - start
        total_time += duration
        total_items += batch_size
        
        print(f"Batch {i+1}: {duration:.2f}s ({batch_size/duration:.1f} items/s)")
        
    avg_speed = total_items / total_time
    print("-" * 30)
    print(f"Total Time:  {total_time:.2f}s")
    print(f"Total Items: {total_items}")
    print(f"Avg Speed:   {avg_speed:.1f} items/s")
    print(f"Time per 1k: {1000/avg_speed:.2f}s")
    print("-" * 30)

if __name__ == "__main__":
    benchmark()
