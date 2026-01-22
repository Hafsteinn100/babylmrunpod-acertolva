
import torch.nn.modules.transformer as t
import inspect
import sys

def read_source():
    fname = t.__file__
    print(f"Reading {fname}")
    try:
        with open(fname, 'r') as f:
            lines = f.readlines()
            start = 850
            end = 880
            for i in range(start, min(end, len(lines))):
                print(f"{i+1}: {lines[i].rstrip()}")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    read_source()
