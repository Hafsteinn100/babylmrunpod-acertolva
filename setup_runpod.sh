#!/bin/bash
echo "--- RunPod Setup ---"

# 1. Install Dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
# Ensure torch is installed (usually is, but good to be safe if base image is bare)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 2. Download Data
echo "Downloading Dataset (IGC-2024)... This may take a while..."
python create_dataset.py

echo "--- Setup Complete ---"
echo "To start training (adjust batch size for your GPU):"
echo "python train_optimized.py --batch_size 256"
