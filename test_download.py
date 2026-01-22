from huggingface_hub import hf_hub_download
import os

print("Testing download logic for RunPod...")
try:
    # Test just the smallest file
    filename = "igc_wiki/igc_wiki.zip"
    print(f"Attempting to download {filename} from arnastofnun/IGC-2024...")
    
    local_path = hf_hub_download(
        repo_id="arnastofnun/IGC-2024",
        repo_type="dataset",
        filename=filename,
    )
    print(f"✅ Success! Downloaded to {local_path}")
except Exception as e:
    print(f"❌ Failed: {e}")
