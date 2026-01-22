import zipfile
import os

def compress_lzma():
    print("Compressing using LZMA (Method 14)...")
    with zipfile.ZipFile('submission_lzma.zip', 'w', compression=zipfile.ZIP_LZMA) as zf:
        zf.write('submission/model.py', arcname='model.py')
        zf.write('submission/model_int8.pt', arcname='model_int8.pt')
    
    size = os.path.getsize('submission_lzma.zip') / (1024*1024)
    print(f"LZMA Size: {size:.3f} MB")

if __name__ == "__main__":
    compress_lzma()
