import urllib.request
import os
import sys
import time

def download_model():
    url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    target_dir = r"app/src/main/assets/models/llm"
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, "Llama-3.2-1B-Instruct-Q4_K_M.gguf")

    if os.path.exists(target_path) and os.path.getsize(target_path) == 807694464:
        print(f"File already downloaded: {target_path} ({os.path.getsize(target_path)} bytes)")
        return target_path

    print(f"Downloading from {url} to {target_path}...")
    start_time = time.time()
    last_print = 0

    def reporthook(block_num, block_size, total_size):
        nonlocal last_print
        downloaded = block_num * block_size
        now = time.time()
        if now - last_print >= 2.0 or (total_size > 0 and downloaded >= total_size):
            pct = (downloaded / total_size * 100) if total_size > 0 else 0
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            speed = mb / (now - start_time) if now > start_time else 0
            print(f"Downloaded {mb:.1f} / {total_mb:.1f} MB ({pct:.1f}%) at {speed:.2f} MB/s", flush=True)
            last_print = now

    urllib.request.urlretrieve(url, target_path, reporthook=reporthook)
    print(f"\nDownload completed: {os.path.getsize(target_path)} bytes in {time.time() - start_time:.1f}s")
    return target_path

if __name__ == "__main__":
    download_model()
