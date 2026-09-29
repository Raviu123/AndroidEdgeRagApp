import os
import sys
import time
import urllib.request
import shutil

def download_file(url, target_path, expected_size=1117320736):
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    temp_path = target_path + ".tmp"
    
    initial_bytes = 0
    if os.path.exists(target_path):
        size = os.path.getsize(target_path)
        if size == expected_size:
            print(f"File already completely downloaded at {target_path} ({size} bytes)")
            return True
        else:
            print(f"File exists but size mismatch: {size} vs {expected_size}. Re-downloading...")
            
    if os.path.exists(temp_path):
        initial_bytes = os.path.getsize(temp_path)
        print(f"Resuming download from byte {initial_bytes}...")

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0")
    if initial_bytes > 0:
        req.add_header("Range", f"bytes={initial_bytes}-")

    print(f"Connecting to {url}...")
    try:
        with urllib.request.urlopen(req) as resp:
            status = resp.status
            print(f"Response status: {status}")
            
            # If server doesn't support range, restart
            mode = "ab" if status == 206 else "wb"
            if mode == "wb":
                initial_bytes = 0

            total_size = expected_size
            downloaded = initial_bytes
            start_time = time.time()
            last_print = 0

            with open(temp_path, mode) as out:
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_print >= 2.0 or downloaded >= total_size:
                        pct = (downloaded / total_size * 100) if total_size > 0 else 0
                        mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        speed = (downloaded - initial_bytes) / (now - start_time) / (1024 * 1024) if (now > start_time) else 0
                        print(f"Progress: {mb:.1f} / {total_mb:.1f} MB ({pct:.1f}%) at {speed:.2f} MB/s", flush=True)
                        last_print = now

            actual_size = os.path.getsize(temp_path)
            if actual_size == expected_size:
                if os.path.exists(target_path):
                    os.remove(target_path)
                os.rename(temp_path, target_path)
                print(f"Successfully downloaded {target_path} ({actual_size} bytes)")
                return True
            else:
                print(f"Download incomplete: {actual_size} != {expected_size}")
                return False

    except Exception as e:
        print(f"Download error: {e}", file=sys.stderr)
        return False

def main():
    url = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    asset_target = os.path.abspath("app/src/main/assets/models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    models_target = os.path.abspath("models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    
    # Download to models_target first, then copy to assets
    success = download_file(url, models_target)
    if not success:
        sys.exit(1)
        
    print(f"Copying to assets location: {asset_target}...")
    os.makedirs(os.path.dirname(asset_target), exist_ok=True)
    if not os.path.exists(asset_target) or os.path.getsize(asset_target) != os.path.getsize(models_target):
        shutil.copy2(models_target, asset_target)
    print(f"Both locations populated successfully!")

if __name__ == "__main__":
    main()
