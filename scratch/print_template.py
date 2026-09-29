import struct
import sys

def main():
    path = "app/src/main/assets/models/llm/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    with open(path, "rb") as f:
        magic = f.read(4)
        ver, t_cnt, kv_cnt = struct.unpack('<IQQ', f.read(20))
        for _ in range(kv_cnt):
            k_len = struct.unpack("<Q", f.read(8))[0]
            k = f.read(k_len).decode('utf-8', errors='replace')
            vt = struct.unpack("<I", f.read(4))[0]
            if vt == 8:
                v_len = struct.unpack("<Q", f.read(8))[0]
                v = f.read(v_len).decode('utf-8', errors='replace')
                if k == "tokenizer.chat_template":
                    print("--- TEMPLATE BEGIN ---")
                    print(v)
                    print("--- TEMPLATE END ---")
                    return
            elif vt in (0, 1, 7):
                f.seek(1, 1)
            elif vt in (2, 3):
                f.seek(2, 1)
            elif vt in (4, 5, 6):
                f.seek(4, 1)
            elif vt in (10, 11, 12):
                f.seek(8, 1)
            elif vt == 9:
                et, ec = struct.unpack("<IQ", f.read(12))
                if et == 8:
                    for _ in range(ec):
                        slen = struct.unpack("<Q", f.read(8))[0]
                        f.seek(slen, 1)
                elif et in (0, 1, 7):
                    f.seek(ec * 1, 1)
                elif et in (2, 3):
                    f.seek(ec * 2, 1)
                elif et in (4, 5, 6):
                    f.seek(ec * 4, 1)
                elif et in (10, 11, 12):
                    f.seek(ec * 8, 1)

if __name__ == "__main__":
    main()
