import struct
import sys

def read_string(f):
    length_bytes = f.read(8)
    if not length_bytes or len(length_bytes) < 8:
        return None
    length = struct.unpack('<Q', length_bytes)[0]
    return f.read(length).decode('utf-8', errors='replace')

def read_val(f, val_type):
    # GGUF value types:
    # 0: UINT8, 1: INT8, 2: UINT16, 3: INT16, 4: UINT32, 5: INT32, 6: FLOAT32, 7: BOOL,
    # 8: STRING, 9: ARRAY, 10: UINT64, 11: INT64, 12: FLOAT64
    if val_type == 0:
        return struct.unpack('<B', f.read(1))[0]
    elif val_type == 1:
        return struct.unpack('<b', f.read(1))[0]
    elif val_type == 2:
        return struct.unpack('<H', f.read(2))[0]
    elif val_type == 3:
        return struct.unpack('<h', f.read(2))[0]
    elif val_type == 4:
        return struct.unpack('<I', f.read(4))[0]
    elif val_type == 5:
        return struct.unpack('<i', f.read(4))[0]
    elif val_type == 6:
        return struct.unpack('<f', f.read(4))[0]
    elif val_type == 7:
        return bool(struct.unpack('<B', f.read(1))[0])
    elif val_type == 8:
        return read_string(f)
    elif val_type == 9:
        elem_type = struct.unpack('<I', f.read(4))[0]
        elem_count = struct.unpack('<Q', f.read(8))[0]
        arr = []
        # if too large (like tokens/vocab), only read first few
        for i in range(elem_count):
            v = read_val(f, elem_type)
            if i < 5:
                arr.append(v)
        if elem_count > 5:
            arr.append(f"... ({elem_count} elements total)")
        return arr
    elif val_type == 10:
        return struct.unpack('<Q', f.read(8))[0]
    elif val_type == 11:
        return struct.unpack('<q', f.read(8))[0]
    elif val_type == 12:
        return struct.unpack('<d', f.read(8))[0]
    else:
        raise ValueError(f"Unknown type: {val_type}")

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else r"app/src/main/assets/models/llm/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b'GGUF':
            print("Not a GGUF file:", magic)
            return
        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        kv_count = struct.unpack('<Q', f.read(8))[0]
        print(f"GGUF Version: {version}, Tensors: {tensor_count}, Metadata KV pairs: {kv_count}")
        
        for _ in range(kv_count):
            key = read_string(f)
            val_type = struct.unpack('<I', f.read(4))[0]
            val = read_val(f, val_type)
            # Filter huge arrays
            if isinstance(val, list) and len(val) > 10:
                print(f"{key}: [array of {len(val)} items]")
            elif isinstance(val, str) and len(val) > 200:
                print(f"{key}: {val[:200]}...")
            else:
                print(f"{key}: {val}")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    main()
