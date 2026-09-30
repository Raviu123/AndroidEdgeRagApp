import os
import re
import math
import pypdf
import numpy as np
import onnxruntime as ort

pdf_path = r"C:\Users\siddu\OfflineDocAI\Grafting_15_Page_Guide.pdf"
onnx_path = r"C:\Users\siddu\OfflineDocAI\app\src\main\assets\models\embedding\all-MiniLM-L6-v2.onnx"
vocab_path = r"C:\Users\siddu\OfflineDocAI\app\src\main\assets\models\embedding\vocab.txt"

# 1. Load Vocab
vocab = {}
with open(vocab_path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        vocab[line.strip()] = idx

def tokenize(text, max_len=256):
    tokens = ["[CLS]"]
    words = re.findall(r"\w+|[^\w\s]", text.lower())
    for w in words:
        if w in vocab:
            tokens.append(w)
        else:
            # subword fallback
            subwords = []
            start = 0
            while start < len(w):
                end = len(w)
                cur_sub = None
                while start < end:
                    sub = w[start:end]
                    if start > 0:
                        sub = "##" + sub
                    if sub in vocab:
                        cur_sub = sub
                        break
                    end -= 1
                if cur_sub is None:
                    tokens.append("[UNK]")
                    break
                tokens.append(cur_sub)
                start = end
    tokens.append("[SEP]")
    if len(tokens) > max_len:
        tokens = tokens[:max_len-1] + ["[SEP]"]
    
    input_ids = [vocab.get(t, vocab.get("[UNK]", 100)) for t in tokens]
    attention_mask = [1] * len(input_ids)
    token_type_ids = [0] * len(input_ids)
    
    # Pad to max_len or dynamic length
    return np.array([input_ids], dtype=np.int64), np.array([attention_mask], dtype=np.int64), np.array([token_type_ids], dtype=np.int64)

# 2. Setup ONNX Session
session = ort.InferenceSession(onnx_path)

def get_embedding(text):
    ids, mask, type_ids = tokenize(text)
    inputs = {
        "input_ids": ids,
        "attention_mask": mask,
        "token_type_ids": type_ids
    }
    outputs = session.run(None, inputs)
    # Mean pooling
    token_embeddings = outputs[0] # [1, seq_len, 384]
    input_mask_expanded = np.expand_dims(mask, -1) # [1, seq_len, 1]
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    mean_pooled = sum_embeddings / sum_mask
    # L2 normalize
    norm = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
    norm = np.clip(norm, a_min=1e-12, a_max=None)
    normalized = mean_pooled / norm
    return normalized[0]

# 3. Extract and Chunk PDF
reader = pypdf.PdfReader(pdf_path)
chunks = []
for i, page in enumerate(reader.pages, 1):
    text = page.extract_text().strip()
    emb = get_embedding(text)
    chunks.append({
        "page": i,
        "text": text,
        "embedding": emb
    })

print(f"Extracted and embedded {len(chunks)} chunks from {pdf_path}")

# 4. Evaluation Queries
test_queries = [
    ("What is grafting?", 1),
    ("What temperature is recommended for healing?", 11),
    ("What is the optimal healing temperature for grafting?", 11),
    ("Types of grafting", [3, 4, 5, 6, 7, 8, 9, 10]),
    ("What tools are required for grafting?", 12),
    ("What causes grafting failure?", 14),
    ("Does the document mention Falcon Heavy?", None)
]

print("\n=== RETRIEVAL BENCHMARK VERIFICATION ===")
for q, expected in test_queries:
    q_emb = get_embedding(q)
    scored = []
    for c in chunks:
        # Cosine similarity (both normalized so dot product)
        score = float(np.dot(q_emb, c["embedding"]))
        scored.append((c["page"], score, c["text"][:80]))
    scored.sort(key=lambda x: x[1], reverse=True)
    
    top3 = scored[:3]
    top_page = top3[0][0]
    
    status = "UNKNOWN"
    if expected is None:
        status = f"PASS (Top score {top3[0][1]:.3f} for absent fact)"
    elif isinstance(expected, int):
        status = "PASS" if top_page == expected else f"FAIL (Got Page {top_page})"
    elif isinstance(expected, list):
        top_pages = [s[0] for s in scored[:len(expected)]]
        overlap = set(top_pages).intersection(set(expected))
        status = f"PASS ({len(overlap)}/{len(expected)} target pages in top {len(expected)})"
        
    print(f"\nQuery: '{q}' -> Expected: Page {expected} -> Status: {status}")
    for rank, (p, s, preview) in enumerate(top3, 1):
        print(f"  Rank {rank}: Page {p:2d} (Score: {s:.4f}) | {preview}...")
