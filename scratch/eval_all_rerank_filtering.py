import math
import onnxruntime as ort
import numpy as np
import sqlite3
import json

with open('vocab.txt', 'r', encoding='utf-8') as f:
    vocab = {line.strip(): i for i, line in enumerate(f)}

def tokenize_pair(q, text, max_len=256):
    tokens = ['[CLS]'] + q.lower().split() + ['[SEP]'] + text.lower().split() + ['[SEP]']
    input_ids = [vocab.get(t, vocab.get('[UNK]', 100)) for t in tokens][:max_len]
    token_type_ids = [0 if i <= tokens.index('[SEP]') else 1 for i in range(len(input_ids))]
    attention_mask = [1] * len(input_ids)
    pad_len = max_len - len(input_ids)
    input_ids += [0] * pad_len
    token_type_ids += [0] * pad_len
    attention_mask += [0] * pad_len
    return np.array([input_ids], dtype=np.int64), np.array([attention_mask], dtype=np.int64), np.array([token_type_ids], dtype=np.int64)

session_rerank = ort.InferenceSession('ms-marco-MiniLM-L-6-v2.onnx')

conn = sqlite3.connect('offline_docai_device.db')
c = conn.cursor()

# Check Turn 1 to Turn 20: what were the rerank scores of the chunks that were ACTUALLY retained in DB?
c.execute("SELECT id, sources FROM messages WHERE conversationId = 17 AND role = 'assistant';")
rows = c.fetchall()

print("Inspect retained chunks rerank scores in Conv 17:")
for mid, sources_json in rows[:5]:
    if not sources_json: continue
    sources = json.loads(sources_json)
    scores = [s.get('score', 0) for s in sources]
    print(f"Msg {mid}: {len(sources)} sources. Sigmoid scores in DB: {[round(x, 4) for x in scores[:5]]}")
    # Sigmoid to logit: log(p / (1 - p))
    logits = [math.log(x / (1.0 - x + 1e-7)) if 0 < x < 1 else -99 for x in scores]
    print(f"  Logits approx: {[round(l, 2) for l in logits[:5]]}")
