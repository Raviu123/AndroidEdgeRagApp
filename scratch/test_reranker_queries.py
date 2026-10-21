import onnxruntime as ort
import numpy as np
import sqlite3

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

c.execute('SELECT text FROM chunks WHERE id = 1538;')
t1538 = c.fetchone()[0]
c.execute('SELECT text FROM chunks WHERE id = 1539;')
t1539 = c.fetchone()[0]

test_queries = [
    'When was construction postponed due to weather?',
    'administrators announced a temporary change in the construction schedule due to weather conditions',
    'construction postponed rain storm weather',
    'Tirion Fordring storm and rain construction postponed'
]

for q in test_queries:
    ids, mask, types = tokenize_pair(q, t1538)
    out = session_rerank.run(None, {'input_ids': ids, 'attention_mask': mask, 'token_type_ids': types})
    logit = float(out[0][0][0])
    ids9, mask9, types9 = tokenize_pair(q, t1539)
    out9 = session_rerank.run(None, {'input_ids': ids9, 'attention_mask': mask9, 'token_type_ids': types9})
    logit9 = float(out9[0][0][0])
    print(f"Query: '{q[:40]}...' -> 1538 logit: {logit:.2f}, 1539 logit: {logit9:.2f}")
