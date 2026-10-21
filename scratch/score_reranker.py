import onnxruntime as ort
import numpy as np
import sqlite3

# Load vocab
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

session = ort.InferenceSession('ms-marco-MiniLM-L-6-v2.onnx')

conn = sqlite3.connect('offline_docai_device.db')
c = conn.cursor()

question = "Did Adam Smith send a message to Li Hua about the upcoming building maintenance schedule before the administrators announced a temporary change in the construction schedule due to weather conditions?"

for cid in [553, 554, 803, 1538, 1539]:
    c.execute('SELECT text FROM chunks WHERE id = ?;', (cid,))
    t = c.fetchone()[0]
    ids, mask, types = tokenize_pair(question, t)
    out = session.run(None, {'input_ids': ids, 'attention_mask': mask, 'token_type_ids': types})
    logit = float(out[0][0][0])
    print(f"Chunk {cid}: Cross-Encoder Logit against Global Question = {logit:.3f}")
