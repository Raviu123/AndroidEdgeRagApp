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

q = 'Did Adam Smith send a message to Li Hua about the upcoming building maintenance schedule before the administrators announced a temporary change in the construction schedule due to weather conditions?'

for cid in [510, 553, 690, 1538, 1539]:
    c.execute('SELECT sectionTitle, text FROM chunks WHERE id = ?;', (cid,))
    sec, txt = c.fetchone()
    content = (sec + '\n' + txt) if sec else txt
    ids, mask, types = tokenize_pair(q, content)
    out = session_rerank.run(None, {'input_ids': ids, 'attention_mask': mask, 'token_type_ids': types})
    logit = float(out[0][0][0])
    print(f"Chunk {cid}: logit = {logit:.2f}")
