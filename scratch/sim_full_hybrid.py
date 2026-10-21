import onnxruntime as ort
import numpy as np
import sqlite3
import math

with open('vocab.txt', 'r', encoding='utf-8') as f:
    vocab = {line.strip(): i for i, line in enumerate(f)}

def tokenize(text, max_len=128):
    words = ['[CLS]'] + text.lower().split() + ['[SEP]']
    input_ids = [vocab.get(w, vocab.get('[UNK]', 100)) for w in words][:max_len]
    mask = [1] * len(input_ids)
    pad = max_len - len(input_ids)
    input_ids += [0] * pad
    mask += [0] * pad
    types = [0] * max_len
    return np.array([input_ids], dtype=np.int64), np.array([mask], dtype=np.int64), np.array([types], dtype=np.int64)

session_embed = ort.InferenceSession('all-MiniLM-L6-v2.onnx')

def embed(text):
    ids, mask, types = tokenize(text)
    out = session_embed.run(None, {'input_ids': ids, 'attention_mask': mask, 'token_type_ids': types})
    token_embs = out[0][0]
    m = mask[0][:, None]
    sum_embs = np.sum(token_embs * m, axis=0)
    sum_mask = np.sum(m, axis=0)
    emb = sum_embs / sum_mask
    return emb / np.linalg.norm(emb)

conn = sqlite3.connect('offline_docai_device.db')
c = conn.cursor()
c.execute('SELECT id, chunkIndex, pageNumber, text, embedding FROM chunks WHERE documentId = 2;')
rows = c.fetchall()
n = len(rows)

chunk_ids = [r[0] for r in rows]
chunk_embs = [np.frombuffer(r[4], dtype='>f4') for r in rows]

def clean_words(text):
    return [w.lower().strip('.,!?:;"()[]{}') for w in text.split()]

avglen = sum(len(clean_words(r[3])) for r in rows) / n

def run_hybrid_search(queries, topK=28):
    fused = {}
    DENSE_K = 50
    BM25_K = 50
    RRF_K = 60.0
    MIN_DENSE = 0.20

    for q_text, q_emb in queries:
        # Dense
        dense_scores = [float(np.dot(q_emb, c_emb)) for c_emb in chunk_embs]
        dense_ranking = [i for i in range(n) if dense_scores[i] >= MIN_DENSE]
        dense_ranking.sort(key=lambda i: dense_scores[i], reverse=True)
        dense_ranking = dense_ranking[:DENSE_K]

        # BM25
        q_terms = [w.lower().strip('.,!?:;"()[]{}') for w in q_text.split()]
        df = {}
        for t in q_terms:
            df[t] = sum(1 for r in rows if t in clean_words(r[3]))
        
        bm25_scores = []
        for i, r in enumerate(rows):
            words = clean_words(r[3])
            tf = {}
            for w in words: tf[w] = tf.get(w, 0) + 1
            s = 0.0
            for t in q_terms:
                if t not in tf: continue
                d = df[t]
                if d == 0: continue
                idf = max(0.0, math.log(1.0 + (n - d + 0.5) / (d + 0.5)))
                s += (idf * idf) * (tf[t] * 2.2) / (tf[t] + 1.2 * (0.25 + 0.75 * len(words) / avglen))
            bm25_scores.append(s)

        bm25_ranking = [i for i in range(n) if bm25_scores[i] > 0]
        bm25_ranking.sort(key=lambda i: bm25_scores[i], reverse=True)
        bm25_ranking = bm25_ranking[:BM25_K]

        for rank, i in enumerate(dense_ranking):
            cid = chunk_ids[i]
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, i in enumerate(bm25_ranking):
            cid = chunk_ids[i]
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

    sorted_fused = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return sorted_fused[:topK]

main_q = 'Did Adam Smith send a message to Li Hua about the upcoming building maintenance schedule before the administrators announced a temporary change in the construction schedule due to weather conditions?'
sub_a = 'Adam Smith send a message to Li Hua about the upcoming building maintenance schedule'
sub_b = 'administrators announced a temporary change in the construction schedule due to weather conditions'

queries = [(main_q, embed(main_q)), (sub_a, embed(sub_a)), (sub_b, embed(sub_b))]
top_fused = run_hybrid_search(queries, topK=28)

print("Top 28 Fused Chunks:")
for rank, (cid, score) in enumerate(top_fused):
    print(f"  #{rank+1}: Chunk {cid} (RRF: {score:.4f})")

print("\nWhere is Chunk 1538?", [rank+1 for rank, (cid, score) in enumerate(top_fused) if cid == 1538])
print("Where is Chunk 1539?", [rank+1 for rank, (cid, score) in enumerate(top_fused) if cid == 1539])
print("Where is Chunk 553?", [rank+1 for rank, (cid, score) in enumerate(top_fused) if cid == 553])
