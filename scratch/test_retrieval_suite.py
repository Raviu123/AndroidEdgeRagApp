import json
import sqlite3
import re
import math
import onnxruntime as ort
import numpy as np

# Load vocab
with open('vocab.txt', 'r', encoding='utf-8') as f:
    vocab = {line.strip(): i for i, line in enumerate(f)}

def tokenize_embed(text, max_len=128):
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
    ids, mask, types = tokenize_embed(text)
    out = session_embed.run(None, {'input_ids': ids, 'attention_mask': mask, 'token_type_ids': types})
    token_embs = out[0][0]
    m = mask[0][:, None]
    sum_embs = np.sum(token_embs * m, axis=0)
    sum_mask = np.sum(m, axis=0)
    emb = sum_embs / sum_mask
    return emb / np.linalg.norm(emb)

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

# Database
conn = sqlite3.connect('offline_docai_device.db')
c = conn.cursor()
c.execute('SELECT id, chunkIndex, pageNumber, text, embedding FROM chunks WHERE documentId = 2;')
rows = c.fetchall()
n = len(rows)

chunk_ids = [r[0] for r in rows]
chunk_indices = [r[1] for r in rows]
chunk_pages = [r[2] for r in rows]
chunk_texts = [r[3] for r in rows]
chunk_embs = [np.frombuffer(r[4], dtype='>f4') for r in rows]

def clean_words(text):
    return [w.lower().strip('.,!?:;"()[]{}') for w in text.split()]

avglen = sum(len(clean_words(t)) for t in chunk_texts) / n

# Precalculate doc freq for all indexed words
print("Building doc frequencies...")
all_df = {}
for t in chunk_texts:
    for w in set(clean_words(t)):
        all_df[w] = all_df.get(w, 0) + 1

LEAD_IN = re.compile(r'^\s*(?:did|was|were|is|are|does|do|has|have|had|can|could|would|should|will)\s+', re.I)
TEMPORAL_CONNECTORS = re.compile(r'\b(before|after|prior\s+to|subsequent\s+to|following|earlier\s+than|later\s+than)\b', re.I)
PRONOUN_SUBJECT = re.compile(r'^(?:he|she|they|it|we|i)\b', re.I)
NAMED_CLAUSE_SUBJECT = re.compile(r'^(?:the\s+[a-z]+|[A-Z][a-zA-Z0-9_-]+)\s+(?:first\s+|also\s+)?(?:[a-z]+ed|announced|asked|told|requested|discussed|shared|proposed|booked|consulted|sent|agreed|did|was|is)\b', re.I)

def cleanEntity(raw):
    s = raw.strip()
    s = re.sub(r'[?.!,;:\"]+$', '', s)
    s = re.sub(r'^(?i:the|a|an)\s+', '', s)
    return s.strip()

def split_clauses(q):
    clean = LEAD_IN.sub('', q.strip()).rstrip('?.! ')
    allMatches = list(TEMPORAL_CONNECTORS.finditer(clean))
    if not allMatches: return None, None
    candidates = []
    for m in allMatches:
        left = clean[:m.start()].strip()
        right = clean[m.end():].strip()
        wordsL = [w for w in left.split() if w]
        wordsR = [w for w in right.split() if w]
        if len(wordsL) < 3 or len(wordsR) < 3: continue
        balance = min(len(wordsL), len(wordsR)) / max(len(wordsL), len(wordsR))
        score = balance * 10.0
        if PRONOUN_SUBJECT.search(right): score += 15.0
        elif NAMED_CLAUSE_SUBJECT.search(right): score += 12.0
        candidates.append((left, right, score))
    if not candidates: return None, None
    best = max(candidates, key=lambda x: x[2])
    return cleanEntity(best[0]), cleanEntity(best[1])

def run_retrieval(question, floor_val=-10.0):
    sideA, sideB = split_clauses(question)
    subQueries = [sideA, sideB] if (sideA and sideB) else []
    
    # 1. Embed queries
    queries = [(question, embed(question))]
    for sq in subQueries:
        queries.append((sq, embed(sq)))
        
    # 2. Hybrid search with topK = 28
    fused = {}
    DENSE_K = 50
    BM25_K = 50
    RRF_K = 60.0
    MIN_DENSE = 0.20

    for q_text, q_emb in queries:
        dense_scores = [float(np.dot(q_emb, c_emb)) for c_emb in chunk_embs]
        dense_ranking = [i for i in range(n) if dense_scores[i] >= MIN_DENSE]
        dense_ranking.sort(key=lambda i: dense_scores[i], reverse=True)
        dense_ranking = dense_ranking[:DENSE_K]

        q_terms = [w.lower().strip('.,!?:;"()[]{}') for w in q_text.split()]
        
        bm25_scores = []
        for i, text in enumerate(chunk_texts):
            words = clean_words(text)
            tf = {}
            for w in words: tf[w] = tf.get(w, 0) + 1
            s = 0.0
            for t in q_terms:
                if t not in tf: continue
                d = all_df.get(t, 0)
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

    top_fused = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:28]
    
    # 3. Rerank
    rerank_pool = []
    for cid, _ in top_fused:
        idx = chunk_ids.index(cid)
        text = chunk_texts[idx]
        ids, mask, types = tokenize_pair(question, text)
        out = session_rerank.run(None, {'input_ids': ids, 'attention_mask': mask, 'token_type_ids': types})
        logit = float(out[0][0][0])
        rerank_pool.append((cid, chunk_indices[idx], chunk_pages[idx], logit))

    rerank_pool.sort(key=lambda x: x[3], reverse=True)
    
    # 4. Filter with floor
    best = rerank_pool[0][3] if rerank_pool else 0.0
    margin = 10.0
    kept = [x for x in rerank_pool if x[3] >= floor_val and x[3] >= (best - margin)]
    return kept

print("Retrieval tester ready.")
