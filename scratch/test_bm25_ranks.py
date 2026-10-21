import sqlite3
import math
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
conn = sqlite3.connect('scratch/live_db/offline_docai.db')
c = conn.cursor()

def tokenize(text):
    return re.findall(r'[a-zA-Z0-9]+', text.lower())

all_chunks = c.execute("SELECT id, pageNumber, text FROM chunks").fetchall()
N = len(all_chunks)

# Build DF
doc_freq = {}
chunk_tokens = {}
for cid, page, text in all_chunks:
    toks = tokenize(text)
    chunk_tokens[cid] = toks
    for term in set(toks):
        doc_freq[term] = doc_freq.get(term, 0) + 1

avg_len = sum(len(t) for t in chunk_tokens.values()) / N

def bm25_score(cid, query_terms):
    toks = chunk_tokens[cid]
    l = len(toks)
    tf_map = {}
    for t in toks:
        tf_map[t] = tf_map.get(t, 0) + 1
    
    score = 0.0
    for term in query_terms:
        tf = tf_map.get(term, 0)
        df = doc_freq.get(term, 0)
        if tf == 0 or df == 0:
            continue
        idf = max(0.0, math.log(1.0 + (N - df + 0.5) / (df + 0.5)))
        qw = idf * idf
        tw = (tf * 2.2) / (tf + 1.2 * (1 - 0.75 + 0.75 * l / avg_len))
        score += qw * tw
    return score

q1 = "Did Adam Smith send a message to Li Hua about the upcoming building maintenance schedule before the administrators announced a temporary change in the construction schedule due to weather conditions?"
side_a = "Adam Smith send a message to Li Hua about the upcoming building maintenance schedule"
side_b = "the administrators announced a temporary change in the construction schedule due to weather conditions"

for name, q in [("FULL QUERY", q1), ("SIDE A", side_a), ("SIDE B", side_b)]:
    terms = [t for t in tokenize(q) if t not in {'did', 'a', 'to', 'the', 'about', 'in', 'due', 'before'}]
    scores = [(cid, bm25_score(cid, terms)) for cid, _, _ in all_chunks]
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Find rank of 1538 and 1539
    rank_38 = next((idx+1 for idx, (cid, s) in enumerate(scores) if cid == 1538), -1)
    rank_39 = next((idx+1 for idx, (cid, s) in enumerate(scores) if cid == 1539), -1)
    
    print(f"=== {name} (BM25) ===")
    print(f"Terms: {terms}")
    print(f"Chunk 1538 rank: {rank_38} (score: {dict(scores).get(1538):.3f})")
    print(f"Chunk 1539 rank: {rank_39} (score: {dict(scores).get(1539):.3f})")
    print("Top 5 chunks:")
    for cid, s in scores[:5]:
        print(f"  Rank #{scores.index((cid, s))+1}: Chunk {cid} (score {s:.3f})")
    print()
