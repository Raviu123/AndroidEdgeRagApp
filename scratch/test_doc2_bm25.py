import sqlite3
import math

conn = sqlite3.connect('offline_docai_device.db')
c = conn.cursor()
c.execute('SELECT id, chunkIndex, pageNumber, text FROM chunks WHERE documentId = 2;')
rows = c.fetchall()
n = len(rows)

def clean_words(text):
    return [w.lower().strip('.,!?:;"()[]{}') for w in text.split()]

avglen = sum(len(clean_words(r[3])) for r in rows) / n

def get_bm25_ranks(query_str):
    q_terms = [w.lower().strip('.,!?:;"()[]{}') for w in query_str.split()]
    df = {}
    for t in q_terms:
        df[t] = sum(1 for r in rows if t in clean_words(r[3]))
    scores = []
    for r in rows:
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
        scores.append((s, r[0], r[1], r[2]))
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores

main_q = 'Did Adam Smith send a message to Li Hua about the upcoming building maintenance schedule before the administrators announced a temporary change in the construction schedule due to weather conditions?'
sub_a = 'Adam Smith send a message to Li Hua about the upcoming building maintenance schedule'
sub_b = 'administrators announced a temporary change in the construction schedule due to weather conditions'

ranks_m = get_bm25_ranks(main_q)
ranks_a = get_bm25_ranks(sub_a)
ranks_b = get_bm25_ranks(sub_b)

rrf = {}
for q_ranks in [ranks_m, ranks_a, ranks_b]:
    for r, (score, cid, cidx, page) in enumerate(q_ranks[:50]):
        if score > 0:
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (60.0 + r + 1)

sorted_rrf = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
top28 = [x[0] for x in sorted_rrf[:28]]
print('In docId=2 only:')
print('Is 1538 in top 28?', 1538 in top28, 'Rank:', [i+1 for i, x in enumerate(sorted_rrf) if x[0] == 1538])
print('Is 1539 in top 28?', 1539 in top28, 'Rank:', [i+1 for i, x in enumerate(sorted_rrf) if x[0] == 1539])
