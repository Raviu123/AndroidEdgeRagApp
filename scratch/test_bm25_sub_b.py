import sys
sys.stdout.reconfigure(encoding='utf-8')
import sqlite3
import math

conn = sqlite3.connect('offline_docai_device.db')
c = conn.cursor()

terms_b = ['administrators', 'announced', 'temporary', 'change', 'construction', 'schedule', 'weather', 'conditions']
c.execute('SELECT id, pageNumber, text FROM chunks;')
rows = c.fetchall()
n = len(rows)

def clean_words(text):
    return [w.lower().strip('.,!?:;"()[]{}') for w in text.split()]

avglen = sum(len(clean_words(r[2])) for r in rows) / n
df = {}
for t in terms_b:
    df[t] = sum(1 for r in rows if t in clean_words(r[2]))

print("Document frequencies:", df)

def score_bm25(text):
    words = clean_words(text)
    tf = {}
    for w in words:
        tf[w] = tf.get(w, 0) + 1
    score = 0.0
    for t in terms_b:
        if t not in tf:
            continue
        d = df[t]
        if d == 0:
            continue
        idf = max(0.0, math.log(1.0 + (n - d + 0.5) / (d + 0.5)))
        qw = idf * idf
        cw = (tf[t] * 2.2) / (tf[t] + 1.2 * (0.25 + 0.75 * len(words) / avglen))
        score += qw * cw
    return score

scores = [(score_bm25(r[2]), r[0], r[1], r[2][:70].replace('\n', ' ')) for r in rows]
scores.sort(key=lambda x: x[0], reverse=True)

print("\nTop 10 BM25 for Sub-Query B:")
for s in scores[:10]:
    print(f"  Score {s[0]:.2f}: Chunk {s[1]} (p.{s[2]}) -> {s[3]}")

rank_1538 = [i+1 for i, s in enumerate(scores) if s[1] == 1538]
rank_1539 = [i+1 for i, s in enumerate(scores) if s[1] == 1539]
print("\nRank of Chunk 1538 in BM25:", rank_1538)
print("Rank of Chunk 1539 in BM25:", rank_1539)
