import sqlite3
import json
import math

conn = sqlite3.connect('offline_docai_device.db')
c = conn.cursor()

# In Msg 739:
c.execute("SELECT sources FROM messages WHERE id = 739;")
sources = json.loads(c.fetchone()[0])

print("Chunks in Msg 739 and their saved sigmoid scores:")
for s in sources:
    score = s.get('score', 0)
    logit = math.log(score / (1.0 - score + 1e-7)) if 0 < score < 1 else -99
    cids = s.get('chunkIds', [])
    print(f"  Chunk {cids}: score={score:.4f}, logit={logit:.2f}")

# Remember in VectorStore.kt line 127:
# r.copy(rerankScore = scores[i], relevanceScore = (1.0 / (1.0 + exp(-scores[i]))).toFloat())
# So s['score'] IS the sigmoid of the reranker logit!
