import sqlite3
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
conn = sqlite3.connect('offline_docai_conv21.db')
c = conn.cursor()

c.execute("SELECT id, role, content, sources, metrics FROM messages WHERE conversationId = 21 ORDER BY id ASC;")
rows = c.fetchall()

turns = []
cur_user = None
for r in rows:
    msg_id, role, content, sources, metrics = r
    if role == 'user':
        cur_user = (msg_id, content)
    elif role == 'assistant' and cur_user:
        turns.append((cur_user[0], cur_user[1], msg_id, content, json.loads(sources) if sources else [], json.loads(metrics) if metrics else {}))
        cur_user = None

print(f"Total turns in Conv 21: {len(turns)}\n")
for i, (uid, utext, aid, atext, sources, metrics) in enumerate(turns):
    ans_start = atext.split('.')[0] if '.' in atext else atext[:50]
    pages = sorted(list(set(s.get('page') for s in sources if s.get('page') is not None)))
    cids = []
    for s in sources: cids.extend(s.get('chunkIds', []))
    print(f"Q{i+1}: {utext[:65]}...")
    print(f"   Model Ans: {ans_start}")
    print(f"   Pages ({len(pages)}): {pages}")
    print(f"   Chunks ({len(cids)}): {cids}")
    print(f"   Metrics: tokens={metrics.get('assembledTokens')} | cands={metrics.get('candidateChunksCount')} | thermal={metrics.get('thermal')} | speedCap={metrics.get('budgetSpeedCapTokens')}")
    print("-" * 60)
