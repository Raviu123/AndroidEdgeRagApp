import sqlite3
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
conn = sqlite3.connect('offline_docai_device.db')
c = conn.cursor()

c.execute('SELECT id, role, content, sources, metrics FROM messages WHERE conversationId = 17 ORDER BY id ASC;')
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

print(f"Total turns: {len(turns)}\n")
for i, (uid, u_text, aid, a_text, s_list, m_obj) in enumerate(turns):
    ans_first_sentence = a_text.split('.')[0] if '.' in a_text else a_text[:50]
    pages = sorted(list(set(s.get('page') for s in s_list if s.get('page') is not None)))
    cids = []
    for s in s_list:
        cids.extend(s.get('chunkIds', []))
    print(f"Q{i+1}: {u_text}")
    print(f"   Model Ans: {ans_first_sentence}")
    print(f"   Pages ({len(pages)}): {pages}")
    print(f"   Chunks ({len(cids)}): {cids[:10]}...")
    print(f"   Metrics: tokens={m_obj.get('assembledTokens')}, thermal={m_obj.get('thermal')}, speedCap={m_obj.get('budgetSpeedCapTokens')}")
    print("-" * 60)
