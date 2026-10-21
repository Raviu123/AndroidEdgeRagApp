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
for idx, (uid, u_text, aid, a_text, s_list, m_obj) in enumerate(turns):
    ans_start = a_text.split('.')[0] if '.' in a_text else a_text[:50]
    pages = [s.get('page') for s in s_list]
    print(f"Turn {idx+1} [User {uid}]: {u_text}")
    print(f"  Response: {ans_start}")
    print(f"  Pages ({len(s_list)}): {pages}")
    print(f"  Metrics: assembledTokens={m_obj.get('assembledTokens')}, thermal={m_obj.get('thermal')}, speedCap={m_obj.get('budgetSpeedCapTokens')}")
    print("-" * 60)
