import sqlite3
import json

conn = sqlite3.connect('offline_docai_device.db')
cursor = conn.cursor()
cursor.execute('SELECT id, role, content, sources, metrics FROM messages WHERE conversationId = 17 ORDER BY id ASC;')
rows = cursor.fetchall()

print(f"Total messages in conv 17: {len(rows)}")
for r in rows:
    msg_id, role, content, sources, metrics = r
    if role == 'user':
        print(f"\n--- USER [{msg_id}]: {content[:75]}...")
    elif role == 'assistant' and sources:
        try:
            s_list = json.loads(sources)
            chunk_ids = [s.get('chunkId', s.get('id', '?')) for s in s_list]
            pages = [s.get('pageNumber', s.get('page', '?')) for s in s_list]
            m_obj = json.loads(metrics) if metrics else {}
            print(f"ASSISTANT [{msg_id}]: Ans: {content[:80]}...")
            print(f"   Chunks ({len(s_list)}): {chunk_ids}")
            print(f"   Pages: {pages}")
            print(f"   Metrics: {m_obj}")
        except Exception as e:
            print(f"ASSISTANT [{msg_id}]: error {e}")
