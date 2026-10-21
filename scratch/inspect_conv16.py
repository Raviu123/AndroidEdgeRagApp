import sqlite3
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
conn = sqlite3.connect('scratch/live_db/offline_docai.db')
c = conn.cursor()

print("=== Conversation 16 Full Inspection ===")
msgs = c.execute("SELECT id, role, content, sources, metrics FROM messages WHERE conversationId = 16 ORDER BY id ASC").fetchall()
print(f"Total messages in Conv 16: {len(msgs)}")

pairs = []
curr_u = None
for m in msgs:
    if m[1] == 'user':
        curr_u = m
    elif m[1] == 'assistant' and curr_u:
        pairs.append((curr_u, m))
        curr_u = None

for i, (u, a) in enumerate(pairs):
    m_info = json.loads(a[4]) if a[4] else {}
    s_info = json.loads(a[3]) if a[3] else []
    print(f"\n[Q{i+1}] {u[2][:80]}...")
    print(f"  Ans: {a[2][:120]}...")
    print(f"  Budget: {m_info.get('budgetTokens')} | Assembled: {m_info.get('assembledTokens')} | Candidates: {m_info.get('candidateChunksCount')}")
    print(f"  Chunks: {[s.get('chunkIds') for s in s_info]}")
    print(f"  Pages: {[s.get('page') for s in s_info]}")
