import sqlite3
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
conn = sqlite3.connect('scratch/live_db/offline_docai.db')
c = conn.cursor()

print("=== Conv 15 Messages and Metrics ===")
rows = c.execute("SELECT id, role, content, sources, metrics FROM messages WHERE conversationId = 15 ORDER BY id ASC").fetchall()
for r in rows:
    if r[1] == 'user':
        print(f"\n[USER]: {r[2]}")
    elif r[1] == 'assistant' and r[4]:
        print(f"[ASSISTANT]: {r[2][:200]}...")
        m = json.loads(r[4])
        print(f"  Model: {m.get('modelName')}")
        print(f"  Budget Tokens: {m.get('budgetTokens')} | Assembled Tokens: {m.get('assembledTokens')}")
        print(f"  Candidate Chunks: {m.get('candidateChunksCount')}")
        sources = json.loads(r[3]) if r[3] else []
        print(f"  Retrieved Chunks: {[s.get('chunkIds') for s in sources]}")
        print(f"  Retrieved Pages: {[s.get('page') for s in sources]}")
