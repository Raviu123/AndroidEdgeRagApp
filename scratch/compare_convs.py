import sys
import sqlite3

sys.stdout.reconfigure(encoding='utf-8')
conn = sqlite3.connect('scratch/live_db/offline_docai.db')
c = conn.cursor()

print("=== CONV 11 (Bad run) ===")
rows = c.execute("SELECT id, role, content, sources, metrics FROM messages WHERE conversationId = 11 AND role = 'assistant' LIMIT 6").fetchall()
for r in rows:
    print(f"ID {r[0]}: Content: {r[2]}")
    print(f"Sources snippet: {str(r[3])[:150]}")
    print(f"Metrics: {r[4]}")
    print("-" * 50)

print("\n=== CONV 12 ===")
rows12 = c.execute("SELECT id, role, content, sources, metrics FROM messages WHERE conversationId = 12 AND role = 'assistant' LIMIT 6").fetchall()
for r in rows12:
    print(f"ID {r[0]}: Content: {r[2]}")
    print(f"Sources snippet: {str(r[3])[:150]}")
    print(f"Metrics: {r[4]}")
    print("-" * 50)
