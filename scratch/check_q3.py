import sqlite3
import sys

sys.stdout.reconfigure(encoding='utf-8')
conn = sqlite3.connect('scratch/live_db/offline_docai.db')
c = conn.cursor()

# Find the first dinner request between Wolfgang and Li Hua
print("--- Checking Dinner Chunks ---")
for r in c.execute("SELECT id, pageNumber, text FROM chunks WHERE text LIKE '%dinner%' AND text LIKE '%Wolfgang%' LIMIT 6").fetchall():
    print(f"Chunk {r[0]} (Page {r[1]}):\n{r[2][:250]}...\n{'-'*40}")
