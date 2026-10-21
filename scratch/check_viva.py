import sqlite3
import sys

sys.stdout.reconfigure(encoding='utf-8')
conn = sqlite3.connect('scratch/live_db/offline_docai.db')
c = conn.cursor()

for r in c.execute("SELECT id, pageNumber, text FROM chunks WHERE text LIKE '%Viva la Vida%'").fetchall():
    print(f"Chunk {r[0]} (Page {r[1]}):\n{r[2]}\n")
