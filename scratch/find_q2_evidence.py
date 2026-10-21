import sqlite3
import sys

sys.stdout.reconfigure(encoding='utf-8')
conn = sqlite3.connect('scratch/live_db/offline_docai.db')
c = conn.cursor()

print("--- Searching for Q2 Evidence ---")
res_sw = c.execute("SELECT id, pageNumber, sectionTitle, text FROM chunks WHERE text LIKE '%Star Wars%'").fetchall()
for r in res_sw:
    print(f"Star Wars: Chunk {r[0]} | Page {r[1]} | Section {r[2]}")
    print(f"Snippet: {r[3][:180]}...\n")

res_ow = c.execute("SELECT id, pageNumber, sectionTitle, text FROM chunks WHERE text LIKE '%Overwatch%'").fetchall()
for r in res_ow:
    print(f"Overwatch: Chunk {r[0]} | Page {r[1]} | Section {r[2]}")
    print(f"Snippet: {r[3][:180]}...\n")
