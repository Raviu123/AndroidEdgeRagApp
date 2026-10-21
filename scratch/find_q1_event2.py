import sqlite3
import sys

sys.stdout.reconfigure(encoding='utf-8')
conn = sqlite3.connect('scratch/live_db/offline_docai.db')
c = conn.cursor()

print("--- Searching for Event 2 of Q1 ---")
results = c.execute("SELECT id, pageNumber, sectionTitle, text FROM chunks WHERE text LIKE '%construction%' AND (text LIKE '%weather%' OR text LIKE '%change%' OR text LIKE '%schedule%')").fetchall()
for r in results:
    print(f"Chunk ID: {r[0]} | Page: {r[1]} | Section: {r[2]}")
    print(f"Text snippet: {r[3][:250]}...")
    print("-" * 50)
