import sqlite3
import math
import sys

sys.stdout.reconfigure(encoding='utf-8')
conn = sqlite3.connect('scratch/live_db/offline_docai.db')
c = conn.cursor()

q1 = "Did Adam Smith send a message to Li Hua about the upcoming building maintenance schedule before the administrators announced a temporary change in the construction schedule due to weather conditions?"

print(f"Analyzing Question 1: {q1}\n")

# Check all chunks containing 'construction'
print("Chunks with 'construction':")
for r in c.execute("SELECT id, pageNumber, text FROM chunks WHERE text LIKE '%construction%'").fetchall():
    print(f"  Chunk {r[0]} (Page {r[1]}): {r[2][:120].strip()}...")

# Target chunk for Event 2: Chunk 1538 or 1539
target_chunk = c.execute("SELECT id, pageNumber, text FROM chunks WHERE id = 1538").fetchone()
print(f"\nTarget Chunk 1538 (Page {target_chunk[1]}):\n{target_chunk[2]}\n")

target_chunk_39 = c.execute("SELECT id, pageNumber, text FROM chunks WHERE id = 1539").fetchone()
print(f"Target Chunk 1539 (Page {target_chunk_39[1]}):\n{target_chunk_39[2]}\n")

# Now check terms of Q1
import re
def tokenize(s):
    return re.findall(r'[a-zA-Z0-9]+', s.lower())

q_terms = set(tokenize(q1))
print("Q1 terms:", q_terms)

chunk_38_terms = set(tokenize(target_chunk[2]))
chunk_39_terms = set(tokenize(target_chunk_39[2]))

print("Common terms with Chunk 1538:", q_terms.intersection(chunk_38_terms))
print("Common terms with Chunk 1539:", q_terms.intersection(chunk_39_terms))

# Check how many chunks in entire DB match 'adam' or 'smith' or 'maintenance' vs 'construction' or 'weather'
n_total = c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
print(f"\nTotal chunks in DB: {n_total}")

print("Chunks with 'adam':", c.execute("SELECT COUNT(*) FROM chunks WHERE text LIKE '%adam%'").fetchone()[0])
print("Chunks with 'maintenance':", c.execute("SELECT COUNT(*) FROM chunks WHERE text LIKE '%maintenance%'").fetchone()[0])
print("Chunks with 'schedule':", c.execute("SELECT COUNT(*) FROM chunks WHERE text LIKE '%schedule%'").fetchone()[0])
print("Chunks with 'construction':", c.execute("SELECT COUNT(*) FROM chunks WHERE text LIKE '%construction%'").fetchone()[0])
print("Chunks with 'weather':", c.execute("SELECT COUNT(*) FROM chunks WHERE text LIKE '%weather%'").fetchone()[0])
