import sqlite3
import struct

conn = sqlite3.connect('scratch/live_db/offline_docai.db')
c = conn.cursor()

r = c.execute("SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL LIMIT 1").fetchone()
if r:
    blob = r[1]
    floats = struct.unpack(f"{len(blob)//4}f", blob)
    print(f"Chunk {r[0]} has {len(floats)} embedding floats")
else:
    print("No embeddings found!")
