import sqlite3
import numpy as np

conn = sqlite3.connect('offline_docai_device.db')
c = conn.cursor()

# Load all chunks
c.execute("SELECT id, text, embedding FROM chunks WHERE embedding IS NOT NULL;")
rows = c.fetchall()

print(f"Total indexed chunks: {len(rows)}")

# Query Q1
# Let's see what happens during hybrid search
