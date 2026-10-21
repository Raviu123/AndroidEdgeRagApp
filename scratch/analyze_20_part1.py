import sqlite3
import json
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
conn = sqlite3.connect('scratch/live_db/offline_docai.db')
c = conn.cursor()

def search_text(keywords, exclude=None):
    query = "SELECT id, pageNumber, sectionTitle, text FROM chunks WHERE "
    clauses = ["text LIKE ?" for _ in keywords]
    query += " AND ".join(clauses)
    params = [f"%{k}%" for k in keywords]
    rows = c.execute(query, params).fetchall()
    return rows

print("=== Analyzing All 20 Questions in the PDF Database ===")

# Q1: Adam Smith building maintenance schedule vs administrators temporary change in construction schedule
print("\n--- Q1 ---")
r1_a = search_text(["AdamSmith", "maintenance schedule"])
print(f"Event A matches: {len(r1_a)}")
for r in r1_a: print(f"  Chunk {r[0]}, Page {r[1]}, Sec {r[2]}: {r[3][:120].strip()}...")
r1_b = search_text(["construction", "postponed"])
print(f"Event B matches: {len(r1_b)}")
for r in r1_b: print(f"  Chunk {r[0]}, Page {r[1]}, Sec {r[2]}: {r[3][:120].strip()}...")

# Q2: Wolfgang & Star Wars vs Overwatch 3
print("\n--- Q2 ---")
r2_a = search_text(["Overwatch 3"])
for r in r2_a: print(f"  Overwatch 3: Chunk {r[0]}, Page {r[1]}: {r[3][:120].strip()}...")
r2_b = search_text(["Star Wars: A New Hope"])
for r in r2_b: print(f"  Star Wars: Chunk {r[0]}, Page {r[1]}: {r[3][:120].strip()}...")

# Q3: Wolfgang first asked dinner vs Li Hua agree
print("\n--- Q3 ---")
r3_a = search_text(["grabbing some dinner tonight"])
for r in r3_a: print(f"  First ask: Chunk {r[0]}, Page {r[1]}: {r[3][:150].strip()}...")
r3_b = search_text(["Saturday night works for me! Looking forward to it"])
for r in r3_b: print(f"  Later agree: Chunk {r[0]}, Page {r[1]}: {r[3][:150].strip()}...")

# Q4: Jennifer thanking schedule vs requested change for Thursday
print("\n--- Q4 ---")
r4_a = search_text(["Jennifer", "new training schedule"])
for r in r4_a: print(f"  New schedule: Chunk {r[0]}, Page {r[1]}: {r[3][:120].strip()}...")
r4_b = search_text(["Jennifer", "Thursday"])
print(f"  Jennifer + Thursday matches: {len(r4_b)}")
for r in r4_b[:3]: print(f"  Chunk {r[0]}, Page {r[1]}: {r[3][:120].strip()}...")

# Q5: Soreness advice vs soreness report in arm
print("\n--- Q5 ---")
r5_a = search_text(["muscle soreness", "tips"])
for r in r5_a: print(f"  Tips: Chunk {r[0]}, Page {r[1]}: {r[3][:120].strip()}...")
r5_b = search_text(["arm", "bench press"])
for r in r5_b: print(f"  Arm: Chunk {r[0]}, Page {r[1]}: {r[3][:120].strip()}...")

# Q6: Protein supplements opinion vs daily protein powder consumption
print("\n--- Q6 ---")
r6_a = search_text(["protein supplements"])
for r in r6_a: print(f"  Supplements: Chunk {r[0]}, Page {r[1]}: {r[3][:120].strip()}...")
r6_b = search_text(["protein powder"])
for r in r6_b: print(f"  Powder: Chunk {r[0]}, Page {r[1]}: {r[3][:120].strip()}...")
