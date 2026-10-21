import sqlite3
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
conn = sqlite3.connect('scratch/live_db/offline_docai.db')
c = conn.cursor()

def search_text(keywords):
    query = "SELECT id, pageNumber, sectionTitle, text FROM chunks WHERE "
    clauses = ["text LIKE ?" for _ in keywords]
    query += " AND ".join(clauses)
    params = [f"%{k}%" for k in keywords]
    return c.execute(query, params).fetchall()

print("=== Analyzing Questions 7 to 20 ===")

# Q7: Yuriko homepage help vs Central Perk seat booking
print("\n--- Q7 ---")
for r in search_text(["Yuriko", "homepage"]):
    print(f"  Homepage: Chunk {r[0]}, Page {r[1]}, Sec {r[2]}: {r[3][:120].strip()}...")
for r in search_text(["Central Perk"]):
    print(f"  Central Perk: Chunk {r[0]}, Page {r[1]}, Sec {r[2]}: {r[3][:120].strip()}...")

# Q8: Progress with fitness plan vs blog post
print("\n--- Q8 ---")
for r in search_text(["blog post", "fitness"]):
    print(f"  Blog post: Chunk {r[0]}, Page {r[1]}, Sec {r[2]}: {r[3][:120].strip()}...")
for r in search_text(["fitness plan", "progress"]):
    print(f"  Progress: Chunk {r[0]}, Page {r[1]}, Sec {r[2]}: {r[3][:120].strip()}...")

# Q9: Thursday class to Friday vs change request for Thursday
print("\n--- Q9 ---")
for r in search_text(["Thursday", "Friday", "class"]):
    print(f"  Thursday->Friday: Chunk {r[0]}, Page {r[1]}, Sec {r[2]}: {r[3][:120].strip()}...")

# Q10: Yuriko music together vs Wolfgang pause playing
print("\n--- Q10 ---")
for r in search_text(["pause", "instrument"]):
    print(f"  Pause: Chunk {r[0]}, Page {r[1]}, Sec {r[2]}: {r[3][:120].strip()}...")

# Q11: Wolfgang Viva la Vida vs discuss what song to play Sunday
print("\n--- Q11 ---")
for r in search_text(["Viva la Vida"]):
    print(f"  Viva: Chunk {r[0]}, Page {r[1]}, Sec {r[2]}: {r[3][:120].strip()}...")

# Q12: Promotion announcement vs dinner on 20260430
print("\n--- Q12 ---")
for r in search_text(["promoted"]):
    print(f"  Promotion: Chunk {r[0]}, Page {r[1]}, Sec {r[2]}: {r[3][:120].strip()}...")
for r in search_text(["20260430"]):
    print(f"  20260430: Chunk {r[0]}, Page {r[1]}, Sec {r[2]}: {r[3][:120].strip()}...")

# Q13: Turalyon construction updates vs Illidan noise complaint
print("\n--- Q13 ---")
for r in search_text(["Illidan", "noise"]):
    print(f"  Noise: Chunk {r[0]}, Page {r[1]}, Sec {r[2]}: {r[3][:120].strip()}...")

# Q14: Hailey high-protein breads vs bakery event invitation
print("\n--- Q14 ---")
for r in search_text(["Hailey", "bread"]):
    print(f"  Hailey bread: Chunk {r[0]}, Page {r[1]}, Sec {r[2]}: {r[3][:120].strip()}...")

# Q15: Chae warm shower vs neuroscience article
print("\n--- Q15 ---")
for r in search_text(["warm shower"]):
    print(f"  Shower: Chunk {r[0]}, Page {r[1]}, Sec {r[2]}: {r[3][:120].strip()}...")
for r in search_text(["neuroscience"]):
    print(f"  Neuroscience: Chunk {r[0]}, Page {r[1]}, Sec {r[2]}: {r[3][:120].strip()}...")

# Q16: Jennifer book recommendations vs guest speaker at gym
print("\n--- Q16 ---")
for r in search_text(["book recommendation"]):
    print(f"  Book: Chunk {r[0]}, Page {r[1]}, Sec {r[2]}: {r[3][:120].strip()}...")
for r in search_text(["guest speaker"]):
    print(f"  Speaker: Chunk {r[0]}, Page {r[1]}, Sec {r[2]}: {r[3][:120].strip()}...")

# Q17: Jennifer reminder nutrition/hydration vs Jake tips
print("\n--- Q17 ---")
for r in search_text(["Jake", "hydrat"]):
    print(f"  Jake: Chunk {r[0]}, Page {r[1]}, Sec {r[2]}: {r[3][:120].strip()}...")

# Q18: GoT favorite characters vs Emily vote on most hateable character
print("\n--- Q18 ---")
for r in search_text(["hateable"]):
    print(f"  Hateable: Chunk {r[0]}, Page {r[1]}, Sec {r[2]}: {r[3][:120].strip()}...")

# Q19: Thane The Last of Us vs Sekiro
print("\n--- Q19 ---")
for r in search_text(["Last of Us"]):
    print(f"  Last of Us: Chunk {r[0]}, Page {r[1]}, Sec {r[2]}: {r[3][:120].strip()}...")
for r in search_text(["Sekiro"]):
    print(f"  Sekiro: Chunk {r[0]}, Page {r[1]}, Sec {r[2]}: {r[3][:120].strip()}...")

# Q20: Barcelona & Bayern Munich vs Barcelona & Real Madrid
print("\n--- Q20 ---")
for r in search_text(["Bayern Munich"]):
    print(f"  Bayern: Chunk {r[0]}, Page {r[1]}, Sec {r[2]}: {r[3][:120].strip()}...")
for r in search_text(["Real Madrid"]):
    print(f"  Real Madrid: Chunk {r[0]}, Page {r[1]}, Sec {r[2]}: {r[3][:120].strip()}...")
