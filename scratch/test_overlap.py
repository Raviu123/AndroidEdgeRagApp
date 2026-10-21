import sqlite3
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
conn = sqlite3.connect('scratch/live_db/offline_docai.db')
c = conn.cursor()

def tokenize(s):
    return set(re.findall(r'[a-zA-Z0-9]+', s.lower())) - {'did', 'a', 'to', 'the', 'about', 'in', 'due', 'before', 'after', 'and', 'or', 'of', 'for', 'with', 'on', 'at', 'from'}

side_a = "Adam Smith send a message to Li Hua about the upcoming building maintenance schedule"
side_b = "the administrators announced a temporary change in the construction schedule due to weather conditions"

terms_a = tokenize(side_a)
terms_b = tokenize(side_b)

print("Terms A:", terms_a)
print("Terms B:", terms_b)

candidate_ids = [690, 553, 510, 996, 803, 1003, 1309, 1598, 1789, 1538, 1539]

print("\n--- Overlap Scores ---")
for cid in candidate_ids:
    r = c.execute("SELECT id, pageNumber, text FROM chunks WHERE id = ?", (cid,)).fetchone()
    if not r: continue
    c_terms = tokenize(r[2])
    score_a = len(terms_a.intersection(c_terms))
    score_b = len(terms_b.intersection(c_terms))
    print(f"Chunk {cid} (Page {r[1]}): Side A score = {score_a} {terms_a.intersection(c_terms)}, Side B score = {score_b} {terms_b.intersection(c_terms)}")
