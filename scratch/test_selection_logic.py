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

# All candidate chunk IDs from the reranked / fused list of Q1
candidate_ids = [690, 553, 510, 996, 803, 1003, 1309, 1598, 1789, 1538, 1539, 2165, 2329]

chunks_data = []
for cid in candidate_ids:
    r = c.execute("SELECT id, pageNumber, sectionTitle, text FROM chunks WHERE id = ?", (cid,)).fetchone()
    if r:
        chunks_data.append({'id': r[0], 'page': r[1], 'section': r[2], 'text': r[3]})

# Simulate subquery fair-share selection
selected = []
selected_ids = set()

# For SubQuery A
matches_a = []
for cd in chunks_data:
    cterms = tokenize(cd['text'])
    ov = len(terms_a.intersection(cterms))
    if ov > 0:
        matches_a.append((cd, ov))
matches_a.sort(key=lambda x: x[1], reverse=True)

print("--- Side A Candidates by Overlap ---")
for cd, ov in matches_a[:4]:
    print(f"  Chunk {cd['id']} (Page {cd['page']}): overlap {ov} {terms_a.intersection(tokenize(cd['text']))}")
    if cd['id'] not in selected_ids and len([x for x in selected if x in matches_a]) < 2:
        selected.append(cd)
        selected_ids.add(cd['id'])

# For SubQuery B
matches_b = []
for cd in chunks_data:
    cterms = tokenize(cd['text'])
    ov = len(terms_b.intersection(cterms))
    if ov > 0:
        matches_b.append((cd, ov))
matches_b.sort(key=lambda x: x[1], reverse=True)

print("\n--- Side B Candidates by Overlap ---")
for cd, ov in matches_b[:4]:
    print(f"  Chunk {cd['id']} (Page {cd['page']}): overlap {ov} {terms_b.intersection(tokenize(cd['text']))}")
    if cd['id'] not in selected_ids and len([x for x in selected if x in matches_b]) < 2:
        selected.append(cd)
        selected_ids.add(cd['id'])

print("\n--- Final Selected Chunks for Prompt ---")
for cd in selected:
    print(f"  Chunk {cd['id']} (Page {cd['page']}, Sec: {cd['section']}): {cd['text'][:100].strip()}...")
