import sys
import re
sys.path.append('.')
from scratch.test_retrieval_suite import run_retrieval, conn, split_clauses

q1 = 'Did Adam Smith send a message to Li Hua about the upcoming building maintenance schedule before the administrators announced a temporary change in the construction schedule due to weather conditions?'

sideA, sideB = split_clauses(q1)
subQueries = [sideA, sideB]
kept_new = run_retrieval(q1, floor_val=-10.0)

# Simulate ContextAssembler packing
c = conn.cursor()
c.execute("SELECT id, text FROM chunks WHERE id IN (" + ",".join(str(x[0]) for x in kept_new) + ");")
text_map = {r[0]: r[1] for r in c.fetchall()}

budgetTokens = 2000
selected = []
used = 0
headerCost = 15

# Subquery fair share target
perSubQueryTarget = 2
for sq in subQueries:
    sq_terms = set(re.findall(r'[a-zA-Z]{3,}', sq.lower()))
    matches = []
    for cand in kept_new:
        cid = cand[0]
        if cid in selected: continue
        c_terms = set(re.findall(r'[a-zA-Z]{3,}', text_map[cid].lower()))
        overlap = len(sq_terms.intersection(c_terms))
        if overlap > 0:
            matches.append((cand, overlap))
    matches.sort(key=lambda x: (x[1], x[0][3]), reverse=True)
    count = 0
    for cand, overlap in matches:
        if count >= perSubQueryTarget: break
        cost = len(text_map[cand[0]].split()) * 1.3 + headerCost
        if used + cost <= budgetTokens:
            selected.append(cand[0])
            used += cost
            count += 1

# Fill remaining budget
for cand in kept_new:
    cid = cand[0]
    if cid in selected: continue
    cost = len(text_map[cid].split()) * 1.3 + headerCost
    if used + cost <= budgetTokens:
        selected.append(cid)
        used += cost

print(f"Total packed into prompt: {len(selected)} chunks (used tokens: ~{int(used)}/{budgetTokens})")
print("Selected chunk IDs:", selected)
print("Is Chunk 553 (Side A) in selected?", 553 in selected)
print("Is Chunk 1538 (Side B) in selected?", 1538 in selected)
print("Is Chunk 1539 (Side B) in selected?", 1539 in selected)
