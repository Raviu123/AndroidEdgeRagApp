import sqlite3
import json
import re

# Load verified dataset
with open('benchmarks/pdf_for_testing/multy-qset-20.json', 'r', encoding='utf-8') as f:
    qset = json.load(f)

conn = sqlite3.connect('offline_docai_device.db')
c = conn.cursor()

c.execute("SELECT id, role, content, sources, metrics FROM messages WHERE conversationId = 17 ORDER BY id ASC;")
rows = c.fetchall()

turns = []
cur_user = None
for r in rows:
    msg_id, role, content, sources, metrics = r
    if role == 'user':
        cur_user = (msg_id, content)
    elif role == 'assistant' and cur_user:
        turns.append((cur_user[1], json.loads(sources) if sources else []))
        cur_user = None

print("=== CONV 17 RETRIEVAL ACCURACY ===")
both_found = 0
one_found = 0
none_found = 0

for i, (q_text, sources) in enumerate(turns):
    item = qset[i]
    c_a = item['event_a']['chunk_index']
    c_b = item['event_b']['chunk_index']
    
    # Get all chunkIndices in retrieved sources
    retrieved_cids = []
    for s in sources:
        retrieved_cids.extend(s.get('chunkIds', []))
        
    retrieved_indices = set()
    if retrieved_cids:
        placeholders = ','.join('?' for _ in retrieved_cids)
        c.execute(f"SELECT chunkIndex FROM chunks WHERE id IN ({placeholders})", retrieved_cids)
        retrieved_indices = set(r[0] for r in c.fetchall())
        
    has_a = (c_a in retrieved_indices) if c_a else True
    has_b = (c_b in retrieved_indices) if c_b else True
    
    if has_a and has_b:
        status = "BOTH"
        both_found += 1
    elif has_a or has_b:
        status = "ONE "
        one_found += 1
    else:
        status = "NONE"
        none_found += 1
        
    print(f"Q{i+1:02d} [{status}] Ev-A ({c_a}): {'YES' if has_a else 'NO '} | Ev-B ({c_b}): {'YES' if has_b else 'NO '} | Chunks: {len(retrieved_cids)} | {q_text[:50]}...")

print("\n" + "="*50)
print(f"BOTH EVENTS RETRIEVED: {both_found} / {len(turns)} ({(both_found/len(turns))*100:.1f}%)")
print(f"ONLY ONE EVENT RETRIEVED: {one_found} / {len(turns)} ({(one_found/len(turns))*100:.1f}%)")
print(f"ZERO EVENTS RETRIEVED: {none_found} / {len(turns)} ({(none_found/len(turns))*100:.1f}%)")
print("="*50)
