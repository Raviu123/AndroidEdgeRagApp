import sqlite3
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

conn = sqlite3.connect('scratch/live_db/offline_docai.db')
c = conn.cursor()

# Load ground truth questions
with open('benchmarks/pdf_for_testing/lihuaworld-qset-1.json', 'r', encoding='utf-8') as f:
    gold_data = json.load(f)

gold_dict = {item['question'].strip(): item for item in gold_data}

# Let's inspect Q1, Q2, Q3, Q4 from Conv 13
conv13_msgs = c.execute("SELECT id, role, content, sources FROM messages WHERE conversationId = 13 ORDER BY id ASC").fetchall()

# Pair user questions with assistant answers
pairs = []
curr_user = None
for m in conv13_msgs:
    if m[1] == 'user':
        curr_user = m
    elif m[1] == 'assistant' and curr_user is not None:
        pairs.append((curr_user, m))
        curr_user = None

print(f"Total paired Q&As in Conv 13: {len(pairs)}\n")

for i in range(min(5, len(pairs))):
    u, a = pairs[i]
    q_text = u[2].strip()
    gold = gold_dict.get(q_text, {})
    evidence_keys = gold.get('evidence', '').split('<and>')
    
    print(f"==================================================")
    print(f"QUESTION {i+1}:")
    print(f"  {q_text}")
    print(f"GOLD ANSWER: {gold.get('answer')} | EVIDENCE NEEDED: {evidence_keys}")
    print(f"MODEL ANSWER:\n  {a[2][:250]}...")
    
    # Check what chunks were retrieved
    sources_raw = a[3]
    sources = json.loads(sources_raw) if sources_raw else []
    print(f"\nRETRIEVED {len(sources)} CHUNKS BY RAG:")
    retrieved_chunk_ids = []
    for s_idx, s in enumerate(sources):
        chunk_ids = s.get('chunkIds', [])
        retrieved_chunk_ids.extend(chunk_ids)
        page = s.get('page')
        sec = s.get('section')
        txt = s.get('text', '')
        print(f"  Source #{s_idx+1}: Page {page}, Section: {sec}, Chunks: {chunk_ids}")
        # print first 120 chars of chunk text
        print(f"    Snippet: {txt[:140].strip()}...")
        
    # Now let's search SQLite chunks table for the REAL evidence chunks!
    print(f"\nSEARCHING DATABASE FOR TRUE EVIDENCE CHUNKS:")
    for ev in evidence_keys:
        # Search for timestamp like 20260121_1000 or 2026-01-21 10:00
        # format in doc: e.g. 2026-01-21 10:00 or 20260121_1000.txt
        term1 = ev.replace('_', ' ')  # 20260121 10:00
        term_date = f"{ev[:4]}-{ev[4:6]}-{ev[6:8]}" # 2026-01-21
        term_time = f"{ev[9:11]}:{ev[11:13]}" if len(ev) > 9 else ""
        query_sql = "SELECT id, pageNumber, sectionTitle, text FROM chunks WHERE text LIKE ? AND text LIKE ?"
        matches = c.execute(query_sql, (f"%{term_date}%", f"%{term_time}%")).fetchall()
        print(f"  Evidence '{ev}' (Date: {term_date} {term_time}):")
        if matches:
            for m in matches:
                was_retrieved = m[0] in retrieved_chunk_ids
                status = ">>> YES, RETRIEVED BY RAG! <<<" if was_retrieved else ">>> MISSED! NOT RETRIEVED! <<<"
                print(f"    Found Chunk ID: {m[0]}, Page: {m[1]}, Section: {m[2]} [{status}]")
                print(f"    Excerpt: {m[3][:180].replace(chr(10), ' ')}...")
        else:
            print(f"    No exact timestamp match found with {term_date} {term_time}")
    print()
