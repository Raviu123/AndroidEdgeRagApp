import sqlite3
import json
import re

# Load verified dataset
with open('benchmarks/pdf_for_testing/multy-qset-20.json', 'r', encoding='utf-8') as f:
    verified_data = json.load(f)

# Connect to database
conn = sqlite3.connect('offline_docai_device.db')
cursor = conn.cursor()
cursor.execute('SELECT id, role, content, sources, metrics FROM messages WHERE conversationId = 17 ORDER BY id ASC;')
rows = cursor.fetchall()

# Map Q&A turns
turns = []
cur_user = None
for r in rows:
    msg_id, role, content, sources, metrics = r
    if role == 'user':
        cur_user = (msg_id, content)
    elif role == 'assistant' and cur_user:
        turns.append({
            'user_id': cur_user[0],
            'question': cur_user[1],
            'asst_id': msg_id,
            'response': content,
            'sources': json.loads(sources) if sources else [],
            'metrics': json.loads(metrics) if metrics else {}
        })
        cur_user = None

print(f"Total evaluated turns: {len(turns)}")

correct_count = 0
total_evaluated = 0

for i, turn in enumerate(turns):
    q_text = turn['question']
    resp = turn['response']
    
    # Find matching verified entry
    match = None
    for item in verified_data:
        if item['question'].strip() == q_text.strip() or item['question'][:50] in q_text:
            match = item
            break
            
    # Extract model predicted answer
    pred = "Unknown"
    resp_clean = resp.strip()
    if resp_clean.lower().startswith("yes") or "yes," in resp_clean[:20].lower():
        pred = "Yes"
    elif resp_clean.lower().startswith("no") or "no," in resp_clean[:20].lower():
        pred = "No"
        
    gold = match['ground_truth_answer'] if match else "N/A"
    is_correct = (pred == gold)
    if match:
        total_evaluated += 1
        if is_correct:
            correct_count += 1
            
    # Analyze retrieved chunks vs target chunks
    retrieved_chunk_ids = []
    for s in turn['sources']:
        retrieved_chunk_ids.extend(s.get('chunkIds', []))
        
    chunk_a = match['event_a']['chunk_index'] if match else None
    chunk_b = match['event_b']['chunk_index'] if match else None
    
    # Note: chunk_index vs id. Let's check if chunk_index or chunkId is in chunks
    # Retrieve chunk_indices for retrieved IDs
    if retrieved_chunk_ids:
        placeholders = ','.join('?' for _ in retrieved_chunk_ids)
        cursor.execute(f"SELECT chunkIndex FROM chunks WHERE id IN ({placeholders})", retrieved_chunk_ids)
        retrieved_indices = set(r[0] for r in cursor.fetchall())
    else:
        retrieved_indices = set()
        
    has_a = (chunk_a in retrieved_indices) if chunk_a is not None else False
    has_b = (chunk_b in retrieved_indices) if chunk_b is not None else False
    
    status = "CORRECT" if is_correct else "WRONG"
    print(f"\nQ{i+1}: {status} | Pred: {pred} | Gold: {gold}")
    print(f"   Q: {q_text[:70]}...")
    if match:
        print(f"   Ev-A (idx {chunk_a}): {'FOUND' if has_a else 'MISSING'} | Ev-B (idx {chunk_b}): {'FOUND' if has_b else 'MISSING'}")
    print(f"   Metrics: tokens={turn['metrics'].get('assembledTokens')}, thermal={turn['metrics'].get('thermal')}, speedCap={turn['metrics'].get('budgetSpeedCapTokens')}")
    print(f"   Resp preview: {resp[:120]}...")

print("\n" + "="*50)
print(f"FINAL SCORE: {correct_count} / {total_evaluated} ({(correct_count/total_evaluated)*100:.1f}%)" if total_evaluated > 0 else "No matching questions.")
print("="*50)
