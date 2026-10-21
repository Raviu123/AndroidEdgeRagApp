import sys
import re
import json
sys.path.append('.')
from scratch.test_retrieval_suite import run_retrieval, conn, split_clauses

# Load questions from lihuaworld-qset-1.json and lihuaworld-qset-2.json
with open('benchmarks/pdf_for_testing/lihuaworld-qset-1.json') as f:
    q1 = json.load(f)
with open('benchmarks/pdf_for_testing/lihuaworld-qset-2.json') as f:
    q2 = json.load(f)

# First 20 questions
all_q = (q1 + q2)[:20]

print(f"Testing retrieval across {len(all_q)} questions...")

success_count = 0

for idx, item in enumerate(all_q):
    qid = item['id']
    qtext = item['question']
    ev_dates = item.get('evidence', '').split('<and>')
    print(f"\n--- Q{idx+1} (ID {qid}): {qtext[:65]}...")
    print(f"  Target Evidence Dates: {ev_dates}")
    
    kept = run_retrieval(qtext, floor_val=-10.0)
    cids = [x[0] for x in kept]
    
    # Check if target evidence chunks are retrieved
    c = conn.cursor()
    placeholders = ','.join('?' for _ in cids)
    c.execute(f"SELECT text FROM chunks WHERE id IN ({placeholders})", cids)
    all_retrieved_text = " ".join(r[0] for r in c.fetchall())
    
    found_dates = []
    for d in ev_dates:
        clean_d = d.replace('_', ' ')
        d_short = clean_d.split()[0]
        if d_short in all_retrieved_text or clean_d in all_retrieved_text:
            found_dates.append(d)
            
    has_both = len(found_dates) == len(ev_dates)
    print(f"  Retrieved {len(cids)} candidates. Found dates: {len(found_dates)}/{len(ev_dates)} -> {found_dates}")
    if has_both:
        success_count += 1
        print("  -> STATUS: PASS (BOTH EVENTS FOUND)")
    else:
        print("  -> STATUS: MISSING", [d for d in ev_dates if d not in found_dates])

print("\n" + "="*50)
print(f"OVERALL RETRIEVAL SCORE: {success_count} / {len(all_q)} ({(success_count/len(all_q))*100:.1f}%)")
print("="*50)
