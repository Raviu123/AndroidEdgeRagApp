import sqlite3
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
conn = sqlite3.connect('scratch/live_db/offline_docai.db')
c = conn.cursor()

with open('benchmarks/pdf_for_testing/lihuaworld-qset-1.json', 'r', encoding='utf-8') as f:
    gold_data = json.load(f)

for q_idx in [0, 2, 3, 4, 5, 6]:
    item = gold_data[q_idx]
    q_id = item['id']
    q_text = item['question']
    ev = item['evidence']
    print(f"=================================================================")
    print(f"QUESTION {q_id+1}: {q_text}")
    print(f"REQUIRED EVIDENCE: {ev}")
    
    # Get the assistant message from Conv 13 for this question
    # Conv 13 user question matches q_text
    asst_msg = c.execute("""
        SELECT a.content, a.sources FROM messages a 
        JOIN messages u ON a.id = u.id + 1 
        WHERE a.conversationId = 13 AND u.content LIKE ?
    """, (f"%{q_text[:40]}%",)).fetchone()
    
    if asst_msg:
        print(f"\nASSISTANT ANSWER:\n{asst_msg[0]}")
        sources = json.loads(asst_msg[1]) if asst_msg[1] else []
        print(f"\nRETRIEVED {len(sources)} CHUNKS:")
        for s in sources:
            print(f"  - Page {s.get('page')}, Section {s.get('section')}: {s.get('text', '')[:100].strip()}...")
            
    print("\nCHECKING WHERE THE ACTUAL EVIDENCE IS IN THE DATABASE:")
    ev_parts = ev.split('<and>')
    for part in ev_parts:
        # e.g. 20260121_10:00 -> find date 2026-01-21 and hour 10:00 or filename 20260121_1000
        fname = part.replace(':', '') + ".txt" # 20260121_1000.txt
        res = c.execute("SELECT id, pageNumber, sectionTitle, text FROM chunks WHERE text LIKE ?", (f"%{fname}%",)).fetchall()
        if not res:
            # try date
            d_str = f"{part[:4]}-{part[4:6]}-{part[6:8]}"
            res = c.execute("SELECT id, pageNumber, sectionTitle, text FROM chunks WHERE text LIKE ? AND text LIKE ?", (f"%{d_str}%", f"%{part[9:14]}%")).fetchall()
        
        print(f"  * Evidence part: {part}")
        if res:
            for r in res:
                # check if this chunk was in retrieved sources
                in_sources = any(r[0] in s.get('chunkIds', []) for s in sources)
                print(f"    Chunk {r[0]} (Page {r[1]}, Sec {r[2]}): IN RETRIEVED SOURCES? {'YES' if in_sources else 'NO - MISSED BY RAG!'}")
                print(f"    Content: {r[3][:160].strip()}...")
        else:
            print(f"    Could not locate chunk for {part}")
    print()
