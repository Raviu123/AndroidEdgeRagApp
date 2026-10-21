import json
import sqlite3
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Load gold benchmark
with open('benchmarks/pdf_for_testing/lihuaworld-qset-1.json', 'r', encoding='utf-8') as f:
    gold_data = json.load(f)

gold_dict = {item['question'].strip(): item for item in gold_data}

conn = sqlite3.connect('scratch/live_db/offline_docai.db')
c = conn.cursor()

user_msgs = c.execute("SELECT id, content FROM messages WHERE conversationId = 16 AND role = 'user' ORDER BY id ASC").fetchall()
asst_msgs = c.execute("SELECT id, content, sources, metrics FROM messages WHERE conversationId = 16 AND role = 'assistant' AND id > 686 ORDER BY id ASC").fetchall()

print(f"Total Questions: {len(user_msgs)}")
print(f"Total Answers: {len(asst_msgs)}")

scored_items = []
for idx, (u, a) in enumerate(zip(user_msgs, asst_msgs), 1):
    q = u[1].strip()
    ans = a[1].strip()
    gold_item = gold_dict.get(q)
    if not gold_item:
        for k, v in gold_dict.items():
            if k in q or q in k:
                gold_item = v
                break
    gold_ans = gold_item.get('answer') if gold_item else 'UNKNOWN'
    q_type = gold_item.get('type') if gold_item else 'N/A'
    
    scored_items.append({
        'index': idx,
        'question': q,
        'gold': gold_ans,
        'type': q_type,
        'model_answer': ans
    })

with open('scratch/conv16_scored.json', 'w', encoding='utf-8') as f:
    json.dump(scored_items, f, indent=2, ensure_ascii=False)

print("Saved to scratch/conv16_scored.json")
