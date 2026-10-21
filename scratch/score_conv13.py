import sys
import json
import sqlite3

sys.stdout.reconfigure(encoding='utf-8')

# Load ground truth questions
with open('benchmarks/pdf_for_testing/lihuaworld-qset-1.json', 'r', encoding='utf-8') as f:
    gold_data = json.load(f)

gold_dict = {item['question'].strip(): item for item in gold_data}

conn = sqlite3.connect('scratch/live_db/offline_docai.db')
c = conn.cursor()

user_msgs = c.execute("SELECT id, content FROM messages WHERE conversationId = 13 AND role = 'user' ORDER BY id ASC").fetchall()
asst_msgs = c.execute("SELECT id, content, sources, metrics FROM messages WHERE conversationId = 13 AND role = 'assistant' AND id > 557 ORDER BY id ASC").fetchall()

results = []
for i, (u, a) in enumerate(zip(user_msgs, asst_msgs), 1):
    q_text = u[1].strip()
    gold_item = gold_dict.get(q_text)
    
    # If not exact match, search closest
    if not gold_item:
        for k, v in gold_dict.items():
            if k in q_text or q_text in k:
                gold_item = v
                break
                
    results.append({
        'index': i,
        'question': q_text,
        'gold': gold_item,
        'assistant_id': a[0],
        'answer': a[1],
        'sources': json.loads(a[2]) if a[2] else [],
        'metrics': json.loads(a[3]) if a[3] else {}
    })

print(f"Loaded {len(results)} paired Q&A items.")
with open('scratch/conv13_evaluation.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Saved evaluation data to scratch/conv13_evaluation.json")
