import json

with open('scratch/conv13_evaluation.json', 'r', encoding='utf-8') as f:
    items = json.load(f)

print("=== Checking Retrieved Chunks for Q1 to Q7 ===")
for item in items[:7]:
    idx = item['index']
    q = item['question']
    sources = item['sources']
    pages = [s.get('page') for s in sources]
    print(f"Q{idx}: {q[:70]}...")
    print(f"   Retrieved Pages: {pages}")
    print(f"   Gold Evidence: {item['gold']['evidence'] if item['gold'] else 'N/A'}")
