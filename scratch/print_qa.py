import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('scratch/conv13_evaluation.json', 'r', encoding='utf-8') as f:
    items = json.load(f)

for item in items:
    idx = item['index']
    q = item['question']
    gold = item['gold']['answer'] if item['gold'] else 'MISSING'
    ans = item['answer'].strip()
    # Check first word/sentence
    first_line = ans.split('\n')[0]
    print(f"[{idx:02d}] GOLD: {gold}")
    print(f"     Q: {q}")
    print(f"     A: {first_line[:120]}")
    print()
