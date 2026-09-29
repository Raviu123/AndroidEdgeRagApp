import json

with open(r'C:\Users\siddu\.gemini\antigravity\brain\4fc90164-f3bf-4e46-8ff2-45974c2b17d4\.system_generated\logs\transcript_full.jsonl', 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        if idx in (3487, 3565, 3596):
            d = json.loads(line)
            print(f"=== LINE {idx} ===")
            print(d.get('content', '')[:1000])
