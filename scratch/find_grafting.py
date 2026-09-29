import json
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

transcript_path = r'C:\Users\siddu\.gemini\antigravity\brain\4fc90164-f3bf-4e46-8ff2-45974c2b17d4\.system_generated\logs\transcript_full.jsonl'

with open(transcript_path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f):
        if 'sweet cherry' in line or 'Grafting_15_Page_Guide' in line or 'whip and tongue' in line:
            data = json.loads(line)
            source = data.get('source', '')
            step_type = data.get('type', '')
            content = data.get('content', '')
            if source == 'USER_EXPLICIT' or 'USER' in source:
                print(f"=== USER Line {line_num} ===")
                print(content[:1500])
                print("\n")
            elif 'Grafting_15_Page_Guide' in content and '15-page' in content:
                print(f"=== {source} / {step_type} Line {line_num} ===")
                print(content[:1000])
                print("\n")
