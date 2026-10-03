import re

with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/validation/CitationValidator.kt', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')
for i, l in enumerate(lines[884:892], start=885):
    print(f'LINE {i}: {repr(l)}')
