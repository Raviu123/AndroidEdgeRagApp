import sqlite3
import json

conn = sqlite3.connect('offline_docai_conv24.db')
c = conn.cursor()

query = """
SELECT m1.id, m1.content, m2.content, m2.sources 
FROM messages m1 
JOIN messages m2 ON m2.id = m1.id + 1 
WHERE m1.conversationId = 24 AND m1.role = 'user'
"""

c.execute(query)
turns = c.fetchall()

# Let's inspect Turn 5 (soreness) and Turn 12 (promotion vs dinner)
with open('turns_detail.txt', 'w', encoding='utf-8') as f:
    for idx in [1, 2, 4, 11]:
        t = turns[idx]
        f.write("=" * 60 + "\n")
        f.write(f"TURN {idx+1}\nQ: {t[1]}\nA: {t[2]}\n")
        sources = json.loads(t[3])
        for s in sources:
            f.write(f"--- Source chunkIds: {s.get('chunkIds')} Page: {s.get('pageSpan')} Section: {s.get('sectionTitle')}\n")
            f.write(s.get('text', '') + "\n")
