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

with open('conv24_analysis_utf8.txt', 'w', encoding='utf-8') as f:
    f.write(f"Total user questions: {len(turns)}\n")
    for idx, t in enumerate(turns):
        f.write("=" * 60 + "\n")
        f.write(f"Turn {idx+1} [User Msg ID {t[0]}]:\n")
        f.write(f"Q: {t[1]}\n")
        f.write(f"A: {t[2]}\n")
        if t[3]:
            sources = json.loads(t[3])
            f.write(f"Sources ({len(sources)} blocks):\n")
            for s in sources:
                cids = s.get('chunkIds', [])
                pg = s.get('pageSpan', '')
                txt = s.get('text', '').replace('\n', ' ')
                f.write(f"  * Page {pg} (chunks {cids}): {txt[:140]}...\n")
