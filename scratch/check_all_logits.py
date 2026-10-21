import sqlite3
import json
import math

conn = sqlite3.connect('offline_docai_device.db')
c = conn.cursor()

c.execute("SELECT id, sources FROM messages WHERE conversationId = 17 AND role = 'assistant';")
for mid, sjson in c.fetchall():
    if sjson:
        s = json.loads(sjson)
        scores = [x.get('score', 0) for x in s]
        logits = [math.log(x / (1.0 - x + 1e-7)) for x in scores if 0 < x < 1]
        if logits:
            min_l = min(logits)
            print(f"Msg {mid}: {len(s)} sources, min logit = {min_l:.2f}, max logit = {max(logits):.2f}")
