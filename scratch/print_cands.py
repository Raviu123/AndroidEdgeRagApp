import sqlite3
import json

conn = sqlite3.connect('offline_docai_device.db')
c = conn.cursor()

c.execute("SELECT id, metrics FROM messages WHERE conversationId = 17 AND role = 'assistant';")
for mid, mjson in c.fetchall():
    if mjson:
        m = json.loads(mjson)
        print(f"Msg {mid}: candCount={m.get('candidateChunksCount')}")
