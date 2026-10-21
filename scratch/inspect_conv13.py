import subprocess
import io
import tarfile
import sqlite3
import json

# 1. Pull latest database from phone
res = subprocess.run(['adb', 'exec-out', 'run-as', 'com.siddu.offlinedocai', 'tar', '-cf', '-', '-C', 'databases', '.'], capture_output=True)
tf = tarfile.open(fileobj=io.BytesIO(res.stdout))
tf.extractall('scratch/live_db')

# 2. Inspect
conn = sqlite3.connect('scratch/live_db/offline_docai.db')
c = conn.cursor()

user_msgs = c.execute("SELECT id, content FROM messages WHERE conversationId = 13 AND role = 'user' ORDER BY id ASC").fetchall()
asst_msgs = c.execute("SELECT id, content, sources, metrics FROM messages WHERE conversationId = 13 AND role = 'assistant' AND id > 557 ORDER BY id ASC").fetchall()

print(f"Total questions: {len(user_msgs)}")
print(f"Total answers: {len(asst_msgs)}")

for i, (u, a) in enumerate(zip(user_msgs, asst_msgs), 1):
    print(f"\n--- Q{i} (User #{u[0]} -> Asst #{a[0]}) ---")
    print(f"Q: {u[1]}")
    print(f"A: {a[1]}")
