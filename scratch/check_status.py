import sqlite3

conn = sqlite3.connect('scratch/live_db/offline_docai.db')
c = conn.cursor()

for conv in c.execute("SELECT id, title, createdAt FROM conversations ORDER BY id DESC LIMIT 5"):
    conv_id = conv[0]
    msg_count = c.execute("SELECT COUNT(*) FROM messages WHERE conversationId = ?", (conv_id,)).fetchone()[0]
    print(f"Conv {conv_id}: '{conv[1]}' | Created: {conv[2]} | Messages: {msg_count}")

# Check messages of conv 13
print("\n--- Last 10 Messages of Conv 13 ---")
for m in c.execute("SELECT id, role, content FROM messages WHERE conversationId = 13 ORDER BY id DESC LIMIT 10").fetchall():
    print(f"[{m[1]} #{m[0]}] {m[2][:120]}...\n")
