import sqlite3
import json

conn = sqlite3.connect('offline_docai_device.db')
c = conn.cursor()

# Actual chunks retrieved in Msg 739
actual_739 = [496, 497, 510, 553, 577, 657, 690, 803, 996, 1309, 1598, 1789, 1837, 1841]

# In sim_full_hybrid, top 28 was:
sim_top_28 = [553, 510, 803, 657, 1003, 1538, 577, 1539, 1598, 996, 690, 497, 509, 1789, 1309, 1312, 1433, 1187, 496, 1271, 1313, 1841, 1599, 1600, 554, 1790, 1281, 1837]

print("Are all actual 739 chunks in sim_top_28?")
for cid in actual_739:
    print(f"  Chunk {cid}: in sim_top_28? {cid in sim_top_28} (sim rank: {sim_top_28.index(cid)+1 if cid in sim_top_28 else 'NO'})")

# Notice that in sim_top_28:
# 1538 was rank #6, 1539 was rank #8!
# Why did 1538 and 1539 NOT appear in actual_739?
