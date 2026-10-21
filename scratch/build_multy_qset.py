import sqlite3
import json
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
conn = sqlite3.connect('scratch/live_db/offline_docai.db')
c = conn.cursor()

def get_chunk(cid):
    return c.execute("SELECT id, pageNumber, sectionTitle, text FROM chunks WHERE id = ?", (cid,)).fetchone()

# Let's inspect each question's exact dates and text
results = []

# --- Q1 ---
# Event A: Adam Smith building maintenance schedule: 2026-01-21 10:00 (Week 3, Page 6, Chunk 553)
# Event B: Tirion Fordring / Admins construction postponed due to weather: 2026-07-01 10:00 (Week 26, Page 71, Chunk 1538)
# Did Event A occur before Event B?
# Jan 21, 2026 is BEFORE July 1, 2026!
results.append({
    "id": 1,
    "question": "Did Adam Smith send a message to Li Hua about the upcoming building maintenance schedule before the administrators announced a temporary change in the construction schedule due to weather conditions?",
    "answer": "Yes",
    "event_a": "2026-01-21 10:00 (Week 3, Page 6, Chunk 553): Adam Smith messages Li Hua about upcoming building maintenance schedule.",
    "event_b": "2026-07-01 10:00 (Week 26, Page 71, Chunk 1538): Construction postponed due to weather / rainstorm.",
    "explanation": "Event A occurred on Jan 21, 2026, which is before Event B on July 1, 2026."
})

# --- Q2 ---
# Event A: Wolfgang asks Li Hua about watching Star Wars (Wait, Li Hua asked Wolfgang on Week 40: 2026-10-09 17:00, Page 125, Chunk 2361)
# Event B: Wolfgang asked Li Hua about Overwatch 3: 2026-01-21 13:00 (Week 3, Page 6, Chunk 555)
# In the benchmark, the question asks: Did Wolfgang ask Li Hua about Star Wars AFTER he asked about Overwatch 3?
# In the text: Wolfgang asked about Overwatch 3 on Jan 21. Later on Oct 9, Li Hua asks Wolfgang about Star Wars.
# Note: If evaluated strictly on speaker: Wolfgang never initiated the Star Wars invite (Li Hua did). But chronologically, the Star Wars discussion happened in Week 40, long after Overwatch 3 in Week 3.
results.append({
    "id": 2,
    "question": "Did Wolfgang ask Li Hua about watching \"Star Wars: A New Hope\" after he asked Li Hua about going to see \"Overwatch 3\"?",
    "answer": "Yes (Strict: No on speaker, Yes on timeline)",
    "benchmark_answer": "Yes",
    "event_a": "2026-10-09 17:00 (Week 40, Page 125, Chunk 2361): Conversation about watching Star Wars: A New Hope at Li Hua's place.",
    "event_b": "2026-01-21 13:00 (Week 3, Page 6, Chunk 555): Wolfgang asks Li Hua to check out Overwatch 3.",
    "explanation": "Overwatch 3 discussion was in Week 3 (Jan 21). Star Wars movie night discussion was in Week 40 (Oct 9). Week 40 is after Week 3 (though in the text Li Hua initiated the Star Wars chat)."
})

# --- Q3 ---
# Event A: Wolfgang first asks dinner: 2026-01-23 17:00 (Week 3, Page 7, Chunk 571) - Li Hua declined!
# Event B: Li Hua later agreed to dinner: 2026-05-18 (Week 20, Page 49) / 2026-09-30 16:00 (Week 39)
# Question: Did Li Hua agree to go out for dinner after Wolfgang first asked him if he wanted to go out for dinner?
results.append({
    "id": 3,
    "question": "Did Li Hua agree to go out for dinner after Wolfgang first asked him if he wanted to go out for dinner?",
    "answer": "Yes (Chronologically later) / No (Direct reply was decline)",
    "benchmark_answer": "Yes",
    "event_a": "2026-01-23 17:00 (Week 3, Page 7, Chunk 571): Wolfgang first asks Li Hua for dinner; Li Hua declines due to being busy.",
    "event_b": "2026-09-30 16:00 (Week 39, Page 121, Chunk 2289): Later in the year, Li Hua agrees to go out for dinner with Wolfgang.",
    "explanation": "Wolfgang first asked on Jan 23 (Li Hua declined). Months later (Sept 30), Li Hua agreed to dinner. Chronologically, the agreement came after the first invitation."
})

# --- Q4 ---
# Question: Did Li Hua send a message to Jennifer thanking her for the new training schedule before he requested a change in his training schedule for Thursday?
# Jennifer sends schedule / Li Hua thanks: 2026-02-04 15:00 (Week 5, Page 8, Chunk 590): "LiHua: Hey Jennifer! Thanks for understanding my situation... [thanks for schedule]"
# Request change for Thursday: 2026-02-11 19:00 (Week 6, Page 10, Chunk 611) / 2026-03-09 17:00 (Week 10, Chunk 699)
results.append({
    "id": 4,
    "question": "Did Li Hua send a message to Jennifer thanking her for the new training schedule before he requested a change in his training schedule for Thursday?",
    "answer": "Yes",
    "event_a": "2026-02-04 15:00 (Week 5, Page 8, Chunk 590): Li Hua thanks Jennifer for the training schedule.",
    "event_b": "2026-02-11 19:00 (Week 6, Page 10, Chunk 611): Li Hua requests schedule adjustment for Thursday.",
    "explanation": "Thanking message occurred on Feb 4, 2026, before the schedule change request on Feb 11, 2026."
})

# --- Q5 ---
# Question: Did Li Hua ask Jennifer for advice on how to prevent muscle soreness after an intense workout session before he told her that he feels soreness in his arm muscles after the workout this week?
results.append({
    "id": 5,
    "question": "Did Li Hua ask Jennifer for advice on how to prevent muscle soreness after an intense workout session before he told her that he feels soreness in his arm muscles after the workout this week?",
    "answer": "Yes",
    "event_a": "2026-02-06 16:00 (Week 5, Page 9, Chunk 600): Li Hua asks Jennifer for tips on dealing with muscle soreness.",
    "event_b": "2026-10-08 14:00 (Week 40, Page 124, Chunk 2348): Jennifer checks in on Li Hua's arm muscle soreness after bench press.",
    "explanation": "Li Hua asked for muscle soreness advice in Week 5 (Feb 6), long before discussing specific arm muscle soreness in Week 40 (Oct 8)."
})

print("Generated first 5 items.")
