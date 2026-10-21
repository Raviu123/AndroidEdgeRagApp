import sqlite3
import json
import re

# Questions asked in conversation 17
QUESTIONS = [
    ("Q1", "Did Adam Smith send a message to Li Hua about the upcoming building maintenance schedule before the administrators announced a temporary change in the construction schedule due to weather conditions?"),
    ("Q2", "Did Wolfgang ask Li Hua about watching \"Star Wars: A New Hope\" after he asked Li Hua about going to see \"Overwatch 3\"?"),
    ("Q3", "Did Li Hua agree to go out for dinner after Wolfgang first asked him if he wanted to go out for dinner?"),
    ("Q4", "Did Li Hua send a message to Jennifer thanking her for the new training schedule before he requested a change in his training schedule for Thursday?"),
    ("Q5", "Did Li Hua ask Jennifer for advice on how to prevent muscle soreness after an intense workout session before he told her that he feels soreness in his arm muscles after the workout this week?"),
    ("Q6", "Did Li Hua send a message to Jennifer asking for her opinion on protein supplements before he consulted her about his daily protein powder consumption?"),
    ("Q7", "Did Yuriko ask Li Hua for help with her studio's homepage before she booked a seat at the \"Central Perk\" cafe?"),
    ("Q8", "Did Li Hua discuss his progress with the fitness plan before he shared a blog post about his recent fitness achievements?"),
    ("Q9", "Did Li Hua send a message to Jennifer asking if he can turn the Thursday class to Friday after he requested a change in his training schedule for Thursday?"),
    ("Q10", "Did Li Hua ask Yuriko to play music together before Wolfgang proposed to pause playing musical instruments?"),
    ("Q11", "Did Wolfgang Schulz recommend the band learns \"Viva la Vida\" by Coldplay after he and Li Hua discussed what song to play this Sunday?"),
    ("Q12", "Did Wolfgang's promotion announcement occur before he invited Li Hua for dinner on 20260430?"),
    ("Q13", "Did Turalyon announce the construction updates and feedback from residents after Illidan Stormrage complained about the construction noise?"),
    ("Q14", "Did Hailey announce the new line of high-protein breads before inviting Li Hua to the special bakery event?"),
    ("Q15", "Did Chae tell Li Hua that taking a warm shower before sleeping can improve the sleep quality before sharing the neuroscience article with her?"),
    ("Q16", "Did Li Hua ask Jennifer Moore for book recommendations on fitness nutrition before she announced the special guest speaker at the gym?"),
    ("Q17", "Did Jennifer remind Li Hua about proper nutrition and hydration before Jake shared his tips for staying hydrated during the match?"),
    ("Q18", "Did the group members talk about their favorite characters in the TV series Game of Thrones after Emily started a vote on the most hateable character?"),
    ("Q19", "Did Li Hua ask Thane about his opinion on The Last of Us before he asked about Sekiro: Shadows Die Twice?"),
    ("Q20", "Did Jake Watson and Li Hua discuss the classic matches between FC Barcelona and FC Bayern Munich before the group members discussed the classic matches between FC Barcelona and Real Madrid?")
]

conn = sqlite3.connect('offline_docai_device.db')
c = conn.cursor()

c.execute('SELECT id, role, content, sources, metrics FROM messages WHERE conversationId = 17 ORDER BY id ASC;')
rows = c.fetchall()

turns = []
cur_user = None
for r in rows:
    msg_id, role, content, sources, metrics = r
    if role == 'user':
        cur_user = (msg_id, content)
    elif role == 'assistant' and cur_user:
        turns.append((cur_user[0], cur_user[1], msg_id, content, json.loads(sources) if sources else [], json.loads(metrics) if metrics else {}))
        cur_user = None

# Compare each
print("=== ANALYSIS OF ALL 20 TURNS ===")
for i, (qid, qtext) in enumerate(QUESTIONS):
    uid, utext, aid, atext, sources, metrics = turns[i]
    ans_clean = atext.strip().lower()
    pred = "Yes" if (ans_clean.startswith("yes") or "yes," in ans_clean[:20]) else ("No" if (ans_clean.startswith("no") or "no," in ans_clean[:20]) else "Other")
    
    # Extract chunk IDs retrieved
    cids = []
    for s in sources:
        cids.extend(s.get('chunkIds', []))
        
    print(f"\n{qid}: Pred={pred}")
    print(f"  Question: {qtext[:70]}...")
    print(f"  Model response snippet: {atext[:100]}...")
    print(f"  Retrieved chunks: {len(cids)} chunks across pages {[s.get('page') for s in sources]}")
    print(f"  Tokens: {metrics.get('assembledTokens')} | Thermal: {metrics.get('thermal')} | SpeedCap: {metrics.get('budgetSpeedCapTokens')}")
