import sqlite3
import json
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
conn = sqlite3.connect('scratch/live_db/offline_docai.db')
c = conn.cursor()

questions = [
  "Did Adam Smith send a message to Li Hua about the upcoming building maintenance schedule before the administrators announced a temporary change in the construction schedule due to weather conditions?",
  "Did Wolfgang ask Li Hua about watching \"Star Wars: A New Hope\" after he asked Li Hua about going to see \"Overwatch 3\"?",
  "Did Li Hua agree to go out for dinner after Wolfgang first asked him if he wanted to go out for dinner?",
  "Did Li Hua send a message to Jennifer thanking her for the new training schedule before he requested a change in his training schedule for Thursday?",
  "Did Li Hua ask Jennifer for advice on how to prevent muscle soreness after an intense workout session before he told her that he feels soreness in his arm muscles after the workout this week?",
  "Did Li Hua send a message to Jennifer asking for her opinion on protein supplements before he consulted her about his daily protein powder consumption?",
  "Did Yuriko ask Li Hua for help with her studio's homepage before she booked a seat at the \"Central Perk\" cafe?",
  "Did Li Hua discuss his progress with the fitness plan before he shared a blog post about his recent fitness achievements?",
  "Did Li Hua send a message to Jennifer asking if he can turn the Thursday class to Friday after he requested a change in his training schedule for Thursday?",
  "Did Li Hua ask Yuriko to play music together before Wolfgang proposed to pause playing musical instruments?",
  "Did Wolfgang Schulz recommend the band learns \"Viva la Vida\" by Coldplay after he and Li Hua discussed what song to play this Sunday?",
  "Did Wolfgang's promotion announcement occur before he invited Li Hua for dinner on 20260430?",
  "Did Turalyon announce the construction updates and feedback from residents after Illidan Stormrage complained about the construction noise?",
  "Did Hailey announce the new line of high-protein breads before inviting Li Hua to the special bakery event?",
  "Did Chae tell Li Hua that taking a warm shower before sleeping can improve the sleep quality before sharing the neuroscience article with her?",
  "Did Li Hua ask Jennifer Moore for book recommendations on fitness nutrition before she announced the special guest speaker at the gym?",
  "Did Jennifer remind Li Hua about proper nutrition and hydration before Jake shared his tips for staying hydrated during the match?",
  "Did the group members talk about their favorite characters in the TV series Game of Thrones after Emily started a vote on the most hateable character?",
  "Did Li Hua ask Thane about his opinion on The Last of Us before he asked about Sekiro: Shadows Die Twice?",
  "Did Jake Watson and Li Hua discuss the classic matches between FC Barcelona and FC Bayern Munich before the group members discussed the classic matches between FC Barcelona and Real Madrid?"
]

print(f"Loaded {len(questions)} questions for analysis.")
