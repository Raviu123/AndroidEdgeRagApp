import json

qset = [
  {
    "id": 1,
    "question": "Did Adam Smith send a message to Li Hua about the upcoming building maintenance schedule before the administrators announced a temporary change in the construction schedule due to weather conditions?",
    "answer": "Yes",
    "evidence_A": "2026-01-21 10:00 (Week 3, Page 6, Chunk 553): Adam Smith notifies Li Hua about upcoming building maintenance schedule.",
    "evidence_B": "2026-07-01 10:00 (Week 26, Page 71, Chunk 1538/1539): Tirion Fordring announces construction is postponed due to weather/rainstorm.",
    "explanation": "Event A occurred on Jan 21, 2026, which is before Event B on July 1, 2026."
  },
  {
    "id": 2,
    "question": "Did Wolfgang ask Li Hua about watching \"Star Wars: A New Hope\" after he asked Li Hua about going to see \"Overwatch 3\"?",
    "answer": "Yes",
    "strict_note": "In text, Wolfgang asked about Overwatch 3 in Week 3, and Li Hua asked Wolfgang about Star Wars in Week 40. Chronologically, Star Wars occurred after Overwatch 3.",
    "evidence_A": "2026-01-21 13:00 (Week 3, Page 6, Chunk 555): Wolfgang asks Li Hua to check out Overwatch 3.",
    "evidence_B": "2026-10-09 17:00 (Week 40, Page 125, Chunk 2361): Conversation between Li Hua and Wolfgang about watching Star Wars: A New Hope.",
    "explanation": "Overwatch 3 was discussed on Jan 21, 2026. Star Wars was discussed on Oct 9, 2026 (after)."
  },
  {
    "id": 3,
    "question": "Did Li Hua agree to go out for dinner after Wolfgang first asked him if he wanted to go out for dinner?",
    "answer": "Yes",
    "strict_note": "When first asked on Jan 23, Li Hua declined; however, Li Hua later agreed to dinner in Week 39. Benchmark treats this chronologically as 'agreement happened after first invitation'.",
    "evidence_A": "2026-01-23 17:00 (Week 3, Page 7, Chunk 571): Wolfgang first asks Li Hua out for dinner (Li Hua initially declines due to work).",
    "evidence_B": "2026-09-30 16:00 (Week 39, Page 121, Chunk 2289): Li Hua agrees to go out for dinner with Wolfgang.",
    "explanation": "Initial dinner invite was Jan 23, 2026; dinner agreement occurred on Sept 30, 2026 (after)."
  },
  {
    "id": 4,
    "question": "Did Li Hua send a message to Jennifer thanking her for the new training schedule before he requested a change in his training schedule for Thursday?",
    "answer": "Yes",
    "evidence_A": "2026-02-04 15:00 (Week 5, Page 8, Chunk 590): Li Hua thanks Jennifer for the training schedule.",
    "evidence_B": "2026-02-11 19:00 (Week 6, Page 10, Chunk 611): Li Hua requests schedule adjustment for Thursday.",
    "explanation": "Thank you message on Feb 4, 2026 is before the Thursday schedule change request on Feb 11, 2026."
  },
  {
    "id": 5,
    "question": "Did Li Hua ask Jennifer for advice on how to prevent muscle soreness after an intense workout session before he told her that he feels soreness in his arm muscles after the workout this week?",
    "answer": "Yes",
    "evidence_A": "2026-02-06 16:00 (Week 5, Page 9, Chunk 600): Li Hua asks Jennifer for tips on dealing with muscle soreness.",
    "evidence_B": "2026-10-08 14:00 (Week 40, Page 124, Chunk 2348): Jennifer checks in on Li Hua's arm muscle soreness after bench press.",
    "explanation": "Muscle soreness prevention question on Feb 6, 2026 occurred before the arm muscle discussion on Oct 8, 2026."
  },
  {
    "id": 6,
    "question": "Did Li Hua send a message to Jennifer asking for her opinion on protein supplements before he consulted her about his daily protein powder consumption?",
    "answer": "Yes",
    "evidence_A": "2026-02-14 16:00 (Week 6, Page 11, Chunk 622): Li Hua asks Jennifer's opinion on protein supplements.",
    "evidence_B": "2026-11-20 20:00 (Week 46, Page 153, Chunk 2792/2793): Li Hua asks Jennifer about daily protein powder intake.",
    "explanation": "Supplements question on Feb 14, 2026 occurred before daily protein powder consultation on Nov 20, 2026."
  },
  {
    "id": 7,
    "question": "Did Yuriko ask Li Hua for help with her studio's homepage before she booked a seat at the \"Central Perk\" cafe?",
    "answer": "Yes",
    "strict_note": "Yuriko asked for homepage help on Feb 23, and confirmed meeting/seat at Central Perk on Feb 25.",
    "evidence_A": "2026-02-23 15:00 (Week 8, Page 12, Chunk 642/643): Yuriko asks Li Hua for help improving her studio's homepage.",
    "evidence_B": "2026-02-25 15:00 (Week 8, Page 12, Chunk 648/649): Yuriko and Li Hua arrange/book their meeting at Central Perk.",
    "explanation": "Homepage help request on Feb 23, 2026 was before the Central Perk arrangement on Feb 25, 2026."
  },
  {
    "id": 8,
    "question": "Did Li Hua discuss his progress with the fitness plan before he shared a blog post about his recent fitness achievements?",
    "answer": "Yes",
    "evidence_A": "2026-01-14 12:00 (Week 2, Page 4) / 2026-02-04 (Week 5): Li Hua discusses fitness plan progress with Jennifer.",
    "evidence_B": "2026-08-16 11:00 (Week 32, Page 95, Chunk 1880): Li Hua posts/shares his fitness achievements blog post.",
    "explanation": "Fitness progress discussions began in Weeks 2-5, well before the blog post in Week 32 (Aug 16)."
  },
  {
    "id": 9,
    "question": "Did Li Hua send a message to Jennifer asking if he can turn the Thursday class to Friday after he requested a change in his training schedule for Thursday?",
    "answer": "Yes",
    "evidence_A": "2026-02-11 19:00 (Week 6, Page 10, Chunk 611): Li Hua requests schedule change for Thursday.",
    "evidence_B": "2026-03-09 17:00 (Week 10, Page 16, Chunk 699): Li Hua asks Jennifer if he can switch class from Thursday to Friday.",
    "explanation": "First Thursday change request was Feb 11, 2026; switching Thursday to Friday was requested on March 9, 2026 (after)."
  },
  {
    "id": 10,
    "question": "Did Li Hua ask Yuriko to play music together before Wolfgang proposed to pause playing musical instruments?",
    "answer": "Yes",
    "evidence_A": "2026-03-18 15:10 (Week 11, Page 18, Chunk 730/738): Li Hua invites Yuriko to jam sessions.",
    "evidence_B": "2026-06-21 16:30 (Week 24, Page 66, Chunk 1460): Wolfgang proposes pausing musical instrument practice.",
    "explanation": "Jam session invitation was on March 18, 2026, before the pause proposal on June 21, 2026."
  },
  {
    "id": 11,
    "question": "Did Wolfgang Schulz recommend the band learns \"Viva la Vida\" by Coldplay after he and Li Hua discussed what song to play this Sunday?",
    "answer": "Yes",
    "evidence_A": "2026-03-29 18:00 (Week 13, Page 25, Chunk 830): Sunday jam song discussion.",
    "evidence_B": "2026-06-25 19:00 (Week 25, Page 68, Chunk 1491): Wolfgang recommends band learn 'Viva la Vida' by Coldplay.",
    "explanation": "Discussion on what song to play was in Week 13; Wolfgang's Coldplay recommendation was in Week 25 (after)."
  },
  {
    "id": 12,
    "question": "Did Wolfgang's promotion announcement occur before he invited Li Hua for dinner on 20260430?",
    "answer": "Yes",
    "evidence_A": "2026-04-28 17:30 (Week 17, Page 36, Chunk 993): Wolfgang announces his promotion and suggests celebrating.",
    "evidence_B": "2026-04-30 18:00 (Week 17, Page 36, Chunk 999/1000): Dinner celebration on April 30, 2026.",
    "explanation": "Wolfgang announced promotion on April 28, before having dinner on April 30, 2026."
  },
  {
    "id": 13,
    "question": "Did Turalyon announce the construction updates and feedback from residents after Illidan Stormrage complained about the construction noise?",
    "answer": "Yes",
    "evidence_A": "2026-05-25 14:00 (Week 21, Page 53, Chunk 1258): Illidan complains about the construction noise.",
    "evidence_B": "2026-06-20 10:00 (Week 24, Page 66, Chunk 1455): Turalyon announces construction updates and feedback from residents.",
    "explanation": "Illidan's complaint occurred in Week 21, before Turalyon's feedback/update announcement in Week 24."
  },
  {
    "id": 14,
    "question": "Did Hailey announce the new line of high-protein breads before inviting Li Hua to the special bakery event?",
    "answer": "Yes",
    "evidence_A": "2026-04-12 10:00 (Week 15, Page 26, Chunk 853): Hailey announces high-protein breads.",
    "evidence_B": "2026-06-11 12:45 (Week 23, Page 62, Chunk 1408/1409): Special bakery event invitation / delivery discussion.",
    "explanation": "High-protein bread announcement on April 12, 2026 occurred before the bakery event on June 11, 2026."
  },
  {
    "id": 15,
    "question": "Did Chae tell Li Hua that taking a warm shower before sleeping can improve the sleep quality before sharing the neuroscience article with her?",
    "answer": "No",
    "evidence_A": "2026-09-26 10:30 (Week 38, Page 120, Chunk 2274/2275): Warm shower mentioned for unwinding.",
    "evidence_B": "2026-07-28 15:00 (Week 30, Page 82, Chunk 1693): Neuroscience article shared.",
    "explanation": "The neuroscience article was shared in Week 30 (July), whereas the warm shower conversation was in Week 38 (September). Warm shower did NOT occur before the article."
  },
  {
    "id": 16,
    "question": "Did Li Hua ask Jennifer Moore for book recommendations on fitness nutrition before she announced the special guest speaker at the gym?",
    "answer": "No",
    "evidence_A": "2026-08-31 19:00 (Week 35, Page 104, Chunk 2024): Li Hua asks Jennifer for book recommendations on fitness nutrition.",
    "evidence_B": "2026-07-13 19:30 (Week 28, Page 78, Chunk 1641): Jennifer announces the special guest speaker at the gym.",
    "explanation": "Guest speaker was announced on July 13, 2026 (Week 28). Book recommendation request was on Aug 31, 2026 (Week 35). Book request did NOT occur before the announcement."
  },
  {
    "id": 17,
    "question": "Did Jennifer remind Li Hua about proper nutrition and hydration before Jake shared his tips for staying hydrated during the match?",
    "answer": "Yes",
    "evidence_A": "2026-06-10 11:00 (Week 23, Page 62, Chunk 1406/1407): Jennifer reminds Li Hua about proper nutrition and hydration.",
    "evidence_B": "2026-10-01 18:00 (Week 39, Page 121, Chunk 2295/2296): Jake shares tips for staying hydrated during the match.",
    "explanation": "Jennifer's reminder on June 10, 2026 (Week 23) occurred before Jake's tips on Oct 1, 2026 (Week 39)."
  },
  {
    "id": 18,
    "question": "Did the group members talk about their favorite characters in the TV series Game of Thrones after Emily started a vote on the most hateable character?",
    "answer": "Yes",
    "evidence_A": "2026-08-31 20:00 (Week 35, Page 106, Chunk 2060): Emily starts a vote on the most hateable GoT character (Joffrey, Cersei, Ramsay).",
    "evidence_B": "2026-09-02 20:30 (Week 35, Page 107, Chunk 2067): Group members transition to discussing their favorite characters.",
    "explanation": "Vote on most hateable character was started first; discussion of favorite characters occurred after."
  },
  {
    "id": 19,
    "question": "Did Li Hua ask Thane about his opinion on The Last of Us before he asked about Sekiro: Shadows Die Twice?",
    "answer": "No",
    "evidence_A": "2026-12-28 14:00 (Week 52, Page 175, Chunk 3138/3139): Li Hua asks Thane about The Last of Us.",
    "evidence_B": "2026-09-30 20:00 (Week 39, Page 122, Chunk 2309): Li Hua asks Thane about Sekiro: Shadows Die Twice.",
    "explanation": "Sekiro was asked about in Week 39 (September). The Last of Us was asked about in Week 52 (December). The Last of Us was NOT asked before Sekiro."
  },
  {
    "id": 20,
    "question": "Did Jake Watson and Li Hua discuss the classic matches between FC Barcelona and FC Bayern Munich before the group members discussed the classic matches between FC Barcelona and Real Madrid?",
    "answer": "No",
    "evidence_A": "2026-12-08 16:00 (Week 50, Page 169, Chunk 3050): Discussion regarding Barcelona and Bayern Munich.",
    "evidence_B": "2026-10-26 16:00 (Week 43, Page 136, Chunk 2538/2545): Group discussion on classic El Clásico matches (Barcelona vs Real Madrid).",
    "explanation": "Barcelona vs Real Madrid discussion was in Week 43 (October). Barcelona vs Bayern Munich was in Week 50 (December). Bayern Munich did NOT occur before Real Madrid."
  }
]

with open('benchmarks/pdf_for_testing/multy-qset-20.json', 'w', encoding='utf-8') as f:
    json.dump(qset, f, indent=2, ensure_ascii=False)

print("Saved verified multy-qset-20.json successfully!")
