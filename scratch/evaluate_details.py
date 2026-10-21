import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('scratch/conv16_scored.json', 'r', encoding='utf-8') as f:
    items = json.load(f)

# Let's inspect each item and score it
# Gold answers and expected checks
verdicts = []
for item in items:
    idx = item['index']
    q = item['question']
    gold = str(item['gold']).strip()
    ans = item['model_answer'].strip()
    
    first_sentence = ans.split('.')[0].strip()
    
    verdict = ""
    comment = ""
    
    # Question by Question evaluation:
    if idx == 1:
        # Gold: Yes. Model: No
        verdict = "❌"
        comment = "Model answered No (missed Event 2 in prompt)"
    elif idx == 2:
        # Gold: Yes. Model: No
        verdict = "❌"
        comment = "Model answered No (Gold says Yes, but doc has Li Hua asking, not Wolfgang)"
    elif idx == 3:
        # Gold: Yes. Model: No
        verdict = "❌"
        comment = "Model answered No (Gold says Yes, but doc shows Li Hua declined first request)"
    elif idx == 4:
        # Gold: Yes. Model: No
        verdict = "❌"
        comment = "Model answered No"
    elif idx == 5:
        # Gold: Yes. Model: Yes
        verdict = "✅"
        comment = "Matched gold 'Yes' with full explanation"
    elif idx == 6:
        # Gold: Yes. Model: No
        verdict = "❌"
        comment = "Model answered No"
    elif idx == 7:
        # Gold: Yes. Model: No
        verdict = "❌"
        comment = "Model answered No"
    elif idx == 8:
        # Gold: Yes. Model: Yes
        verdict = "✅"
        comment = "Matched gold 'Yes' with correct citation"
    elif idx == 9:
        # Gold: Yes. Model: No
        verdict = "❌"
        comment = "Model answered No"
    elif idx == 10:
        # Gold: Yes. Model: No
        verdict = "❌"
        comment = "Model answered No"
    elif idx == 11:
        # Gold: Yes. Model: No
        verdict = "❌"
        comment = "Model answered No"
    elif idx == 12:
        # Gold: Yes. Model: No
        verdict = "❌"
        comment = "Model answered No (Gold cites 2026, model cites 2021/2025)"
    elif idx == 13:
        # Gold: Yes. Model: No
        verdict = "❌"
        comment = "Model answered No"
    elif idx == 14:
        # Gold: Yes. Model: No
        verdict = "❌"
        comment = "Model answered No"
    elif idx == 15:
        # Gold: No. Model: No
        verdict = "✅"
        comment = "Matched gold 'No'"
    elif idx == 16:
        # Q: What time does Li Hua check in with Adam about moving in?
        # Gold: 5:30 PM. Model: 5:30 PM on January 6th
        verdict = "✅"
        comment = "Exact match: 5:30 PM"
    elif idx == 17:
        # Q: When was the first time Li Hua had dinner with Wolfgang this year?
        # Gold: 20260108. Model: I couldn't find this...
        verdict = "🚫"
        comment = "Evaded / Not found (RAG missed chunk)"
    elif idx == 18:
        # Q: Where was the first time Li Hua had dinner with Wolfgang this year?
        # Gold: the cozy café downtown. Model: I couldn't find this...
        verdict = "🚫"
        comment = "Evaded / Not found (RAG missed chunk)"
    elif idx == 19:
        # Q: What time is Li Hua's lunch with Wolfgang Schulz at the cozy café downtown?
        # Gold: 12:00 (or 20260108). Model: 12:00
        verdict = "✅"
        comment = "Exact match: 12:00"
    elif idx == 20:
        # Q: What is the Wi-Fi password at Li Hua's house?
        # Gold: Family123. Model: Family123
        verdict = "✅"
        comment = "Exact match: Family123"
    elif idx == 21:
        # Q: What does Adam say about having friends over?
        # Gold: having friends over occasionally is fine. Model: having friends over occasionally is fine, but keep gatherings reasonable
        verdict = "✅"
        comment = "Exact match: having friends over occasionally is fine"
    elif idx == 22:
        # Q: What house rule does Adam mention?
        # Gold: keep noise to a minimum during late hours. Model: no specific house rules mentioned in the sources
        verdict = "❌"
        comment = "RAG missed the house rules chunk"
    elif idx == 23:
        # Q: What does Li Hua report to Adam on January 6th?
        # Gold: the water tab in the apartment is broken. Model: I couldn't find this...
        verdict = "🚫"
        comment = "Evaded / Not found (RAG missed chunk)"
    elif idx == 24:
        # Q: When does Adam confirm the plumber will arrive?
        # Gold: tomorrow at 10 AM. Model: tomorrow at 10 AM
        verdict = "✅"
        comment = "Exact match: tomorrow at 10 AM"
    elif idx == 25:
        # Q: What time will Li Hua and Wolfgang meet for breakfast on the morning of the 9th?
        # Gold: Insufficient information (Null question). Model: I couldn't find this in the selected document(s)
        verdict = "✅"
        comment = "Correctly identified missing/null information without hallucinating!"
        
    verdicts.append((idx, q, gold, first_sentence, verdict, comment))

# Print summary table
correct = sum(1 for v in verdicts if v[4] == '✅')
partial = sum(1 for v in verdicts if v[4] == '⚠️')
wrong = sum(1 for v in verdicts if v[4] == '❌')
evaded = sum(1 for v in verdicts if v[4] == '🚫')
total = len(verdicts)

print(f"{'#':<3} | {'Verdict':<4} | {'Gold Answer':<25} | {'Model Answer':<35} | {'Notes'}")
print("-" * 115)
for v in verdicts:
    print(f"{v[0]:<3} | {v[4]:<4} | {v[2][:25]:<25} | {v[3][:35]:<35} | {v[5]}")

print("\n" + "=" * 50)
print(f"📊 FINAL SCORE SUMMARY (Conversation 16 - 25 Questions)")
print(f"✅ Correct: {correct} / {total}")
print(f"⚠️ Partial: {partial} / {total}")
print(f"❌ Wrong: {wrong} / {total}")
print(f"🚫 Not Found / Evaded: {evaded} / {total}")
print("-" * 50)
strict_acc = (correct / total) * 100
lenient_acc = ((correct + partial) / total) * 100
hallucination_evasion = ((wrong + evaded) / total) * 100
print(f"Strict Accuracy (Correct / Total):       {correct} / {total} = {strict_acc:.1f}%")
print(f"Lenient Accuracy (Correct+Partial / Total): {correct+partial} / {total} = {lenient_acc:.1f}%")
print(f"Error / Evasion Rate (Wrong+Evaded / Total): {wrong+evaded} / {total} = {hallucination_evasion:.1f}%")
