import sqlite3
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
conn = sqlite3.connect('scratch/live_db/offline_docai.db')
c = conn.cursor()

def search_one(terms):
    q = "SELECT id, pageNumber, sectionTitle, text FROM chunks WHERE " + " AND ".join(["text LIKE ?" for _ in terms])
    return c.execute(q, [f"%{t}%" for t in terms]).fetchall()

# Let's inspect Q6 to Q20 evidence
print("--- Q6 ---")
print("Supplements:", search_one(["protein supplements"]))
print("Daily powder:", search_one(["protein powder", "intake"]))

print("\n--- Q7 ---")
print("Yuriko homepage help:", [(r[0], r[1], r[2], r[3][:100]) for r in search_one(["improving the homepage"])])
print("Central Perk meeting:", [(r[0], r[1], r[2], r[3][:100]) for r in search_one(["Central Perk", "Thursday"])])

print("\n--- Q8 ---")
print("Fitness progress:", [(r[0], r[1], r[2], r[3][:100]) for r in search_one(["Jennifer", "progress", "fitness"])][:2])
print("Blog post:", [(r[0], r[1], r[2], r[3][:100]) for r in search_one(["blog post", "fitness goal"])])

print("\n--- Q9 ---")
print("Thursday change:", [(r[0], r[1], r[2], r[3][:100]) for r in search_one(["Thursday", "schedule", "change"])][:2])
print("Thursday to Friday:", [(r[0], r[1], r[2], r[3][:100]) for r in search_one(["Thursday to Friday"])])

print("\n--- Q10 ---")
print("Yuriko jam:", [(r[0], r[1], r[2], r[3][:100]) for r in search_one(["Yuriko", "jam", "music"])][:2])
print("Pause instruments:", [(r[0], r[1], r[2], r[3][:100]) for r in search_one(["pause", "playing"])])

print("\n--- Q11 ---")
print("Viva la Vida jam:", [(r[0], r[1], r[2], r[3][:100]) for r in search_one(["Viva la Vida", "Coldplay"])])

print("\n--- Q12 ---")
print("Promotion dinner:", [(r[0], r[1], r[2], r[3][:100]) for r in search_one(["promoted", "celebrate"])])

print("\n--- Q13 ---")
print("Construction noise complaint:", [(r[0], r[1], r[2], r[3][:100]) for r in search_one(["Illidan", "noise", "construction"])])
print("Turalyon update / quiet hours:", [(r[0], r[1], r[2], r[3][:100]) for r in search_one(["Turalyon", "noise", "hours"])])

print("\n--- Q14 ---")
print("High protein bread:", [(r[0], r[1], r[2], r[3][:100]) for r in search_one(["high-protein", "bread"])])
print("Special bakery event:", [(r[0], r[1], r[2], r[3][:100]) for r in search_one(["bakery", "event", "anniversary"])])

print("\n--- Q15 ---")
print("Warm shower:", [(r[0], r[1], r[2], r[3][:100]) for r in search_one(["warm shower"])])
print("Neuroscience:", [(r[0], r[1], r[2], r[3][:100]) for r in search_one(["neuroscience"])])

print("\n--- Q16 ---")
print("Book recs:", [(r[0], r[1], r[2], r[3][:100]) for r in search_one(["book", "recommend", "nutrition"])])
print("Guest speaker:", [(r[0], r[1], r[2], r[3][:100]) for r in search_one(["guest speaker"])])

print("\n--- Q17 ---")
print("Jennifer reminder:", [(r[0], r[1], r[2], r[3][:100]) for r in search_one(["hydration", "nutrition", "Jennifer"])])
print("Jake tips:", [(r[0], r[1], r[2], r[3][:100]) for r in search_one(["Jake", "hydrat", "match"])])

print("\n--- Q18 ---")
print("GoT favorite:", [(r[0], r[1], r[2], r[3][:100]) for r in search_one(["favorite", "Thrones"])])
print("Hateable character vote:", [(r[0], r[1], r[2], r[3][:100]) for r in search_one(["hateable"])])

print("\n--- Q19 ---")
print("The Last of Us Thane:", [(r[0], r[1], r[2], r[3][:100]) for r in search_one(["Last of Us", "Thane"])])
print("Sekiro Thane:", [(r[0], r[1], r[2], r[3][:100]) for r in search_one(["Sekiro", "Thane"])])

print("\n--- Q20 ---")
print("Barcelona Bayern:", [(r[0], r[1], r[2], r[3][:100]) for r in search_one(["Barcelona", "Bayern"])])
print("Barcelona Real Madrid:", [(r[0], r[1], r[2], r[3][:100]) for r in search_one(["Barcelona", "Real Madrid"])])
