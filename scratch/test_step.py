import re

stop_words = set([
    "what", "when", "where", "how", "this", "that", "there", "their", "who", "why",
    "them", "with", "from", "your", "about", "the", "is", "are", "was", "were", "and",
    "or", "in", "on", "at", "to", "a", "an", "of", "for", "by", "does", "do", "can",
    "you", "tell", "me", "explain", "give", "show", "please", "which", "will", "would",
    "mentioned", "say", "talk", "says", "talks", "i", "my", "we", "our", "all", "any",
    "be", "been", "being", "have", "has", "had", "using", "uses", "used"
])

def stem(raw):
    word = "".join(c for c in raw.lower() if c.isalnum())
    if len(word) <= 3:
        return word
    base = word
    if word.endswith("ing") and len(word) > 5: base = word[:-3]
    elif word.endswith("ation") and len(word) > 6: base = word[:-5]
    elif word.endswith("tion") and len(word) > 5: base = word[:-4]
    elif word.endswith("ment") and len(word) > 5: base = word[:-4]
    elif word.endswith("ance") and len(word) > 5: base = word[:-4]
    elif word.endswith("ence") and len(word) > 5: base = word[:-4]
    elif word.endswith("able") and len(word) > 5: base = word[:-4]
    elif word.endswith("ible") and len(word) > 5: base = word[:-4]
    elif word.endswith("ical") and len(word) > 6: base = word[:-4]
    elif word.endswith("ic") and len(word) > 4: base = word[:-2]
    elif word.endswith("ive") and len(word) > 5: base = word[:-3]
    elif word.endswith("ity") and len(word) > 5: base = word[:-3]
    elif word.endswith("ly") and len(word) > 4: base = word[:-2]
    elif word.endswith("al") and len(word) > 4: base = word[:-2]
    elif word.endswith("um") and len(word) > 4: base = word[:-2]
    elif word.endswith("ied") and len(word) > 4: base = word[:-3] + "y"
    elif word.endswith("ed") and len(word) > 4: base = word[:-2]
    elif word.endswith("ies") and len(word) > 4: base = word[:-3] + "y"
    elif word.endswith("es") and len(word) > 4: base = word[:-2]
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 3: base = word[:-1]

    syns = {
        "verif": "ensur", "verifi": "ensur", "verify": "ensur", "ensur": "ensur", "ensure": "ensur", "check": "ensur",
        "fuse": "join", "fusion": "join", "contact": "join", "unit": "join", "unite": "join", "bind": "join",
        "analysi": "analys", "analysis": "analys", "analyz": "analys", "analyze": "analys",
        "demonstr": "demonstr", "demonstrat": "demonstr", "demonstrate": "demonstr",
        "inspect": "inspect", "inspecti": "inspect",
        "requir": "requir", "require": "requir", "need": "requir",
        "specif": "specif", "specific": "specif",
        "operat": "operat", "operate": "operat",
        "design": "design", "designing": "design",
        "cambium": "cambi", "cambial": "cambi", "cambi": "cambi",
        "desicc": "dry", "desiccation": "dry", "dry": "dry", "drying": "dry",
        "heal": "heal", "healing": "heal", "callus": "heal",
        "protect": "protect", "shield": "protect", "seal": "protect", "wax": "protect",
        "prevent": "prevent", "avoid": "prevent",
        "fail": "fail", "failure": "fail",
        "cause": "cause", "reason": "cause",
        "allow": "enabl", "permit": "enabl", "enabl": "enabl",
        "provid": "provid", "suppli": "provid",
        "wound": "cut", "cut": "cut",
        "growth": "grow", "grow": "grow"
    }
    return syns.get(base, base)

def tokenizeAndStem(text):
    tokens = re.split(r"[^a-zA-Z0-9]+", text.lower())
    res = []
    for t in tokens:
        t = t.strip()
        if t and t not in stop_words and len(t) > 1:
            res.append(stem(t))
    return res

def computeSupportScore(claimText, evidenceText):
    claimStems = tokenizeAndStem(claimText)
    if not claimStems: return 1.0
    evidenceStems = set(tokenizeAndStem(evidenceText))
    if not evidenceStems: return 0.0
    matched = sum(1 for s in claimStems if s in evidenceStems)
    tokenOverlap = matched / len(claimStems)
    phraseBonus = 0.0
    if len(claimStems) >= 3:
        claimTokens = [t for t in re.split(r"[^a-zA-Z0-9]+", claimText.lower()) if len(t) > 2]
        for i in range(len(claimTokens) - 2):
            trigram = f"{claimTokens[i]} {claimTokens[i+1]} {claimTokens[i+2]}"
            if trigram in evidenceText.lower():
                phraseBonus = 0.25
                break
    return min(1.0, (tokenOverlap * 0.75) + phraseBonus), claimStems, evidenceStems, matched

claim = "You should inspect the graft union, rootstock, and scion for these issues."
evidence = """Page 13: Aftercare and Maintenance. Section 13: Post-Graft Care.
Inspect grafts weekly.
Remove suckers emerging from the rootstock below the graft union to direct vigor into the scion.
Page 14: Common Causes of Failure. Section 14: Troubleshooting Graft Failure.
Failures typically result from poor cambial contact, desiccation of tissues, grafting during incorrect physiological seasons, or pathogen infection."""

score, cStems, eStems, matched = computeSupportScore(claim, evidence)
print(f"Claim Stems: {cStems}")
print(f"Matched ({matched}/{len(cStems)}): {[s for s in cStems if s in eStems]}")
print(f"Score: {score}")
