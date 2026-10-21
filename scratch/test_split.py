import re

LEAD_IN = re.compile(r'^\s*(?:did|was|were|is|are|does|do|has|have|had|can|could|would|should|will)\s+', re.I)
TEMPORAL_CONNECTORS = re.compile(r'\b(before|after|prior\s+to|subsequent\s+to|following|earlier\s+than|later\s+than)\b', re.I)
PRONOUN_SUBJECT = re.compile(r'^(?:he|she|they|it|we|i)\b', re.I)
NAMED_CLAUSE_SUBJECT = re.compile(r'^(?:the\s+[a-z]+|[A-Z][a-zA-Z0-9_-]+)\s+(?:first\s+|also\s+)?(?:[a-z]+ed|announced|asked|told|requested|discussed|shared|proposed|booked|consulted|sent|agreed|did|was|is)\b', re.I)

def cleanEntity(raw):
    s = raw.strip()
    s = re.sub(r'[?.!,;:\"]+$', '', s)
    s = re.sub(r'^(?i:the|a|an)\s+', '', s)
    return s.strip()

def extractSequentialClauses(q):
    clean = LEAD_IN.sub('', q.strip()).rstrip('?.! ')
    allMatches = list(TEMPORAL_CONNECTORS.finditer(clean))
    if not allMatches: return None
    candidates = []
    for m in allMatches:
        left = clean[:m.start()].strip()
        right = clean[m.end():].strip()
        wordsL = [w for w in left.split() if w]
        wordsR = [w for w in right.split() if w]
        balance = min(len(wordsL), len(wordsR)) / max(len(wordsL), len(wordsR))
        score = balance * 10.0
        if PRONOUN_SUBJECT.search(right): score += 15.0
        elif NAMED_CLAUSE_SUBJECT.search(right): score += 12.0
        candidates.append((left, right, score, m.group(0)))
    best = max(candidates, key=lambda x: x[2])
    return cleanEntity(best[0]), cleanEntity(best[1]), best[3]

questions = [
    'Did Adam Smith send a message to Li Hua about the upcoming building maintenance schedule before the administrators announced a temporary change in the construction schedule due to weather conditions?',
    'Did Wolfgang ask Li Hua about watching "Star Wars: A New Hope" after he asked Li Hua about going to see "Overwatch 3"?',
    'Did Li Hua agree to go out for dinner after Wolfgang first asked him if he wanted to go out for dinner?'
]

for q in questions:
    res = extractSequentialClauses(q)
    print('Q:', q[:60] + '...')
    print('  Side A:', res[0])
    print('  Side B:', res[1])
    print('  Connector:', res[2])
