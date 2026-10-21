import re

q1 = "Did Adam Smith send a message to Li Hua about the upcoming building maintenance schedule before the administrators announced a temporary change in the construction schedule due to weather conditions?"

LEAD_IN = re.compile(r"""^\s*(?:did|was|were|is|are|does|do|has|have|had|can|could|would|should|will)\s+""", re.IGNORECASE)
TEMPORAL_CONNECTORS = re.compile(r"""\b(before|after|prior\s+to|subsequent\s+to|following|earlier\s+than|later\s+than)\b""", re.IGNORECASE)

clean = LEAD_IN.sub("", q1.strip()).rstrip('?.! ')
m = TEMPORAL_CONNECTORS.search(clean)
if m:
    left = clean[:m.start()].strip()
    right = clean[m.end():].strip()
    print("Side A (left):", left)
    print("Side B (right):", right)
