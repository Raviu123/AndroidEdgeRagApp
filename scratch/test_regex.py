import re

text = "The types of grafting include whip-and-tongue. [Source 1, Page 1] (Page 1)"
CITATION_REGEX = re.compile(r"\[(?:(?:Doc|Document)\s*[:=]?\s*(\d+),\s*)?(?:Source|SOURCE_ID)\s*[:=]?\s*(\d+)(?:,\s*Pages?\s*[:=]?\s*(\d+))?\]|\(Pages?\s+(\d+(?:\s*-\s*\d+)?)\)|\[Pages?\s+(\d+(?:\s*-\s*\d+)?)\]")

for m in CITATION_REGEX.finditer(text):
    print("Match:", m.group(0), m.groups())

# What about sanitizeCitations?
validSourceIds = {1}
validPages = {1}

def sanitizeCitations(t):
    def repl(match):
        docId = match.group(1)
        sourceId = match.group(2)
        pageFromSource = match.group(3)
        pageParen = match.group(4)
        pageBracket = match.group(5)
        page = pageFromSource or (pageParen.split('-')[0].strip() if pageParen else None) or (pageBracket.split('-')[0].strip() if pageBracket else None)
        
        s_id = int(sourceId) if sourceId else None
        p_id = int(page) if page else None
        
        if s_id is not None and validSourceIds and s_id not in validSourceIds:
            return ""
        elif p_id is not None and validPages and p_id not in validPages:
            return ""
        else:
            return match.group(0)

    res = CITATION_REGEX.sub(repl, t)
    res = re.sub(r"[ \t]{2,}", " ", res)
    res = re.sub(r"\n{3,}", "\n\n", res)
    return res.strip()

print("Sanitized:", repr(sanitizeCitations(text)))

# Now cleanAndRenumberList
cleaned = sanitizeCitations(text)
# Step 1
cleaned = re.sub(r"(?:\b\d+[\.):]\s+)+(\d+[\.):]\s+)(?=[A-Za-z*•\[\-])", r"\1", cleaned)
cleaned = re.sub(r"(?:\b\d+[\.):]\s+){2,}", "", cleaned)

# Step 2
cleaned = re.sub(r"(?m)^[ \t]*\d+[\.):][ \t]*$\r?\n?", "", cleaned)

# Step 3
orphanRegex = re.compile(r"(^|[\s])(\d+)[\.:]\s*(?=[,\.;:\[\]]|\Z)", re.MULTILINE)
# wait, in kotlin:
# val orphanRegex = Regex("""(^|[\\s])(\\d+)[\.:]\\s*(?=[,\.;:\[\]]|\\z)""", RegexOption.MULTILINE)

# Step 4: lines...
# Step 5: inlineItemRegex:
inlineItemRegex = re.compile(r"(^|[\s\n:])(\d+)[\.):](\s+)([A-Za-z*•\[\-])")

print("Cleaned so far:", repr(cleaned))

