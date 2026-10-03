import re

def update_file():
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/validation/CitationValidator.kt', 'r', encoding='utf-8') as f:
        content = f.read()

    old_matched = '''val matchedTokens = qTokens.filter { cLower.contains(it) }'''
    new_matched = '''val matchedTokens = qTokens.filter { t -> cLower.contains(t) || cLower.contains(t.trimEnd('s')) || (t.endsWith("ing") && cLower.contains(t.removeSuffix("ing"))) }'''

    content = content.replace(old_matched, new_matched)
    
    # Let's also fix importantEntities logic to do the same
    old_entities = '''if (importantEntities.any { cLower.contains(it) }) return true'''
    new_entities = '''if (importantEntities.any { t -> cLower.contains(t) || cLower.contains(t.trimEnd('s')) || (t.endsWith("ing") && cLower.contains(t.removeSuffix("ing"))) }) return true'''
    content = content.replace(old_entities, new_entities)

    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/validation/CitationValidator.kt', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    update_file()
