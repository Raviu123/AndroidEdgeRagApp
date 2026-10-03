import re

def update_file():
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/retrieval/QueryIntentRouter.kt', 'r', encoding='utf-8') as f:
        content = f.read()

    # Update ENTITY_LOOKUP_REGEX
    old_regex = r'''^\s*(?i:what\s+is|tell\s+me\s+about)\s+([A-Z][a-zA-Z0-9\-_]+(?:\s+[A-Z0-9][a-zA-Z0-9\-_]+)*)[\s\?\.\!]*$|^\s*(?i:who\s+is)\s+(.+?)[\s\?\.\!]*$|(?i)\bexplain\s+the\s+([A-Z][a-zA-Z0-9\-_]+(?:\s+[A-Z0-9][a-zA-Z0-9\-_]+)*)\s+(?:mission|project|system|protocol|architecture|algorithm)\b'''
    new_regex = r'''^\s*(?i:what\s+is|tell\s+me\s+about|what\s+does\s+the\s+document\s+say\s+about)\s+([A-Z][a-zA-Z0-9\-_]+(?:\s+[A-Z0-9][a-zA-Z0-9\-_]+)*)[\s\?\.\!]*$|^\s*(?i:who\s+is)\s+(.+?)[\s\?\.\!]*$|(?i)\bexplain\s+the\s+([A-Z][a-zA-Z0-9\-_]+(?:\s+[A-Z0-9][a-zA-Z0-9\-_]+)*)\s+(?:mission|project|system|protocol|architecture|algorithm)\b'''
    
    content = content.replace(old_regex, new_regex)

    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/retrieval/QueryIntentRouter.kt', 'w', encoding='utf-8') as f:
        f.write(content)

    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/validation/CitationValidator.kt', 'r', encoding='utf-8') as f:
        content2 = f.read()

    old_bypass = '''        if (classification?.intent == com.siddu.offlinedocai.ai.retrieval.QueryIntent.EXHAUSTIVE_LIST || classification?.intent == com.siddu.offlinedocai.ai.retrieval.QueryIntent.FIND_ALL) {
            return true
        }'''
    new_bypass = '''        if (classification?.intent == com.siddu.offlinedocai.ai.retrieval.QueryIntent.EXHAUSTIVE_LIST || classification?.intent == com.siddu.offlinedocai.ai.retrieval.QueryIntent.FIND_ALL || classification?.intent == com.siddu.offlinedocai.ai.retrieval.QueryIntent.ENTITY_LOOKUP) {
            return true
        }'''
    content2 = content2.replace(old_bypass, new_bypass)
    
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/validation/CitationValidator.kt', 'w', encoding='utf-8') as f:
        f.write(content2)

if __name__ == '__main__':
    update_file()
