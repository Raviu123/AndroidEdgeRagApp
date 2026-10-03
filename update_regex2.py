import re

def update_file():
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/retrieval/QueryIntentRouter.kt', 'r', encoding='utf-8') as f:
        content = f.read()

    old_regex = r'''^\s*(?i:what\s+is|tell\s+me\s+about|what\s+does\s+the\s+document\s+say\s+about)\s+([A-Z][a-zA-Z0-9\-_]+(?:\s+[A-Z0-9][a-zA-Z0-9\-_]+)*)[\s\?\.\!]*$|^\s*(?i:who\s+is)\s+(.+?)[\s\?\.\!]*$|(?i)\bexplain\s+the\s+([A-Z][a-zA-Z0-9\-_]+(?:\s+[A-Z0-9][a-zA-Z0-9\-_]+)*)\s+(?:mission|project|system|protocol|architecture|algorithm)\b'''
    new_regex = r'''^\s*(?i:what\s+is|tell\s+me\s+about|what\s+does\s+the\s+document\s+say\s+about)\s+([A-Z][a-zA-Z0-9\-_]+(?:\s+[A-Z0-9][a-zA-Z0-9\-_]*)*)[\s\?\.\!]*$|^\s*(?i:who\s+is)\s+(.+?)[\s\?\.\!]*$|(?i)\bexplain\s+the\s+([A-Z][a-zA-Z0-9\-_]+(?:\s+[A-Z0-9][a-zA-Z0-9\-_]*)*)\s+(?:mission|project|system|protocol|architecture|algorithm)\b'''
    
    content = content.replace(old_regex, new_regex)

    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/retrieval/QueryIntentRouter.kt', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    update_file()
