import re

def update_file():
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/llm/GGUFLLM.kt', 'r', encoding='utf-8') as f:
        content = f.read()

    old_regex = r'''what\s+is|what|list\s+all\s+the|list\s+all|list\s+the|list\s+every|list|show\s+all|show\s+me|show|extract\s+all|extract|give\s+me\s+all|give\s+me|name\s+all|name|tell\s+me\s+about\s+the|tell\s+me\s+about|which)\s+"""'''
    new_regex = r'''what\s+is|what|list\s+all\s+the|list\s+all|list\s+the|list\s+every|list|show\s+all|show\s+me|show|extract\s+all|extract|give\s+me\s+all|give\s+me|name\s+all|name|tell\s+me\s+about\s+the|tell\s+me\s+about|what\s+does\s+the\s+document\s+say\s+about|which)\s+"""'''
    content = content.replace(old_regex, new_regex)

    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/llm/GGUFLLM.kt', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    update_file()
