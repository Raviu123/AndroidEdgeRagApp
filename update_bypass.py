import re

def update_file():
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/validation/CitationValidator.kt', 'r', encoding='utf-8') as f:
        content = f.read()

    new_bypass = '''        if (classification?.intent == com.siddu.offlinedocai.ai.retrieval.QueryIntent.EXHAUSTIVE_LIST || classification?.intent == com.siddu.offlinedocai.ai.retrieval.QueryIntent.FIND_ALL) {
            return true
        }
        
        // Exact target extraction based relevance'''

    content = content.replace('// Exact target extraction based relevance', new_bypass)
    
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/validation/CitationValidator.kt', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Replaced CitationValidator bypass")

if __name__ == '__main__':
    update_file()
