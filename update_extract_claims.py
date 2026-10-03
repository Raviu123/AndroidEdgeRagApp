import re

def update_file():
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/validation/CitationValidator.kt', 'r', encoding='utf-8') as f:
        content = f.read()

    new_filter = '''.filter { it.isNotEmpty() && it.any { c -> c.isLetterOrDigit() } }
                    .filter { stripCitations(it).isNotBlank() }'''
                    
    content = content.replace('.filter { it.isNotEmpty() && it.any { c -> c.isLetterOrDigit() } }', new_filter)
    
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/validation/CitationValidator.kt', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Replaced extractClaims")

if __name__ == '__main__':
    update_file()
