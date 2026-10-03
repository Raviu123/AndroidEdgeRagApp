import re

def update_file():
    with open('app/src/main/kotlin/com/siddu/offlinedocai/domain/repository/ChatRepository.kt', 'r', encoding='utf-8') as f:
        content = f.read()
        
    new_inst = '''val lengthInstruction = when (answerLength) {
            AnswerLength.SHORT -> """
- Give the minimum sufficient answer supported by the evidence.
- Never invent facts.
- Never use outside knowledge.
- Never repeat a fact.
- Never include irrelevant evidence.
- Do not create empty list items.
- Do not output headings as answers.
- For lists, preserve all requested items.
- Stop when the relevant evidence has been adequately covered.
""".trimIndent()
            AnswerLength.LONG -> """
- Give the most complete answer that is directly relevant to the question and supported by the evidence.
- Never invent facts.
- Never use outside knowledge.
- Never repeat a fact.
- Never include irrelevant evidence.
- Do not create empty list items.
- Do not output headings as answers.
- For lists, preserve all requested items.
- Expand items with grounded descriptions only when available.
- Cover additional distinct relevant evidence.
- Stop when the relevant evidence has been adequately covered.
""".trimIndent()
        }'''
    
    pattern = re.compile(r'val lengthInstruction = when \(answerLength\) \{.*?\}\s*return """', re.DOTALL)
    
    content, count = pattern.subn(new_inst + '\n        return """', content)
    print(f"Replaced ChatRepository lengthInstruction {count} times")
    
    with open('app/src/main/kotlin/com/siddu/offlinedocai/domain/repository/ChatRepository.kt', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    update_file()
