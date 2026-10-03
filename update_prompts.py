import re

def update_file():
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/llm/GGUFLLM.kt', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # We want to replace the old instructions in buildQwenPrompt and buildLlamaPrompt.
    
    # In buildQwenPrompt, replace the whole block of lengthInstruction
    # Same for buildLlamaPrompt
    
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
    
    # We will use regex to replace val lengthInstruction = when (answerLength) { ... }
    # up to the closing brace before val userInstruction.
    
    pattern = re.compile(r'val lengthInstruction = when \(answerLength\) \{.*?\}\s*val userInstruction', re.DOTALL)
    
    content, count = pattern.subn(new_inst + '\n        val userInstruction', content)
    print(f"Replaced lengthInstruction {count} times")
    
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/llm/GGUFLLM.kt', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    update_file()
