import re

def update_file():
    with open('app/src/test/java/com/siddu/offlinedocai/AnswerLengthControlTest.kt', 'r', encoding='utf-8') as f:
        content = f.read()

    # Short Qwen
    content = re.sub(
        r'assertTrue\("Must contain exact Short prompt instruction", prompt\.contains\([^)]+\)\)',
        r'assertTrue("Must contain exact Short prompt instruction", prompt.contains("- Give the minimum sufficient answer supported by the evidence."))',
        content
    )
    content = re.sub(
        r'assertTrue\("Must contain user instruction reminder", prompt\.contains\("Instruction: Answer concisely in 1[^\"]+2 sentences using the minimum sufficient grounded facts\."\)\)',
        r'assertTrue("Must contain user instruction reminder", prompt.contains("Instruction: Give the minimum sufficient answer supported by the evidence."))',
        content
    )
    content = re.sub(
        r'assertFalse\("Must NOT contain Short prompt instruction", prompt\.contains\("Answer concisely in 1[^\"]+2 sentences when possible\."\)\)',
        r'assertFalse("Must NOT contain Short prompt instruction", prompt.contains("- Give the minimum sufficient answer supported by the evidence."))',
        content
    )

    # Short Llama
    content = re.sub(
        r'assertTrue\(shortPrompt\.contains\("Answer concisely in 1[^\"]+2 sentences when possible\."\)\)',
        r'assertTrue(shortPrompt.contains("- Give the minimum sufficient answer supported by the evidence."))',
        content
    )

    with open('app/src/test/java/com/siddu/offlinedocai/AnswerLengthControlTest.kt', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    update_file()
