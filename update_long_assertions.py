import re

def update_file():
    with open('app/src/test/java/com/siddu/offlinedocai/AnswerLengthControlTest.kt', 'r', encoding='utf-8') as f:
        content = f.read()

    # Long Qwen
    content = re.sub(
        r'assertTrue\("Must contain user instruction reminder", prompt\.contains\("Instruction: Provide a detailed answer covering all distinct relevant grounded facts without repeating information\."\)\)',
        r'assertTrue("Must contain user instruction reminder", prompt.contains("Instruction: Give the most complete answer that is directly relevant to the question and supported by the evidence."))',
        content
    )
    content = re.sub(
        r'assertTrue\("Must contain distinct factual points instruction", prompt\.contains\([^)]+\)\)',
        r'// Removed distinct factual points instruction',
        content
    )

    with open('app/src/test/java/com/siddu/offlinedocai/AnswerLengthControlTest.kt', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    update_file()
