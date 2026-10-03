import re

def update_file():
    with open('app/src/test/java/com/siddu/offlinedocai/AnswerLengthControlTest.kt', 'r', encoding='utf-8') as f:
        content = f.read()

    # Short Qwen
    content = re.sub(
        r'assertTrue\("Must contain exact Short prompt instruction", prompt\.contains\("Answer concisely in 1?"2 sentences when possible\."\)\)',
        r'assertTrue("Must contain exact Short prompt instruction", prompt.contains("- Give the minimum sufficient answer supported by the evidence."))',
        content
    )
    
    # Long Qwen
    content = re.sub(
        r'assertTrue\("Must contain exact Long prompt instruction", prompt\.contains\("Give a substantially more detailed answer when the evidence contains additional relevant information\."\)\)',
        r'assertTrue("Must contain exact Long prompt instruction", prompt.contains("- Give the most complete answer that is directly relevant to the question and supported by the evidence."))',
        content
    )

    with open('app/src/test/java/com/siddu/offlinedocai/AnswerLengthControlTest.kt', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Replaced tests")

if __name__ == '__main__':
    update_file()
