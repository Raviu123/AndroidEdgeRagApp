import re

def update_file():
    with open('app/src/test/java/com/siddu/offlinedocai/AnswerLengthControlTest.kt', 'r', encoding='utf-8') as f:
        content = f.read()

    # Short assertions for Qwen
    content = content.replace('assertTrue("Must contain Short prompt instruction", prompt.contains("Answer concisely in 1?"2 sentences when possible."))',
                              'assertTrue("Must contain Short prompt instruction", prompt.contains("- Give the minimum sufficient answer supported by the evidence."))')
    content = content.replace('assertTrue("Must contain user instruction reminder", prompt.contains("Instruction: Answer concisely in 1?"2 sentences."))',
                              'assertTrue("Must contain user instruction reminder", prompt.contains("Instruction: Answer concisely in 1?"2 sentences using the minimum sufficient grounded facts."))')
    
    # Short assertions for Llama
    content = content.replace('assertTrue(shortPrompt.contains("Answer concisely in 1?"2 sentences when possible."))',
                              'assertTrue(shortPrompt.contains("- Give the minimum sufficient answer supported by the evidence."))')
                              
    # Long assertions for Llama
    content = content.replace('assertTrue(longPrompt.contains("Give a substantially more detailed answer when the evidence contains additional relevant information."))',
                              'assertTrue(longPrompt.contains("- Give the most complete answer that is directly relevant to the question and supported by the evidence."))')

    with open('app/src/test/java/com/siddu/offlinedocai/AnswerLengthControlTest.kt', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Replaced Llama tests")

if __name__ == '__main__':
    update_file()
