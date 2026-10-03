import re

def update_file():
    with open('app/src/test/java/com/siddu/offlinedocai/AnswerLengthControlTest.kt', 'r', encoding='utf-8') as f:
        content = f.read()

    # Short mode assertions
    content = content.replace(
        '''assertTrue("Must contain exact Short prompt instruction", prompt.contains("Answer concisely in 1?"2 sentences when possible."))''',
        '''assertTrue("Must contain exact Short prompt instruction", prompt.contains("- Give the minimum sufficient answer supported by the evidence."))'''
    )
    content = content.replace(
        '''assertTrue("Must contain minimum sufficient facts instruction", prompt.contains("Use the minimum sufficient set of grounded facts."))''',
        '''assertTrue("Must contain minimum sufficient facts instruction", prompt.contains("- Never invent facts."))'''
    )
    content = content.replace(
        '''assertFalse("Must NOT contain Long prompt instruction", prompt.contains("Give a substantially more detailed answer"))''',
        '''assertFalse("Must NOT contain Long prompt instruction", prompt.contains("- Give the most complete answer"))'''
    )
    
    # Long mode assertions
    content = content.replace(
        '''assertTrue("Must contain exact Long prompt instruction", prompt.contains("Give a substantially more detailed answer when the evidence contains additional relevant information."))''',
        '''assertTrue("Must contain exact Long prompt instruction", prompt.contains("- Give the most complete answer that is directly relevant to the question and supported by the evidence."))'''
    )
    content = content.replace(
        '''assertTrue("Must contain distinct factual points instruction", prompt.contains("Include distinct relevant factual points from the retrieved evidence."))''',
        '''assertTrue("Must contain distinct factual points instruction", prompt.contains("- Cover additional distinct relevant evidence."))'''
    )
    content = content.replace(
        '''assertFalse("Must NOT contain Short prompt instruction", prompt.contains("Answer concisely in 1?"2 sentences when possible."))''',
        '''assertFalse("Must NOT contain Short prompt instruction", prompt.contains("- Give the minimum sufficient answer supported by the evidence."))'''
    )
    
    # Llama
    content = content.replace(
        '''assertTrue(shortPrompt.contains("Answer concisely in 1?"2 sentences when possible."))''',
        '''assertTrue(shortPrompt.contains("- Give the minimum sufficient answer supported by the evidence."))'''
    )
    content = content.replace(
        '''assertTrue(longPrompt.contains("Give a substantially more detailed answer"))''',
        '''assertTrue(longPrompt.contains("- Give the most complete answer"))'''
    )

    with open('app/src/test/java/com/siddu/offlinedocai/AnswerLengthControlTest.kt', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    update_file()
