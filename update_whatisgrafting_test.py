import re

def update_file():
    with open('app/src/test/java/com/siddu/offlinedocai/AnswerLengthControlTest.kt', 'r', encoding='utf-8') as f:
        content = f.read()

    content = content.replace('assertTrue("Long must include additional grounded information", longAnswer.contains("cambium alignment", ignoreCase = true))',
                              'println("LONG ANSWER: " + longAnswer)\n        assertTrue("Long must include additional grounded information", longAnswer.contains("cambium alignment", ignoreCase = true))')

    with open('app/src/test/java/com/siddu/offlinedocai/AnswerLengthControlTest.kt', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    update_file()
