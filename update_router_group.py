import re

def update_file():
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/retrieval/QueryIntentRouter.kt', 'r', encoding='utf-8') as f:
        content = f.read()

    old_regex = '''|^(?:projects|skills|certifications|tools|materials|methods|techniques)$|\\bwhat\\s+(?:projects|skills|certifications|tools|materials)\\s+are\\s+mentioned\\b'''
    new_regex = '''|^((?:projects|skills|certifications|tools|materials|methods|techniques))$|\\bwhat\\s+((?:projects|skills|certifications|tools|materials))\\s+are\\s+mentioned\\b'''

    content = content.replace(old_regex, new_regex)
    
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/retrieval/QueryIntentRouter.kt', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Replaced QueryIntentRouter regex capture group")

if __name__ == '__main__':
    update_file()
