import re

def update_file():
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/retrieval/QueryIntentRouter.kt', 'r', encoding='utf-8') as f:
        content = f.read()

    old_list_regex = '''    private val EXHAUSTIVE_LIST_REGEX = Regex(
        """(?i)\\b(?:list|give\\s+me|show|extract|name)\\s+(?:all(?:\\s+the)?|every|the\\s+complete\\s+list(?:\\s+of)?|the\\s+entire\\s+list(?:\\s+of)?)\\s+(.+?)(?:\\?|$|\\.)|\\bwhat\\s+are\\s+all(?:\\s+the)?\\s+(.+?)(?:\\?|$|\\.)|\\b(?:complete|entire)\\s+list\\s+of\\s+(.+?)(?:\\?|$|\\.)|\\blist\\s+every\\s+(.+?)\\s+(?:mentioned|in\\s+the\\s+document)?(?:\\?|$|\\.)|\\b(?:list|extract|name)\\s+(?:all\\s+)?(?:the\\s+)?([a-zA-Z0-9\\-_]+s)\\b(?:\\s+in\\s+the\\s+document|\\?|$|\\.)|\\bwhat\\s+are\\s+the\\s+([a-zA-Z0-9\\-_]+s)\\b(?:\\s+in\\s+the\\s+document|\\?|$|\\.)|\\btypes?\\s+of\\s+(.+?)(?:\\?|$|\\.)"""
    )'''

    new_list_regex = '''    private val EXHAUSTIVE_LIST_REGEX = Regex(
        """(?i)\\b(?:list|give\\s+me|show|extract|name)\\s+(?:all(?:\\s+the)?|every|the\\s+complete\\s+list(?:\\s+of)?|the\\s+entire\\s+list(?:\\s+of)?)\\s+(.+?)(?:\\?|$|\\.)|\\bwhat\\s+are\\s+all(?:\\s+the)?\\s+(.+?)(?:\\?|$|\\.)|\\b(?:complete|entire)\\s+list\\s+of\\s+(.+?)(?:\\?|$|\\.)|\\blist\\s+every\\s+(.+?)\\s+(?:mentioned|in\\s+the\\s+document)?(?:\\?|$|\\.)|\\b(?:list|extract|name)\\s+(?:all\\s+)?(?:the\\s+)?([a-zA-Z0-9\\-_]+s)\\b(?:\\s+in\\s+the\\s+document|\\?|$|\\.)|\\bwhat\\s+are\\s+the\\s+([a-zA-Z0-9\\-_]+s)\\b(?:\\s+in\\s+the\\s+document|\\?|$|\\.)|\\btypes?\\s+of\\s+(.+?)(?:\\?|$|\\.)|^(?:projects|skills|certifications|tools|materials|methods|techniques)$|\\bwhat\\s+(?:projects|skills|certifications|tools|materials)\\s+are\\s+mentioned\\b"""
    )'''

    content = content.replace(old_list_regex, new_list_regex)
    
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/retrieval/QueryIntentRouter.kt', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Replaced QueryIntentRouter")

if __name__ == '__main__':
    update_file()
