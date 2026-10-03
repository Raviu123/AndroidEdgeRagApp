import re

def update_file():
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/llm/GGUFLLM.kt', 'r', encoding='utf-8') as f:
        content = f.read()

    content = content.replace('val isEligible = c.isTopicRelevant && !c.isKeywordOnly && c.score >= 15.0 &&',
                              'println("CANDIDATE: " + c.text + " SCORE: " + c.score + " RELEVANT: " + c.isTopicRelevant + " KW: " + c.isKeywordOnly + " VAL: " + CitationValidator.isClaimRelevantToQuery(c.text, qTrimmed, classification))\n                val isEligible = c.isTopicRelevant && !c.isKeywordOnly && c.score >= 15.0 &&')

    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/llm/GGUFLLM.kt', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    update_file()
