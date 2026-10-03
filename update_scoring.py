import re

def update_file():
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/llm/GGUFLLM.kt', 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the scoring loop
    old_scoring = '''                    if (topicWords.isNotEmpty()) {
                        val hasAllTopicWords = topicWords.all { kw -> matchesWordOrSynonym(sLower, kw) }
                        if (hasAllTopicWords) {
                            score += 25.0
                            if (sLower.contains(" is a ") || sLower.contains(" is an ") || sLower.contains(" refers to ") || sLower.contains(" consists of ") || sLower.contains(" is defined as ")) {
                                score += 35.0
                                isDefinitionMatch = true
                            }
                        }
                    }'''
    
    new_scoring = '''                    if (topicWords.isNotEmpty()) {
                        val hasAllTopicWords = topicWords.all { kw -> matchesWordOrSynonym(sLower, kw) }
                        if (hasAllTopicWords) {
                            score += 25.0
                            if (sLower.contains(" is a ") || sLower.contains(" is an ") || sLower.contains(" refers to ") || sLower.contains(" consists of ") || sLower.contains(" is defined as ")) {
                                score += 35.0
                                isDefinitionMatch = true
                            }
                        }
                    }
                    if (targetTopic.isNotEmpty() && targetTopic.length > 2) {
                        if (sLower.startsWith(targetTopic) || sLower.contains("$targetTopic (") || sLower.contains("$targetTopic:")) {
                            score += 40.0
                        }
                    }'''
    
    content = content.replace(old_scoring, new_scoring)

    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/llm/GGUFLLM.kt', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Replaced scoring")

if __name__ == '__main__':
    update_file()
