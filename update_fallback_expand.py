import re

def update_file():
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/llm/GGUFLLM.kt', 'r', encoding='utf-8') as f:
        content = f.read()

    old_code = '''        val mainAnswer = "${top.text} ${formatCitation(top.block)}"'''
    
    new_code = '''        var expandedMain = top.text
        if (classification?.intent == com.siddu.offlinedocai.ai.retrieval.QueryIntent.ENTITY_LOOKUP || classification?.intent == com.siddu.offlinedocai.ai.retrieval.QueryIntent.CONCEPTUAL_QUERY) {
            val sents = top.block.text.split(SENTENCE_SPLIT_REGEX).map { it.trim() }
            val idx = sents.indexOf(top.text)
            if (idx != -1 && idx < sents.size - 1) {
                for (i in idx + 1 until minOf(idx + 3, sents.size)) {
                    if (sents[i].length > 15 && !com.siddu.offlinedocai.ai.validation.CitationValidator.isHeadingOrTitle(sents[i])) {
                        expandedMain = "$expandedMain ${sents[i]}"
                    }
                }
            }
        }
        val mainAnswer = "$expandedMain ${formatCitation(top.block)}"'''

    content = content.replace(old_code, new_code)

    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/llm/GGUFLLM.kt', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    update_file()
