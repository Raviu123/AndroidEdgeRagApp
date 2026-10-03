import re

def update_file():
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/llm/GGUFLLM.kt', 'r', encoding='utf-8') as f:
        content = f.read()

    new_exhaustive = '''private fun synthesizeExhaustiveList(
        blocks: List<ParsedSourceBlock>,
        classification: QueryClassification,
        qLower: String,
        isBudgetLimited: Boolean,
        answerLength: AnswerLength = AnswerLength.SHORT
    ): String {
        val targetTerm = classification.targetTerm?.lowercase()?.trim() ?: extractTargetTopic(qLower)
        val extractedItems = extractListItems(blocks, targetTerm)

        if (extractedItems.isEmpty()) {
            return "I searched the provided document(s), but could not find items answering \\"${classification.rawQuery}\\"."
        }

        val topicLabel = if (targetTerm.isNotEmpty() && targetTerm !in STOP_WORDS) targetTerm else "items"
        val header = if (isBudgetLimited) {
            "The retrieved evidence contains the following $topicLabel (Note: Available context was budget-limited, additional items may exist in the document):"
        } else {
            "Based on the document, the $topicLabel are:"
        }

        // Deduplicate distinct items
        val distinctItems = mutableListOf<Pair<String, String>>()
        val seenNames = mutableSetOf<String>()
        val splitRegex = Regex("""\\s+[?"?"\\-]\\s+|\\s*:\\s+|\\s+(?:is\\s+used|is\\s+utilized|is\\s+defined|is\\s+commonly|is\\s+designed|is\\s+an?|requires|involves|uses|features)\\b""")

        for (item in extractedItems) {
            val full = item.first
            val clean = full.replace(Regex("""(?i)^(?:project|method|type|phase|step)\\s*\\d+[:\\-\\s]+"""), "").trim()
            val namePart = clean.split(splitRegex).first().trim()
            val label = if (namePart.length in 2..60) namePart else full
            val nameLower = label.lowercase()
            
            if (nameLower !in seenNames) {
                seenNames.add(nameLower)
                distinctItems.add(Pair(label, item.second))
            }
        }

        val itemsFormatted = if (answerLength == AnswerLength.SHORT) {
            distinctItems.joinToString("\\n") { "? ${it.first} ${it.second}" }
        } else {
            // For LONG, extract the full text for the item to get description, but we rely on extractListItems matching
            val longItems = distinctItems.mapIndexed { index, pair -> 
                val originalFullItem = extractedItems.find { it.second == pair.second }?.first ?: pair.first
                "? $originalFullItem ${pair.second}" 
            }
            longItems.joinToString("\\n\\n")
        }

        return "$header\\n\\n$itemsFormatted".trim()
    }'''

    pattern = re.compile(r'private fun synthesizeExhaustiveList\(.*?return "\$header\\n\\n\$itemsFormatted"\.trim\(\)\s*\}', re.DOTALL)
    content, count = pattern.subn(new_exhaustive, content)
    print(f"Replaced synthesizeExhaustiveList {count} times")
    
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/llm/GGUFLLM.kt', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    update_file()
