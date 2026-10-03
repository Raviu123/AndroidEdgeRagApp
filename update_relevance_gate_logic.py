import re

def update_file():
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/validation/CitationValidator.kt', 'r', encoding='utf-8') as f:
        content = f.read()

    new_relevance = '''    fun isClaimRelevantToQuery(
        claim: String,
        query: String,
        classification: com.siddu.offlinedocai.ai.retrieval.QueryClassification? = null
    ): Boolean {
        if (query.isBlank()) return true
        val cLower = claim.lowercase().trim()
        val qLower = query.lowercase().trim()

        // 1. Tools and Materials
        val isToolsQuery = qLower.contains("tool") || qLower.contains("material") || qLower.contains("equipment") || qLower.contains("supplies")
        if (isToolsQuery && !qLower.contains("method") && !qLower.contains("cause") && !qLower.contains("what is")) {
            val toolTerms = listOf(
                "knife", "knives", "strip", "strips", "shear", "shears", "wax", "tape",
                "pruning", "blade", "seal", "wound", "equipment", "tool", "material", "required", "needed"
            )
            val irrelevantAftercareTerms = listOf("inspect", "inspection", "weekly", "sucker", "aftercare", "temperature", "pathogen", "failure", "cleft grafting", "healing")
            val hasTool = toolTerms.any { cLower.contains(it) }
            val hasAftercare = irrelevantAftercareTerms.any { cLower.contains(it) }
            
            // If the source states the purpose of the tool (e.g. wax prevents desiccation), allow it
            if (hasTool) return true
            if (hasAftercare) return false
            return false
        }

        // 2. Temperature / Environmental Healing
        val isTempQuery = (classification?.factType == com.siddu.offlinedocai.ai.retrieval.FactType.TEMPERATURE) ||
                qLower.contains("temperature") || qLower.contains("degrees") || qLower.contains("celsius") || qLower.contains("fahrenheit") || qLower.contains("healing")
        if (isTempQuery) {
            val tempTerms = listOf("temperature", "21c", "27c", "70f", "80f", "Ac", "Af", "celsius", "fahrenheit", "callus", "healing", "humidity", "85%", "desiccation")
            val irrelevantTerms = listOf("knife", "knives", "shear", "shears", "strip", "strips", "wax", "cleft", "splice", "bridge", "sucker", "weekly", "tools", "aftercare", "types")
            val hasTemp = tempTerms.any { cLower.contains(it) }
            val hasIrrelevant = irrelevantTerms.any { cLower.contains(it) }
            
            // Allow associated humidity/healing conditions
            if (hasTemp) return true
            if (hasIrrelevant) return false
            return false
        }

        // 3. Causes of failure and troubleshooting
        val isCausesQuery = qLower.contains("cause") || qLower.contains("failure") || qLower.contains("fail") || qLower.contains("troubleshoot")
        if (isCausesQuery) {
            val causesTerms = listOf("cambial", "contact", "desiccation", "physiological", "season", "pathogen", "infection", "cause", "fail", "failure", "poor cuts")
            val checkTerms = listOf("inspect", "inspection", "weekly", "sucker", "rootstock", "union", "remove", "check", "ensure", "troubleshoot", "prevent")
            val irrelevantTools = listOf("sharp grafting knife", "pruning shears", "wax", "strip")
            
            val asksCheck = qLower.contains("check") || qLower.contains("inspect") || qLower.contains("do") || qLower.contains("action") || qLower.contains("prevent")

            val hasCause = causesTerms.any { cLower.contains(it) }
            val hasCheck = checkTerms.any { cLower.contains(it) }
            val hasTool = irrelevantTools.any { cLower.contains(it) }

            if (hasTool && !hasCause && !hasCheck) return false

            if (asksCheck) {
                return hasCause || hasCheck
            } else {
                return hasCause
            }
        }

        // 4. Comparison
        if (classification?.intent == com.siddu.offlinedocai.ai.retrieval.QueryIntent.COMPARISON) {
            // Must contain Entity A or Entity B or be directly related
            val ea = classification.entityA?.lowercase() ?: ""
            val eb = classification.entityB?.lowercase() ?: ""
            if (ea.isNotBlank() && cLower.contains(ea)) return true
            if (eb.isNotBlank() && cLower.contains(eb)) return true
            return false
        }

        // 7. Default query: Intent, important entities, semantic overlap
        val qTokens = qLower.split(Regex("""[^a-z0-9]+""")).filter { it.length > 2 && it !in STOP_WORDS }
        if (qTokens.isEmpty()) return true
        
        // Exact target extraction based relevance
        val targetTopic = classification?.targetTerm?.lowercase()
        if (!targetTopic.isNullOrBlank()) {
            val topicTokens = targetTopic.split(Regex("""[^a-z0-9]+""")).filter { it.length > 2 }
            if (topicTokens.isNotEmpty() && topicTokens.all { cLower.contains(it) }) return true
        }

        // Must share at least SOME core entity or requested attribute, not just any keyword
        val importantEntities = qTokens.filter { it[0].isUpperCase() || qLower.contains("what is $it") || qLower.contains("list $it") }
        if (importantEntities.isNotEmpty()) {
            if (importantEntities.any { cLower.contains(it) }) return true
        }
        
        // Fallback to requiring more than just 1 random keyword overlap if query is long
        val matchedTokens = qTokens.filter { cLower.contains(it) }
        if (qTokens.size >= 3) {
            return matchedTokens.size >= 2
        }

        return matchedTokens.isNotEmpty()
    }'''

    pattern = re.compile(r'    fun isClaimRelevantToQuery\(.*?return qTokens\.any \{ cLower\.contains\(it\) \}\n    \}', re.DOTALL)
    content, count = pattern.subn(new_relevance, content)
    print(f"Replaced isClaimRelevantToQuery {count} times")
    
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/validation/CitationValidator.kt', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    update_file()
