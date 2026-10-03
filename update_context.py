import re

def update_file():
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/llm/GGUFLLM.kt', 'r', encoding='utf-8') as f:
        content = f.read()
    if 'import com.siddu.offlinedocai.ai.validation.FallbackLevel' not in content:
        content = content.replace('import com.siddu.offlinedocai.ai.validation.CitationValidator', 'import com.siddu.offlinedocai.ai.validation.CitationValidator\nimport com.siddu.offlinedocai.ai.validation.FallbackLevel')
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/llm/GGUFLLM.kt', 'w', encoding='utf-8') as f:
        f.write(content)

    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/llm/GGUFLLM.kt', 'r', encoding='utf-8') as f:
        content = f.read()

    # We need to insert a function `prepareFilteredContext` inside GGUFLLM class.
    # And replace `effectiveContext` definition.

    filtered_context_code = '''
    private fun prepareFilteredContext(
        blocks: List<ParsedSourceBlock>,
        classification: QueryClassification,
        qTrimmed: String,
        answerLength: AnswerLength
    ): String {
        if (blocks.isEmpty()) return "No relevant passages were found."
        
        val isExhaustive = classification.intent == QueryIntent.EXHAUSTIVE_LIST || classification.intent == QueryIntent.FIND_ALL
        
        if (isExhaustive) {
            val targetTerm = classification.targetTerm?.lowercase()?.trim() ?: extractTargetTopic(qTrimmed.lowercase())
            val extractedItems = extractListItems(blocks, targetTerm)
            if (extractedItems.isEmpty()) return "No relevant passages were found."
            
            // Deduplicate items based on name
            val distinctItems = mutableListOf<Pair<String, String>>()
            val seenNames = mutableSetOf<String>()
            for (item in extractedItems) {
                val nameLower = item.first.lowercase().trim()
                if (nameLower !in seenNames) {
                    seenNames.add(nameLower)
                    distinctItems.add(item)
                }
            }
            
            val sb = java.lang.StringBuilder()
            for ((index, item) in distinctItems.withIndex()) {
                sb.append("${index + 1}. ${item.first}")
                if (answerLength == AnswerLength.LONG && item.second.isNotBlank()) {
                    sb.append(" - ${item.second}")
                }
                sb.append("\\n")
            }
            return sb.toString()
        } else {
            // General fact extraction, relevance gate, and deduplication
            val relevantFacts = mutableListOf<String>()
            val seenFacts = mutableSetOf<String>()
            
            for (block in blocks) {
                val sentences = block.text.split(Regex("""(?<!\\b(?:e\\.g|i\\.e|mr|dr|ms|mrs|vs|etc|no|fig|pg|al|\\d))(?<=[.!?])\\s+(?=[A-Z0-9\\-\\*•])|\\n+"""))
                for (sentence in sentences) {
                    val s = sentence.trim()
                    if (s.length < 5) continue
                    if (CitationValidator.isHeadingOrTitle(s)) continue
                    
                    // Relevance gate
                    if (CitationValidator.isClaimRelevantToQuery(s, qTrimmed, classification)) {
                        val sLower = s.lowercase()
                        if (sLower !in seenFacts) {
                            // Semantic deduplication (basic substring/overlap check for now)
                            var isDuplicate = false
                            for (existing in seenFacts) {
                                if (existing.contains(sLower) || sLower.contains(existing)) {
                                    isDuplicate = true
                                    break
                                }
                            }
                            if (!isDuplicate) {
                                seenFacts.add(sLower)
                                relevantFacts.add("[Source ${block.sourceId}, Page ${block.page}] $s")
                            }
                        }
                    }
                }
            }
            
            if (relevantFacts.isEmpty()) return "No relevant passages were found."
            
            // Limit coverage based on Short/Long
            val finalFacts = if (answerLength == AnswerLength.SHORT) {
                relevantFacts.take(maxOf(1, relevantFacts.size / 3)) // Take a subset for SHORT
            } else {
                relevantFacts
            }
            
            return finalFacts.joinToString("\\n")
        }
    }
'''

    # Find the right place to insert this function.
    # Let's insert it before `private fun synthesizePageLookup`
    
    if 'private fun prepareFilteredContext' not in content:
        content = content.replace('private fun synthesizePageLookup', filtered_context_code + '\n    private fun synthesizePageLookup')
    
    # Replace the effectiveContext logic
    old_logic = '''val effectiveContext = if (contextText == "[NO_EVIDENCE_FOUND]" || contextText.isBlank() || contextText.startsWith("[PAGE_NOT_FOUND:") || contextText.startsWith("[PAGE_RANGE_NOT_FOUND:")) {
                            "No relevant passages were found in the uploaded document."
                        } else {
                            contextText
                        }'''
                        
    new_logic = '''val effectiveContext = if (contextText == "[NO_EVIDENCE_FOUND]" || contextText.isBlank() || contextText.startsWith("[PAGE_NOT_FOUND:") || contextText.startsWith("[PAGE_RANGE_NOT_FOUND:")) {
                            "No relevant passages were found in the uploaded document."
                        } else {
                            prepareFilteredContext(blocks, classification, qTrimmed, answerLength)
                        }'''
                        
    content = content.replace(old_logic, new_logic)

    # Also add strict validation of Qwen output
    # old Qwen call block
    qwen_call_pattern = re.compile(r'(val neuralOutput = llamaRunner\.inference\(.*?temperature = DEFAULT_TEMPERATURE\s*\)\s*Log\.i\("RAG_DEBUG", "RAG_DEBUG_08_NATIVE_DONE: rawOutputLen=\$\{neuralOutput\.length\}"\)\s*val cleanOutput = CitationValidator\.cleanAndRenumberList\(neuralOutput.*?\.trim\(\)\)\s*if \(cleanOutput\.isNotBlank\(\)\) \{\s*rawAnswer = cleanOutput\s*usedNative = true\s*currentEngineState = EngineState\.GGUF_NATIVE\s*Log\.i\(TAG, "AI_ENGINE=GGUF_NATIVE MODEL=\$\{activeModel\.name\}"\)\s*\})', re.DOTALL)
    
    match = qwen_call_pattern.search(content)
    if match:
        old_qwen_call = match.group(1)
        new_qwen_call = old_qwen_call.replace('''if (cleanOutput.isNotBlank()) {
                            rawAnswer = cleanOutput
                            usedNative = true
                            currentEngineState = EngineState.GGUF_NATIVE
                            Log.i(TAG, "AI_ENGINE=GGUF_NATIVE MODEL=${activeModel.name}")
                        }''', '''if (cleanOutput.isNotBlank()) {
                            val tempValidation = CitationValidator.validate(cleanOutput, blocks, qTrimmed, isBudgetLimited, answerLength)
                            val claims = CitationValidator.extractClaims(cleanOutput)
                            val hasDuplicateClaims = claims.size > claims.map { it.lowercase() }.toSet().size
                            
                            val isExhaustive = classification.intent == QueryIntent.EXHAUSTIVE_LIST || classification.intent == QueryIntent.FIND_ALL
                            var missingItems = false
                            if (isExhaustive) {
                                val targetTerm = classification.targetTerm?.lowercase()?.trim() ?: extractTargetTopic(qLower)
                                val extractedItems = extractListItems(blocks, targetTerm)
                                // Extract list names from cleanOutput
                                val outputLines = cleanOutput.lines()
                                var extractedCount = 0
                                for (item in extractedItems) {
                                    val name = item.first.lowercase()
                                    if (cleanOutput.lowercase().contains(name)) extractedCount++
                                }
                                if (extractedCount < extractedItems.size && extractedItems.isNotEmpty()) {
                                    missingItems = true
                                }
                            }
                            
                            var hasMalformedLists = false
                            if (cleanOutput.contains(Regex("""\d+\\.\\s*\\d+\\.""")) || cleanOutput.contains(Regex("""\\n\\d+\\.\\s*\\n"""))) {
                                hasMalformedLists = true
                            }
                            
                            var hasRawHeadings = false
                            for (line in neuralOutput.lines()) {
                                if (CitationValidator.isHeadingOrTitle(line)) {
                                    hasRawHeadings = true
                                    break
                                }
                            }

                            if (tempValidation.fallbackLevel != FallbackLevel.L1_FULLY_SUPPORTED || 
                                hasDuplicateClaims || 
                                missingItems || 
                                hasMalformedLists || 
                                hasRawHeadings) {
                                Log.w(TAG, "Native Qwen output failed strict validation (L1=${tempValidation.fallbackLevel == FallbackLevel.L1_FULLY_SUPPORTED}, dup=$hasDuplicateClaims, missingItems=$missingItems, malformed=$hasMalformedLists, headings=$hasRawHeadings), falling back to deterministic synthesis.")
                                usedNative = false
                            } else {
                                rawAnswer = tempValidation.validatedAnswer
                                usedNative = true
                                currentEngineState = EngineState.GGUF_NATIVE
                                Log.i(TAG, "AI_ENGINE=GGUF_NATIVE MODEL=${activeModel.name}")
                            }
                        }''')
        content = content.replace(old_qwen_call, new_qwen_call)
        print("Replaced Qwen call with strict validation")
    else:
        print("Could not find Qwen call block")
        
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/llm/GGUFLLM.kt', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    update_file()
