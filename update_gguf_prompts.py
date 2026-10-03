import re

def update_file():
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/llm/GGUFLLM.kt', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Update prompts correctly
    new_inst = '''val lengthInstruction = when (answerLength) {
            AnswerLength.SHORT -> """
- Give the minimum sufficient answer supported by the evidence.
- Never invent facts.
- Never use outside knowledge.
- Never repeat a fact.
- Never include irrelevant evidence.
- Do not create empty list items.
- Do not output headings as answers.
- For lists, preserve all requested items.
- Stop when the relevant evidence has been adequately covered.
""".trimIndent()
            AnswerLength.LONG -> """
- Give the most complete answer that is directly relevant to the question and supported by the evidence.
- Never invent facts.
- Never use outside knowledge.
- Never repeat a fact.
- Never include irrelevant evidence.
- Do not create empty list items.
- Do not output headings as answers.
- For lists, preserve all requested items.
- Expand items with grounded descriptions only when available.
- Cover additional distinct relevant evidence.
- Stop when the relevant evidence has been adequately covered.
""".trimIndent()
        }'''
    
    # We replace the lengthInstruction block, making sure to ONLY match until the close brace of the when statement
    # The when block ends with """.trimIndent() \n        }
    pattern = re.compile(r'val lengthInstruction = when \(answerLength\) \{.*?\}\.trimIndent\(\)\n        \}', re.DOTALL)
    content, count = pattern.subn(new_inst, content)
    print(f"Replaced lengthInstruction {count} times")

    # 2. Add validation to Qwen output
    # Find the native execution block
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

                            if (tempValidation.fallbackLevel != com.siddu.offlinedocai.ai.validation.FallbackLevel.L1_FULLY_SUPPORTED || 
                                hasDuplicateClaims || 
                                missingItems || 
                                hasMalformedLists || 
                                hasRawHeadings) {
                                Log.w(TAG, "Native Qwen output failed strict validation (L1=${tempValidation.fallbackLevel == com.siddu.offlinedocai.ai.validation.FallbackLevel.L1_FULLY_SUPPORTED}, dup=$hasDuplicateClaims, missingItems=$missingItems, malformed=$hasMalformedLists, headings=$hasRawHeadings), falling back to deterministic synthesis.")
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
