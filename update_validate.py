import re

def update_file():
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/validation/CitationValidator.kt', 'r', encoding='utf-8') as f:
        content = f.read()

    # Update validate signature
    old_validate = '''    fun validate(
        answer: String,
        blocks: List<ParsedSourceBlock>,
        query: String = "",
        isBudgetLimited: Boolean = false,
        answerLength: com.siddu.offlinedocai.ai.llm.AnswerLength = com.siddu.offlinedocai.ai.llm.AnswerLength.SHORT
    ): FullValidationResult {'''

    new_validate = '''    fun validate(
        answer: String,
        blocks: List<ParsedSourceBlock>,
        query: String = "",
        isBudgetLimited: Boolean = false,
        answerLength: com.siddu.offlinedocai.ai.llm.AnswerLength = com.siddu.offlinedocai.ai.llm.AnswerLength.SHORT,
        classification: com.siddu.offlinedocai.ai.retrieval.QueryClassification? = null
    ): FullValidationResult {'''

    content = content.replace(old_validate, new_validate)
    
    # Update validateClaimSupport call in validate
    old_call = '''val result = validateClaimSupport(claim, candidateBlocks, blocks, validatedCitations, query)'''
    new_call = '''val result = validateClaimSupport(claim, candidateBlocks, blocks, validatedCitations, query, classification)'''
    content = content.replace(old_call, new_call)
    
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/validation/CitationValidator.kt', 'w', encoding='utf-8') as f:
        f.write(content)

    # Update GGUFLLM.kt to pass classification
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/llm/GGUFLLM.kt', 'r', encoding='utf-8') as f:
        gguf = f.read()
        
    old_gguf_call1 = '''val tempValidation = CitationValidator.validate(cleanOutput, blocks, qTrimmed, isBudgetLimited, answerLength)'''
    new_gguf_call1 = '''val tempValidation = CitationValidator.validate(cleanOutput, blocks, qTrimmed, isBudgetLimited, answerLength, classification)'''
    gguf = gguf.replace(old_gguf_call1, new_gguf_call1)

    old_gguf_call2 = '''val validationResult = CitationValidator.validate(rawAnswer, blocks, qTrimmed, isBudgetLimited, answerLength)'''
    new_gguf_call2 = '''val validationResult = CitationValidator.validate(rawAnswer, blocks, qTrimmed, isBudgetLimited, answerLength, classification)'''
    gguf = gguf.replace(old_gguf_call2, new_gguf_call2)
    
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/llm/GGUFLLM.kt', 'w', encoding='utf-8') as f:
        f.write(gguf)

if __name__ == '__main__':
    update_file()
