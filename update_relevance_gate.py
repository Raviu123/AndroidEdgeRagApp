import re

def update_file():
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/llm/GGUFLLM.kt', 'r', encoding='utf-8') as f:
        content = f.read()

    # We need to add the relevance gate
    old_validation = '''if (tempValidation.fallbackLevel != com.siddu.offlinedocai.ai.validation.FallbackLevel.L1_FULLY_SUPPORTED || 
                                hasDuplicateClaims || 
                                missingItems || 
                                hasMalformedLists || 
                                hasRawHeadings) {'''
    
    new_validation = '''var hasIrrelevantClaims = false
                            for (claim in claims) {
                                if (!CitationValidator.isClaimRelevantToQuery(claim, qLower)) {
                                    hasIrrelevantClaims = true
                                    break
                                }
                            }
                            
                            // Check AnswerLength semantics
                            var lengthFailed = false
                            if (answerLength == AnswerLength.SHORT && claims.size > 3 && !isExhaustive) {
                                lengthFailed = true
                            }

                            if (tempValidation.fallbackLevel != com.siddu.offlinedocai.ai.validation.FallbackLevel.L1_FULLY_SUPPORTED || 
                                hasDuplicateClaims || 
                                missingItems || 
                                hasMalformedLists || 
                                hasRawHeadings ||
                                hasIrrelevantClaims ||
                                lengthFailed) {'''
    
    content = content.replace(old_validation, new_validation)
    
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/llm/GGUFLLM.kt', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Replaced validation")

if __name__ == '__main__':
    update_file()
