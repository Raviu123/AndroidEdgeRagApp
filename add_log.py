import re

def update_file():
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/llm/GGUFLLM.kt', 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the candidates add logic
    old_code = '''candidates.add(ScoredSentence(block, sent, score))'''
    new_code = '''if (qLower.contains("phase a")) { android.util.Log.i("RAG_DEBUG", "Sent: [$sLower] Target: [$targetTopic] Score: $score") }
                candidates.add(ScoredSentence(block, sent, score))'''

    content = content.replace(old_code, new_code)

    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/llm/GGUFLLM.kt', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    update_file()
