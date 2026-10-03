import re

def update_file():
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/llm/GGUFLLM.kt', 'r', encoding='utf-8') as f:
        content = f.read()

    old_short = """AnswerLength.SHORT -> \"\"\"
- Answer concisely in 1-2 sentences when possible.
- Use the minimum sufficient set of grounded facts.
- Do not enumerate unnecessary supporting details.
- For list questions, provide the item names/titles without lengthy descriptions.
- Do not repeat information.
\"\"\".trimIndent()"""

    new_short = """AnswerLength.SHORT -> \"\"\"
- Give the minimum sufficient answer supported by the evidence.
- Never invent facts.
- Never use outside knowledge.
- Never repeat a fact.
- Never include irrelevant evidence.
- Do not create empty list items.
- Do not output headings as answers.
- For lists, preserve all requested items.
- Stop when the relevant evidence has been adequately covered.
\"\"\".trimIndent()"""

    old_long = """AnswerLength.LONG -> \"\"\"
- Give a substantially more detailed answer when the evidence contains additional relevant information.
- Include distinct relevant factual points from the retrieved evidence.
- Focus strictly on answering the specific question asked. Do not include facts from other document sections or topics that do not directly answer the question (e.g., when asked about tools and materials, do not include post-graft care or aftercare).
- Do not output empty list items or bare numbering sequences (such as '1. 2. 3. 4.').
- Do not repeat the same fact just to make the answer longer.
- Do not introduce external knowledge.
- For list questions:
  - include all distinct relevant items supported by the context
  - keep each numbered item to a single concise sentence (e.g. '1. Item Name: grounded description') directly from the evidence so all items are fully listed without truncation
  - do not create nested sub-bullets or multi-line breakdowns for items
  - do not repeat descriptions or sentences across multiple items
- For conceptual questions:
  - give the core definition
  - then include additional directly relevant supporting facts, mechanisms, requirements, anatomy, or relationships present in the evidence
- For troubleshooting/causes questions:
  - include the relevant causes
  - include relevant checks/actions/prevention steps present in the evidence
- For exact-fact questions:
  - answer the requested fact directly
  - optionally include closely related supporting conditions from the same evidence when they materially help the answer
- Stop when the relevant evidence has been adequately covered.
- Never repeat facts simply to increase length.
\"\"\".trimIndent()"""

    new_long = """AnswerLength.LONG -> \"\"\"
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
\"\"\".trimIndent()"""

    # Because there are special unicode dashes in the original text, let's use regex
    pattern_short = re.compile(r'AnswerLength\.SHORT -> """\n- Answer concisely in.*?Do not repeat information\.\n"""\.trimIndent\(\)', re.DOTALL)
    content, count_short = pattern_short.subn(new_short, content)
    
    pattern_long = re.compile(r'AnswerLength\.LONG -> """\n- Give a substantially more detailed answer.*?Never repeat facts simply to increase length\.\n"""\.trimIndent\(\)', re.DOTALL)
    content, count_long = pattern_long.subn(new_long, content)

    print(f"Replaced SHORT {count_short} times, LONG {count_long} times")
    
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/llm/GGUFLLM.kt', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    update_file()
