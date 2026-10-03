import re
import sys

def parse():
    logcat = sys.stdin.read()
    questions = re.split(r'EVALUATION MATRIX: \[', logcat)
    if len(questions) > 1:
        questions = questions[1:]
        
    for block in questions:
        q_match = re.search(r'(?:Q:|USER QUERY:)\s*(.*?)\n', block)
        if not q_match:
            continue
        question = q_match.group(1).strip()
        
        print(f"QUESTION: {question}")
        ans_match = re.search(r'(?:FINAL ANSWER|GENERATED & VALIDATED ANSWER):\n(.*?)\n(?:==|CITATION|EXPECTED)', block, re.DOTALL)
        if ans_match:
            print(f"ANSWER:\n{ans_match.group(1).strip()}\n")
            
if __name__ == "__main__":
    parse()
