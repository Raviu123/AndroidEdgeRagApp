import os, re

html_dir = 'app/build/reports/tests/testDebugUnitTest/classes'
for fname in sorted(os.listdir(html_dir)):
    html = open(os.path.join(html_dir, fname), encoding='utf-8', errors='ignore').read()
    # Find FINAL_ANSWER_TRACE for each failed test
    fail_blocks = re.findall(r'class="failures".*?(\d+)', html)
    if fail_blocks and int(fail_blocks[0]) > 0:
        print(f"=== {fname} ===")
        # Get traces
        traces = re.findall(r'FINAL_ANSWER_TRACE: (.*?)(?:</pre>|<br>)', html, re.DOTALL)
        for t in traces[:3]:
            print(f"  TRACE: {t[:200]}")
        print()
