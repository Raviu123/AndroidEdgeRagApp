import os, re

html_dir = 'app/build/reports/tests/testDebugUnitTest/classes'

def get_test_details(fname, test_name):
    html = open(os.path.join(html_dir, fname), encoding='utf-8', errors='ignore').read()
    # Find test sections
    # Tests are marked with id attributes
    block = re.search(rf'id="{re.escape(test_name)}".*?(?=<h3|$)', html, re.DOTALL)
    if block:
        text = block.group(0)
        # Strip HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'&quot;', '"', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'\s+', ' ', text)
        return text[:2000]
    return None

# Read all stdout from failing tests
tests_to_check = [
    ('com.siddu.offlinedocai.SynthesisTest.html', 'testC_TellMeAboutTheProjects'),
    ('com.siddu.offlinedocai.SynthesisTest.html', 'testE_Skills'),
    ('com.siddu.offlinedocai.RAGRetrievalRelevanceTest.html', 'test2_ExactAnswerExists_HardwareBatterySpecs'),
    ('com.siddu.offlinedocai.RAGRetrievalRelevanceTest.html', 'test15_PageLookup_NASA_SE_Handbook_TOC_Synthesis_CleansDottedLeaders'),
]

for fname, test in tests_to_check:
    print(f"\n=== {test} ===")
    html = open(os.path.join(html_dir, fname), encoding='utf-8', errors='ignore').read()
    # Find FINAL_ANSWER_TRACE lines near this test
    idx = html.find(test)
    if idx >= 0:
        block = html[idx:idx+5000]
        traces = re.findall(r'FINAL_ANSWER_TRACE: (.*?)(?:<br|</pre)', block, re.DOTALL)
        for t in traces[:2]:
            clean = re.sub(r'<[^>]+>', '', t)
            clean = re.sub(r'&[a-z]+;', ' ', clean)
            print(f"  {clean[:400]}")
        # Also look for assertion failure
        errs = re.findall(r'java\.lang\.AssertionError: (.*?)(?:<br|</)', block, re.DOTALL)
        for e in errs[:2]:
            clean = re.sub(r'<[^>]+>', '', e)
            print(f"  ASSERT_ERR: {clean[:300]}")
