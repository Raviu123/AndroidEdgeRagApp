import os, re

html_dir = 'app/build/reports/tests/testDebugUnitTest/classes'

tests_to_check = [
    ('com.siddu.offlinedocai.SynthesisTest.html', 'testC_TellMeAboutTheProjects'),
    ('com.siddu.offlinedocai.SynthesisTest.html', 'testE_Skills'),
    ('com.siddu.offlinedocai.RAGRetrievalRelevanceTest.html', 'test15_PageLookup_NASA_SE_Handbook_TOC_Synthesis_CleansDottedLeaders'),
    ('com.siddu.offlinedocai.RAGRetrievalRelevanceTest.html', 'test6_Generalization_WhichProjects'),
    ('com.siddu.offlinedocai.GenericGroundedSynthesisTest.html', 'testRealDoc6_MultiSource_VerificationAndValidation'),
    ('com.siddu.offlinedocai.EndToEndRAGEvaluationTest.html', 'testN_ListQuestion_Projects'),
]

for fname, test_name in tests_to_check:
    html = open(os.path.join(html_dir, fname), encoding='utf-8', errors='ignore').read()
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print('='*60)
    idx = html.find(test_name)
    if idx < 0:
        print("  NOT FOUND")
        continue
    # Take a wider block to get full stdout
    block = html[idx:idx+15000]
    # Extract all pre-tag content (test stdout)
    pre_blocks = re.findall(r'<pre[^>]*>(.*?)</pre>', block, re.DOTALL)
    for pb in pre_blocks[:5]:
        clean = re.sub(r'<[^>]+>', '', pb)
        for esc, char in [('&quot;','"'),('&lt;','<'),('&gt;','>'),('&amp;','&'),('&#39;',"'")]:
            clean = clean.replace(esc, char)
        # Find QUERY, ANSWER, FINAL_ANSWER_TRACE lines
        for line in clean.split('\n'):
            line = line.strip()
            if line.startswith('QUERY:') or line.startswith('ANSWER:') or 'FINAL_ANSWER_TRACE' in line or 'AssertionError' in line or 'R5:' in line or 'Context Assembly' in line:
                print(f"  {line[:300]}")
