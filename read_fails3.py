import os, re

html_dir = 'app/build/reports/tests/testDebugUnitTest/classes'

# Get all stdout output from a test file  
tests_to_check = [
    ('com.siddu.offlinedocai.SynthesisTest.html', ['testC_TellMeAboutTheProjects', 'testE_Skills']),
    ('com.siddu.offlinedocai.RAGRetrievalRelevanceTest.html', ['test2_ExactAnswerExists_HardwareBatterySpecs', 'test15_PageLookup_NASA_SE_Handbook_TOC_Synthesis_CleansDottedLeaders']),
    ('com.siddu.offlinedocai.GenericGroundedSynthesisTest.html', ['testRealDoc6_MultiSource_VerificationAndValidation']),
    ('com.siddu.offlinedocai.EndToEndRAGEvaluationTest.html', ['testN_ListQuestion_Projects', 'testU_MultiHopQuestion_CategoryProcesses']),
]

for fname, test_names in tests_to_check:
    html = open(os.path.join(html_dir, fname), encoding='utf-8', errors='ignore').read()
    for test in test_names:
        print(f"\n============================")
        print(f"TEST: {test}")
        print(f"============================")
        idx = html.find(test)
        if idx < 0:
            print("  NOT FOUND")
            continue
        block = html[idx:idx+8000]
        # Get all stdout lines (System.out or pre blocks)
        stdout = re.findall(r'<pre[^>]*>(.*?)</pre>', block, re.DOTALL)
        for s in stdout[:3]:
            clean = re.sub(r'<[^>]+>', '', s)
            clean = re.sub(r'&quot;', '"', clean)
            clean = re.sub(r'&lt;', '<', clean)
            clean = re.sub(r'&gt;', '>', clean)
            clean = re.sub(r'&amp;', '&', clean)
            # Find FINAL_ANSWER_TRACE
            trace_match = re.search(r'FINAL_ANSWER_TRACE: (.+?)(?:\n|$)', clean)
            if trace_match:
                print(f"  FINAL_ANSWER: {trace_match.group(1)[:400]}")
            # Find assertion
            assert_match = re.search(r'java\.lang\.AssertionError: (.+?)(?:\n|at )', clean)
            if assert_match:
                print(f"  ASSERT_ERROR: {assert_match.group(1)[:300]}")
