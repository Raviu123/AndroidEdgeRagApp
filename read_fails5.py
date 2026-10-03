import os, re

html_dir = 'app/build/reports/tests/testDebugUnitTest/classes'

# The HTML report mixes stdout across tests in parallelized test runs.
# Let's get ALL stdout from the file and search for specific markers
tests_to_check = [
    ('com.siddu.offlinedocai.SynthesisTest.html', 'Tell me about the projects', 'testC'),
    ('com.siddu.offlinedocai.SynthesisTest.html', 'skills', 'testE'),
]

for fname, query_marker, test_label in tests_to_check:
    print(f"\n{'='*60}")
    print(f"TEST LABEL: {test_label} | QUERY MARKER: {query_marker}")
    print('='*60)
    html = open(os.path.join(html_dir, fname), encoding='utf-8', errors='ignore').read()
    clean = re.sub(r'<[^>]+>', '', html)
    for esc, char in [('&quot;','"'),('&lt;','<'),('&gt;','>'),('&amp;','&'),('&#39;',"'")]:
        clean = clean.replace(esc, char)
    # Find the query block
    idx = clean.find(f'QUERY: {query_marker}')
    if idx < 0:
        print("  Query not found in stdout")
        continue
    block = clean[idx:idx+2000]
    print(block[:1500])
