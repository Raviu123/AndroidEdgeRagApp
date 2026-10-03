import os, re, sys

html_dir = 'app/build/reports/tests/testDebugUnitTest/classes'
fname = 'com.siddu.offlinedocai.SynthesisTest.html'
html = open(os.path.join(html_dir, fname), encoding='utf-8', errors='ignore').read()
clean = re.sub(r'<[^>]+>', '', html)
clean = clean.replace('&quot;', '"').replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')

# Find skills query answer
idx = clean.find('QUERY: skills')
block = clean[idx:idx+2000]
# Print safely - replace unicode
safe = block.encode('ascii', 'replace').decode('ascii')
print(safe)
