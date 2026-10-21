import sqlite3
import numpy as np

conn = sqlite3.connect('offline_docai_device.db')
c = conn.cursor()

# Get chunk 1538 and 553
c.execute('SELECT id, pageNumber, text FROM chunks WHERE id = 1538;')
r_1538 = c.fetchone()
print("Chunk 1538:", r_1538[0], "p." + str(r_1538[1]), r_1538[2][:120])

c.execute('SELECT id, pageNumber, text FROM chunks WHERE id = 553;')
r_553 = c.fetchone()
print("Chunk 553:", r_553[0], "p." + str(r_553[1]), r_553[2][:120])

# Let's inspect terms in chunk 1538
import re
def simple_terms(text):
    return set(re.findall(r'[a-zA-Z]{3,}', text.lower()))

terms_1538 = simple_terms(r_1538[2])
print("\nTerms in 1538:", sorted(list(terms_1538)))

sub_b = "administrators announced a temporary change in the construction schedule due to weather conditions"
terms_b = simple_terms(sub_b)
print("\nTerms in Sub-Query B:", sorted(list(terms_b)))
print("Overlap between Sub-Query B and Chunk 1538:", terms_b.intersection(terms_1538))
