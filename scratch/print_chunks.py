import pypdf
import re

def chunk_text(text, page_number, chunk_size=400, overlap=80):
    words = re.split(r'\s+', text.strip())
    if not words or words == ['']:
        return []
    chunks = []
    start = 0
    index = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append({
            'index': index,
            'pageNumber': page_number,
            'text': chunk_text
        })
        if end == len(words):
            break
        start = end - overlap
        index += 1
    return chunks

chunks = []
with open('Grafting_15_Page_Guide.pdf', 'rb') as f:
    reader = pypdf.PdfReader(f)
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            page_chunks = chunk_text(text.strip(), i+1)
            chunks.extend(page_chunks)

for i, c in enumerate(chunks):
    print(f"--- Chunk {i+1} Page {c['pageNumber']} ---")
    print(c['text'][:200])
