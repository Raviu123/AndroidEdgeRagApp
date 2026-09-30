import re
import math
import pypdf

# Stopwords from VectorStore.kt
stopWords = {
    "what", "when", "where", "how", "this", "that", "there", "their", "who", "why",
    "them", "with", "from", "the", "is", "are", "was", "were", "and", "or", "in", "on", "at", "to", "a", "an",
    "of", "for", "by", "about", "give", "tell", "explain", "show", "me", "some", "can", "you", "does", "do", "define"
}

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

def extract_pdf_chunks(filepath):
    chunks = []
    with open(filepath, 'rb') as f:
        reader = pypdf.PdfReader(f)
        for i, page in enumerate(reader.pages):
            page_number = i + 1
            text = page.extract_text()
            if text:
                text = text.strip()
                page_chunks = chunk_text(text, page_number)
                chunks.extend(page_chunks)
    return chunks

def retrieve(query, chunks, top_k=10):
    clean_query = query.lower().strip()
    query_tokens = [t for t in re.split(r'[^a-zA-Z0-9]+', clean_query) if len(t) > 1]
    keywords = [t for t in query_tokens if t not in stopWords]
    
    is_explicit_summary = any(req in clean_query for req in ["summarize document", "summary of document", "overview of document", "document summary"])
    
    if is_explicit_summary:
        if len(chunks) <= top_k:
            return chunks
        step = len(chunks) / top_k
        selected = []
        for i in range(top_k):
            idx = min(int(i * step), len(chunks) - 1)
            if chunks[idx] not in selected:
                selected.append(chunks[idx])
        return selected

    doc_count = float(len(chunks))
    idf_map = {}
    for kw in keywords:
        containing_docs = sum(1 for c in chunks if kw in c['text'].lower())
        if containing_docs > 0:
            idf_map[kw] = math.log((doc_count + 1.0) / (containing_docs + 0.5)) + 1.0
        else:
            idf_map[kw] = 0.0

    phrase = " ".join(keywords)

    scored_chunks = []
    for chunk in chunks:
        chunk_text_lower = chunk['text'].lower()
        score = 0.0
        
        if len(phrase) >= 4 and phrase in chunk_text_lower:
            score += 50.0
            
        matched_keyword_count = 0
        for kw in keywords:
            if kw in chunk_text_lower:
                matched_keyword_count += 1
                occurrences = chunk_text_lower.split(kw)
                occurrences = len(occurrences) - 1
                tf = occurrences / (len(re.split(r'\s+', chunk_text_lower)) + 1.0)
                idf = idf_map.get(kw, 1.0)
                score += tf * idf * 10.0
                
        if len(keywords) > 0 and matched_keyword_count == len(keywords):
            score += 20.0
            
        if score > 0:
            scored_chunks.append((chunk, score))
            
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [(c[0], c[1]) for c in scored_chunks[:top_k]]

queries = [
    "What is grafting?",
    "What are the scion and rootstock?",
    "Why is grafting useful?",
    "How to select the plant?",
    "What are the different simple grafting methods?",
    "What are the types of bud grafting?",
    "What are the steps for making a successful graft?",
    "What causes graft failure?",
    "What are the advanced applications of grafting?"
]

chunks = extract_pdf_chunks('Grafting_15_Page_Guide.pdf')
for i, c in enumerate(chunks):
    c['id'] = i + 1

for q in queries:
    print(f"\nQuery: {q}")
    res = retrieve(q, chunks)
    print(f"Top 5 Pages: {[c[0]['pageNumber'] for c in res[:5]]}")
    for i, c in enumerate(res[:5]):
        chunk_info = c[0]
        score = c[1]
        snip = chunk_info['text'][:150].replace('\n', ' ')
        print(f"Rank {i+1}: Page {chunk_info['pageNumber']} - Chunk {chunk_info['id']} - Score {score:.2f} - {snip}...")
