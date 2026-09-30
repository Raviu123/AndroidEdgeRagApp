import os
import re
import math
import difflib
import pypdf
import numpy as np
import onnxruntime as ort

# =========================================================================
# STEP 1 & 2: PDF LOAD & EXTRACTION
# =========================================================================
pdf_path = "Grafting_15_Page_Guide.pdf"
stat = os.stat(pdf_path)
file_size = stat.st_size
mtime = stat.st_mtime

reader = pypdf.PdfReader(pdf_path)
page_count = len(reader.pages)

pages_data = []

def detect_heading(line):
    trimmed = line.strip()
    if not trimmed or len(trimmed) > 90:
        return None
    if trimmed.startswith("#"):
        return re.sub(r"^#+\s*", "", trimmed).strip()
    m = re.match(r"^(?:(?:chapter|section|appendix|part)\s+[0-9a-zA-Z\.]+|[0-9]+(?:\.[0-9]+)+|[A-Z]\.[0-9]+)\s*[:\-]?\s*(.*)$", trimmed, re.I)
    if m:
        title = m.group(1).strip()
        return trimmed if title else m.group(0).strip()
    m_page = re.match(r"^Page\s+\d+:\s*(.*)$", trimmed, re.I)
    if m_page:
        return trimmed
    words = trimmed.split()
    if 1 <= len(words) <= 6 and all(w and (w[0].isupper() or not w.isalpha()) for w in words):
        if not trimmed.endswith((".", ",", ";")) and len(trimmed) <= 45:
            return trimmed
    return None

for i, page in enumerate(reader.pages):
    raw_text = page.extract_text() or ""
    raw_text = raw_text.replace("\r\n", "\n").strip()
    char_count = len(raw_text)
    word_count = len(raw_text.split())
    first_100 = raw_text[:100].replace("\n", " ")
    last_100 = raw_text[-100:].replace("\n", " ") if char_count > 100 else first_100
    
    heading = None
    for line in raw_text.split("\n"):
        h = detect_heading(line)
        if h:
            heading = h
            break
            
    pages_data.append({
        "page": i + 1,
        "text": raw_text,
        "char_count": char_count,
        "word_count": word_count,
        "first_100": first_100,
        "last_100": last_100,
        "heading": heading
    })

print(f"File: {pdf_path}, Size: {file_size}, Pages: {page_count}")
for p in pages_data:
    print(f"Page {p['page']}: chars={p['char_count']}, words={p['word_count']}, heading='{p['heading']}'")
    print(f"  First 100: {p['first_100']}")
    print(f"  Last  100: {p['last_100']}")

# =========================================================================
# STEP 3: TEXT SIMILARITY ACROSS PAGES
# =========================================================================
print("\n=== STEP 3: DUPLICATE PAGE ANALYSIS ===")
def get_body(text):
    lines = text.split("\n")
    body_lines = [l for l in lines if not re.match(r"^Page\s+\d+:", l.strip(), re.I)]
    return " ".join(body_lines).strip()

similarity_matrix = np.zeros((15, 15))
for i in range(15):
    for j in range(15):
        t1 = pages_data[i]["text"]
        t2 = pages_data[j]["text"]
        seq = difflib.SequenceMatcher(None, t1, t2)
        similarity_matrix[i, j] = seq.ratio()

for i in range(15):
    for j in range(i + 1, 15):
        sim = similarity_matrix[i, j]
        if sim > 0.60:
            print(f"Pages {i+1} and {j+1}: Similarity = {sim:.4f} (Body 1: '{get_body(pages_data[i]['text'])}')")

# =========================================================================
# STEP 8, 9, 10, 11, 12: EMBEDDING & RETRIEVAL SIMULATION
# =========================================================================
vocab = {}
with open("app/src/main/assets/models/embedding/vocab.txt", "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        w = line.strip()
        if w:
            vocab[w] = idx

cls_id = vocab.get("[CLS]", 101)
sep_id = vocab.get("[SEP]", 102)
unk_id = vocab.get("[UNK]", 100)

def wordpiece_tokenize(word):
    if word in vocab:
        return [word]
    res = []
    start = 0
    wlen = len(word)
    while start < wlen:
        end = wlen
        cur = None
        while start < end:
            sub = word[start:end]
            if start > 0:
                sub = "##" + sub
            if sub in vocab:
                cur = sub
                break
            end -= 1
        if cur is None:
            res.append("[UNK]")
            break
        res.append(cur)
        start = end
    return res

def tokenize_text(text, max_len=256):
    tokens = [cls_id]
    clean = text.lower()
    words = [w for w in re.split(r"(?<=[^\w\s])|(?=[^\w\s])|\s+", clean) if w.strip()]
    for w in words:
        for piece in wordpiece_tokenize(w):
            tokens.append(vocab.get(piece, unk_id))
    tokens.append(sep_id)
    
    input_ids = np.zeros(max_len, dtype=np.int64)
    attention_mask = np.zeros(max_len, dtype=np.int64)
    actual_len = min(len(tokens), max_len)
    for i in range(actual_len):
        input_ids[i] = tokens[i]
        attention_mask[i] = 1
    return input_ids, attention_mask

session = ort.InferenceSession("app/src/main/assets/models/embedding/all-MiniLM-L6-v2.onnx")

def get_embedding(text):
    input_ids, attention_mask = tokenize_text(text)
    token_type_ids = np.zeros(256, dtype=np.int64)
    inputs = {
        "input_ids": np.expand_dims(input_ids, 0),
        "attention_mask": np.expand_dims(attention_mask, 0),
        "token_type_ids": np.expand_dims(token_type_ids, 0)
    }
    outputs = session.run(None, inputs)
    token_embeddings = outputs[0][0] # [256, 384]
    mask = attention_mask
    valid_tokens = np.sum(mask)
    if valid_tokens > 0:
        mean_vec = np.sum(token_embeddings[:valid_tokens], axis=0) / valid_tokens
    else:
        mean_vec = np.zeros(384, dtype=np.float32)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec = mean_vec / norm
    return mean_vec

chunks = []
for p in pages_data:
    emb = get_embedding(p["text"])
    chunks.append({
        "chunk_id": p["page"],
        "page": p["page"],
        "text": p["text"],
        "section": p["heading"],
        "embedding": emb,
        "is_l2_normalized": abs(np.linalg.norm(emb) - 1.0) < 1e-4,
        "is_all_zero": np.all(emb == 0),
        "dimension": len(emb)
    })

print("\n=== STEP 8: EMBEDDING AUDIT FOR CHUNKS ===")
for p_num in [2, 4, 6, 7, 8, 11, 12]:
    c = chunks[p_num - 1]
    print(f"Page {p_num} (Chunk {c['chunk_id']}): Dim={c['dimension']}, NonZero={not c['is_all_zero']}, L2Norm={c['is_l2_normalized']}")

stop_words = set([
    "what", "when", "where", "how", "this", "that", "there", "their", "who", "why",
    "whom", "whose", "which", "them", "with", "from", "the", "is", "are", "was", "were",
    "and", "or", "in", "on", "at", "to", "a", "an", "of", "for", "by", "about", "give",
    "tell", "explain", "show", "me", "some", "can", "you", "does", "do", "did", "define",
    "list", "name", "mention", "describe", "find", "any", "all", "please", "these", "those",
    "be", "been", "being", "have", "has", "had", "would", "should", "could"
])

def run_retrieval(query):
    q_emb = get_embedding(query)
    q_norm = np.linalg.norm(q_emb)
    
    clean_q = query.lower().strip()
    raw_tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", clean_q) if len(t) > 1]
    query_keywords = [t for t in raw_tokens if t not in stop_words]
    query_phrase = " ".join(query_keywords)
    
    doc_count = float(len(chunks))
    chunk_words_map = {c["chunk_id"]: [w for w in re.split(r"[^a-zA-Z0-9]+", c["text"].lower()) if w] for c in chunks}
    avg_dl = np.mean([len(w) for w in chunk_words_map.values()])
    
    idf_map = {}
    for kw in query_keywords:
        containing = sum(1 for c in chunks if kw in chunk_words_map[c["chunk_id"]])
        if containing > 0:
            idf_map[kw] = math.log((doc_count - containing + 0.5) / (containing + 0.5) + 1.0) + 1.0
        else:
            idf_map[kw] = 0.5
            
    scored = []
    k1 = 1.2
    b = 0.75
    
    for c in chunks:
        cid = c["chunk_id"]
        c_text_lower = c["text"].lower()
        c_words = chunk_words_map[cid]
        doc_len = float(len(c_words))
        
        dense_score = float(np.dot(q_emb, c["embedding"]))
        
        bm25_score = 0.0
        for kw in query_keywords:
            term_occ = c_words.count(kw)
            if term_occ > 0:
                idf = idf_map.get(kw, 1.0)
                tf = (term_occ * (k1 + 1.0)) / (term_occ + k1 * (1.0 - b + b * (doc_len / avg_dl)))
                bm25_score += tf * idf * 8.0
                
        phrase_bonus = 0.0
        if len(query_phrase) >= 4 and query_phrase in c_text_lower:
            phrase_bonus = 20.0
            
        initial_score = (dense_score * 10.0) + bm25_score + phrase_bonus
        scored.append({
            "chunk": c,
            "dense": dense_score,
            "bm25": bm25_score,
            "phrase_bonus": phrase_bonus,
            "initial": initial_score
        })
        
    dense_sorted = sorted(scored, key=lambda x: x["dense"], reverse=True)
    dense_ranks = {s["chunk"]["chunk_id"]: idx + 1 for idx, s in enumerate(dense_sorted)}
    
    bm25_sorted = sorted(scored, key=lambda x: x["bm25"], reverse=True)
    bm25_ranks = {s["chunk"]["chunk_id"]: idx + 1 for idx, s in enumerate(bm25_sorted)}
    
    k_rrf = 60.0
    for s in scored:
        cid = s["chunk"]["chunk_id"]
        dr = dense_ranks[cid]
        br = bm25_ranks[cid]
        s["rrf"] = (1.0 / (k_rrf + dr)) + (1.0 / (k_rrf + br))
        
        rerank = s["initial"] + (s["rrf"] * 500.0)
        c_text_lower = s["chunk"]["text"].lower()
        if len(query_phrase) >= 4 and query_phrase in c_text_lower:
            rerank += 25.0
            
        matched_terms = sum(1 for kw in query_keywords if kw in chunk_words_map[cid])
        coverage = matched_terms / float(len(query_keywords)) if query_keywords else 0.0
        if coverage >= 1.0:
            rerank += 20.0
        elif coverage >= 0.5:
            rerank += 10.0
        elif matched_terms == 0:
            rerank -= 35.0
            
        s["structural_intent"] = rerank - (s["dense"] * 10.0 + s["bm25"] + s["rrf"] * 500.0)
        s["final"] = rerank
        
    return scored, q_emb

queries = [
    "What are the scion and rootstock?",
    "Why is grafting useful?",
    "What are the advanced applications of grafting?",
    "how to select best plants"
]

for q in queries:
    res, q_emb = run_retrieval(q)
    print(f"\n=======================================================")
    print(f"QUERY: \"{q}\"")
    print(f"Query Embedding: Dim={len(q_emb)}, NonZero={not np.all(q_emb==0)}, Norm={np.linalg.norm(q_emb):.4f}")
    print(f"=======================================================")
    
    print("\n--- STEP 9: DENSE-ONLY TOP 10 ---")
    d_sort = sorted(res, key=lambda x: x["dense"], reverse=True)[:10]
    for r, s in enumerate(d_sort, 1):
        print(f"Rank {r:2d} | Page {s['chunk']['page']:2d} | Chunk {s['chunk']['chunk_id']:2d} | Dense: {s['dense']:.4f}")
        
    print("\n--- STEP 10: LEXICAL-ONLY TOP 10 ---")
    b_sort = sorted(res, key=lambda x: x["bm25"], reverse=True)[:10]
    for r, s in enumerate(b_sort, 1):
        print(f"Rank {r:2d} | Page {s['chunk']['page']:2d} | Chunk {s['chunk']['chunk_id']:2d} | BM25: {s['bm25']:.4f}")
        
    print("\n--- STEP 11: RRF TOP 10 ---")
    r_sort = sorted(res, key=lambda x: x["rrf"], reverse=True)[:10]
    for r, s in enumerate(r_sort, 1):
        print(f"Rank {r:2d} | Page {s['chunk']['page']:2d} | Chunk {s['chunk']['chunk_id']:2d} | RRF: {s['rrf']:.5f}")
        
    print("\n--- STEP 12: FINAL RANKING TOP 10 ---")
    f_sort = sorted(res, key=lambda x: x["final"], reverse=True)[:10]
    for r, s in enumerate(f_sort, 1):
        print(f"Rank {r:2d} | Page {s['chunk']['page']:2d} | Chunk {s['chunk']['chunk_id']:2d} | Dense: {s['dense']:.4f} | Lex: {s['bm25']:.2f} | RRF: {s['rrf']:.5f} | Struct: {s['structural_intent']:.2f} | Final: {s['final']:.2f}")

