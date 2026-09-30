import os
import re
import math
import pypdf
import numpy as np
import onnxruntime as ort

# =========================================================================
# 1. LOAD PDF AND EXTRACT PAGES
# =========================================================================
pdf_path = "Grafting_15_Page_Guide.pdf"
reader = pypdf.PdfReader(pdf_path)
pages_text = []
for i, page in enumerate(reader.pages):
    t = page.extract_text() or ""
    # Normalize CRLF
    t = t.replace("\r\n", "\n").strip()
    pages_text.append(t)

print(f"Total Pages in PDF: {len(pages_text)}")

# =========================================================================
# 2. HEADING DETECTION (Matching TextChunker.detectHeading)
# =========================================================================
def detect_heading(line, prev_line_empty=True, next_line_empty=True):
    trimmed = line.strip()
    if not trimmed or len(trimmed) > 90:
        return None
    if trimmed.startswith("#"):
        return re.sub(r"^#+\s*", "", trimmed).strip()
    
    # outline match: chapter/section/appendix or 1.2 or A.1
    m = re.match(r"^(?:(?:chapter|section|appendix|part)\s+[0-9a-zA-Z\.]+|[0-9]+(?:\.[0-9]+)+|[A-Z]\.[0-9]+)\s*[:\-]?\s*(.*)$", trimmed, re.I)
    if m:
        title = m.group(1).strip()
        return trimmed if title else m.group(0).strip()
    
    # colon match: "Requirements:", "Page 2: Benefits"
    m_col = re.match(r"^([A-Za-z0-9\s/&_\-]{3,50}):$", trimmed)
    if m_col:
        return m_col.group(1).strip()
    
    # Page X: Title
    m_page = re.match(r"^Page\s+\d+:\s*(.*)$", trimmed, re.I)
    if m_page:
        return trimmed
    
    # Isolated short uppercase
    if 3 <= len(trimmed) <= 60 and trimmed.isupper() and any(c.isalpha() for c in trimmed):
        if not trimmed.endswith((".", ",", ";")) and (prev_line_empty or len(trimmed) <= 40):
            return trimmed
            
    # Short title case
    words = trimmed.split()
    if 1 <= len(words) <= 6 and all(w and (w[0].isupper() or not w.isalpha()) for w in words):
        if prev_line_empty and not trimmed.endswith((".", ",", ";")) and len(trimmed) <= 45:
            return trimmed
            
    return None

# =========================================================================
# 3. CHUNKER (Matching TextChunker.chunkDocument)
# =========================================================================
class Chunk:
    def __init__(self, index, page_number, text, page_span, section_title, paragraph_index=0):
        self.index = index
        self.page_number = page_number
        self.text = text
        self.page_span = page_span
        self.section_title = section_title
        self.paragraph_index = paragraph_index
        self.id = index + 1
        self.document_id = 1
        self.embedding = None

def chunk_document(pages_text, target_words=350, max_words=500, overlap_words=60):
    chunks = []
    # Since each page in Grafting_15_Page_Guide.pdf is short (< 60 words),
    # let's see how TextChunker.chunkDocument behaves:
    # If targetWords=350, does TextChunker group short pages or does it emit per page?
    # Let's check: In DocumentRepository line 84:
    # val chunks = TextChunker.chunkDocument(result.pages)
    # Default targetWords = 350, maxWords = 500, overlapWords = 60.
    # In TextChunker:
    # It parses each page into SemanticBlock(startPage=page.pageNumber, endPage=page.pageNumber...)
    # Then checks isCrossPageContinuation:
    # current.endPage < next.startPage && current.blockType == PARAGRAPH && next.blockType == PARAGRAPH && isSentenceContinuation(current.text, next.text)
    # If a page has a heading (e.g. "Page 2: Benefits"), it is blockType=HEADING, so no cross-page merge!
    # Then in emitChunk: accumulates blocks until currentWordCount + blockWords > targetWords.
    # Wait! If targetWords=350, all 15 pages together have ~310 words total!
    # Let's check how many total words are in Grafting_15_Page_Guide.pdf!
    return chunks

total_doc_words = sum(len(p.split()) for p in pages_text)
print(f"Total words across all 15 pages of PDF: {total_doc_words}")

# Also check single-page chunkText vs chunkDocument:
# In ControlledGraftingSmokeAndroidTest line 142:
# val documentChunks = TextChunker.chunkDocument(extractedDoc.pages, targetWords = 35, maxWords = 50)
# In Step2BModelBenchmarkAndroidTest line 67 / ingestTestDocument:
# val documentChunks = TextChunker.chunkDocument(extractedDoc.pages, targetWords = 35, maxWords = 50)
# In SingleProductionQueryTraceTest:
# It creates chunks per page!
# In simulate_retrieval.py:
# extract_pdf_chunks creates chunks per page: page_chunks = chunk_text(text.strip(), i+1) -> 1 chunk per page!
# Let's see: if chunk per page: exactly 15 chunks (1 chunk per page).
