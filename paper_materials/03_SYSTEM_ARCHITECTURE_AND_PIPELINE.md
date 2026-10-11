# Section 3: System Architecture & Methodology

This document details the complete mathematical formulation, algorithmic pipelines, data representations, and formal pseudocode for the 5 processing stages comprising the **OfflineDocAI** architecture.

---

## 1. Architectural Overview & Dataflow

OfflineDocAI executes entirely on edge hardware without external dependencies. The overall architecture is organized into five tightly coupled subsystems:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 OFFLINEDOCAI ARCHITECTURAL PIPELINE                                    │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 [Stage 1: Document Ingestion]
      │  Raw File (PDF / DOCX / TXT)
      ▼
   DocumentProcessor (Page-level text, table & metadata extraction)
      ▼
   TextChunker (Hierarchical structural block parsing, WordPiece token budgeting, overlap injection)
      │
      ▼
 [Stage 2: Semantic Vector Space & Persistence]
      │  Structural Chunks: {c_i, text, pageSpan, sectionTitle, tokenCount}
      ▼
   ONNXEmbeddingEngine (all-MiniLM-L6-v2: Mean Pooling + L2 Normalization -> v in R^384)
      ▼
   Room SQLite Database (Direct zero-copy ByteBuffer serialization: 1536 bytes / vector BLOB)
      │
      ▼
 [Stage 3: Query Intent Classification & Hybrid Multi-Signal Retrieval]
      │  User Query q
      ▼
   QueryIntentRouter (Deterministic 6-intent classification: PAGE_LOOKUP, SUMMARY, COMPARISON, ...)
      ▼
   VectorStore (Multi-Signal Scoring: Dense Cosine Sim + Okapi BM25 + Reciprocal Rank Fusion RRF)
      ▼
   Relevance Floor Filtering (min_dense >= 0.20) + Spatial Neighbor Expansion
      │
      ▼
 [Stage 4: Dynamic Context Assembly & Native Neural Inference]
      │  Top-k Candidate Chunks
      ▼
   ContextAssembler (Greedy intent budgeting, chunk merging, algorithmic boundary overlap removal)
      ▼
   PromptBuilder (Untrusted marker sanitization, ChatML/Llama3 templating, reasoning tag handler)
      ▼
   LlamaRunner / llama_jni.cpp (C++17 native SIMD inference on quantized GGUF SLM weights)
      │
      ▼
 [Stage 5: Citation Sanitization & Grounding Verification]
      │  Generated Neural Response
      ▼
   CitationSanitizer (Canonical tag rewriting, ungrounded claim pruning, closed-world refusal)
```

---

## 2. Stage 1: Document Ingestion & Hierarchical Structural Chunking

### 2.1 Multi-Format Page Extraction & Spatial Multi-Column Reconstruction
Documents are ingested via lightweight edge-compatible parsers:
- **PDF Documents**: Parsed page-by-page using PDFBox-Android (`PDDocument`, `PDFTextStripper`). It implements a **Spatial Multi-Column Layout Reconstruction Engine** (`reconstructPageText`) that clusters character tokens by baseline, detects vertical column gutters in the central $35\%$--$65\%$ width band, separates full-width title/banner spans from dual-column body text, and re-orders text into proper reading order (Top Spans $\to$ Left Column $\to$ Right Column $\to$ Bottom Spans), eliminating horizontal column interleaving. Preserves 1-indexed page bounds ($P_{\text{start}}, P_{\text{end}}$).
- **DOCX Documents**: Parsed paragraph-by-paragraph using Apache POI (`XWPFDocument`), identifying structural heading styles (`Heading 1`, `Heading 2`) and table grids.
- **Plain Text**: Read via buffered streaming UTF-8 readers with multi-newline paragraph detection.

### 2.2 Hierarchical Block Decomposition
Raw document text is decomposed into an ordered sequence of structural blocks $B = \langle b_1, b_2, \dots, b_m \rangle$, where each block belongs to one of four mutually exclusive types:
$$\text{Type}(b) \in \{\text{HEADING}, \text{LIST\_ITEM}, \text{TABLE\_ROW}, \text{PARAGRAPH}\}$$

1. **Heading Detection**: Identified via Markdown markers (`#`), outline numbering (`Section 2.1`, `Chapter IV`), or capitalized label prefixes (`Overview:`). The active section heading $S_{\text{active}}$ is propagated to all subsequent sibling blocks.
2. **List Item Detection**: Preserves ordered and unordered bullet prefixes (`•`, `-`, `*`, `1.`, `a.`) along with child indentation lines.
3. **Table Row Preservation**: Detects column delimiters (`|`, `\t`) and groups rows into contiguous atomic blocks.
4. **Cross-Page Paragraph Continuation**: When a paragraph on page $p$ does not end with terminal punctuation (`.`, `!`, `?`) and page $p+1$ begins with a lowercase letter, the blocks are concatenated across page boundaries, updating the block's page span to `p-(p+1)`.

### 2.3 Token-Bounded Chunk Packing
Blocks are packed into chunks measured in **WordPiece embedding tokens** using the following constraints:
- **Target Chunk Size ($L_{\text{target}}$)**: 128 tokens
- **Maximum Chunk Ceiling ($L_{\text{max}}$)**: 180 tokens
- **Chunk Overlap ($O_{\text{tokens}}$)**: 32 tokens
- **Heading Reservation**: 24 tokens

```
Algorithm 1: Hierarchical Structure-Preserving Chunking
-------------------------------------------------------------------------------------------------------
Input : Extracted document pages P = <p_1, p_2, ..., p_n>, Token counter function Tau(.),
        Target tokens L_target = 128, Maximum tokens L_max = 180, Overlap tokens O = 32
Output: List of structural chunks C

1: B <- ParseBlocks(P)                          // Identify HEADING, LIST_ITEM, TABLE_ROW, PARAGRAPH
2: B <- MergeCrossPageParagraphs(B)            // Re-join sentences split across physical page breaks
3: U <- ConvertToAtomicUnits(B, Tau, L_max - 24) // Split oversized paragraphs at sentence boundaries
4: C <- empty list, current_chunk <- empty list, current_tokens <- 0
5: for each unit u in U do
6:     if u.type == HEADING and current_chunk is not empty then
7:         EmitChunk(C, current_chunk)          // New section forces a chunk boundary
8:         current_chunk <- empty list, current_tokens <- 0
9:     end if
10:    if current_tokens + u.tokens > L_max then
11:        EmitChunk(C, current_chunk)
12:        overlap_units <- GetTrailingUnits(current_chunk, O) // Carry over trailing sentences
13:        current_chunk <- overlap_units
14:        current_tokens <- SumTokens(overlap_units)
15:    end if
16:    Append u to current_chunk
17:    current_tokens <- current_tokens + u.tokens
18: end for
19: if current_chunk is not empty then EmitChunk(C, current_chunk)
20: return C
-------------------------------------------------------------------------------------------------------
```

---

## 3. Stage 2: Edge Embedding & Zero-Copy Persistence

### 3.1 On-Device Dense Vectorization
Each chunk $c_i \in C$ is vectorized using Microsoft ONNX Runtime Android with `all-MiniLM-L6-v2`:
1. The chunk text is tokenized into token IDs $\mathbf{t}$, attention mask $\mathbf{m}$, and token type IDs $\mathbf{s}$ with max sequence length $256$.
2. The ONNX model produces token hidden states $\mathbf{H} \in \mathbb{R}^{L \times 384}$.
3. Token representations are pooled using active-mask mean pooling:
   $$\mathbf{v}_{\text{raw}} = \frac{\sum_{j=1}^{L} m_j \cdot \mathbf{h}_j}{\sum_{j=1}^{L} m_j}$$
4. The vector is $L_2$-unit normalized:
   $$\mathbf{e} = \frac{\mathbf{v}_{\text{raw}}}{\|\mathbf{v}_{\text{raw}}\|_2} = \frac{\mathbf{v}_{\text{raw}}}{\sqrt{\sum_{k=1}^{384} v_k^2}}$$

### 3.2 Zero-Copy SQLite Vector Persistence
To eliminate JSON serialization and memory copy overhead on edge CPUs, 384-dimensional float arrays are serialized directly to raw byte arrays:
$$\text{FloatArray}(384) \xrightarrow{\text{Direct ByteBuffer (Little-Endian)}} \text{ByteArray}(1536)$$
Chunks are persisted in SQLite via Room ORM with schema:
```sql
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    documentId INTEGER NOT NULL,
    pageNumber INTEGER NOT NULL,
    chunkIndex INTEGER NOT NULL,
    text TEXT NOT NULL,
    sectionTitle TEXT,
    pageSpan TEXT NOT NULL,
    paragraphIndex INTEGER NOT NULL,
    embedding BLOB,
    embeddingModelId TEXT NOT NULL
);
```

---

## 4. Stage 3: Query Intent Classification & Hybrid Multi-Signal Retrieval

### 4.1 Deterministic Query Intent Classification
Incoming queries are deterministically classified in $<1\text{ ms}$ into one of 6 structural intent classes:

$$\text{Intent}(q) \in \{\text{PAGE\_LOOKUP}, \text{PAGE\_RANGE}, \text{SUMMARY}, \text{COMPARISON}, \text{ENUMERATION}, \text{GENERAL}\}$$

```
Algorithm 2: Deterministic Query Intent Routing
-------------------------------------------------------------------------------------------------------
Input : User Query string q
Output: QueryClassification { intent, pageStart, pageEnd, entityA, entityB, subQueries }

1: if MatchRegex(q, PAGE_RANGE) then
2:     (p_start, p_end) <- ExtractPageNumbers(q)
3:     return QueryClassification(PAGE_RANGE, pageStart=p_start, pageEnd=p_end)
4: else if MatchRegex(q, PAGE_LOOKUP) then
5:     p_target <- ExtractPageNumber(q)
6:     return QueryClassification(PAGE_LOOKUP, pageStart=p_target, pageEnd=p_target)
7: else if MatchRegex(q, SUMMARY_PATTERNS) then
8:     return QueryClassification(SUMMARY)
9: else if MatchRegex(q, COMPARISON_PATTERNS) then
10:    (entA, entB) <- ExtractComparativeEntities(q)
11:    return QueryClassification(COMPARISON, entityA=entA, entityB=entB, subQueries=[entA, entB])
12: else if MatchRegex(q, ENUMERATION_PATTERNS) then
13:    return QueryClassification(ENUMERATION)
14: else
15:    return QueryClassification(GENERAL)
16: end if
-------------------------------------------------------------------------------------------------------
```

### 4.2 Multi-Signal Hybrid Scoring & Reciprocal Rank Fusion (RRF)
For open-ended conceptual queries (`GENERAL`, `COMPARISON`, `ENUMERATION`), candidate chunks are ranked by fusing dense semantic similarity and lexical BM25 scores:

1. **Dense Cosine Similarity**:
   $$\text{Sim}_{\text{dense}}(q, c_i) = \mathbf{e}_q \cdot \mathbf{e}_{c_i} = \sum_{k=1}^{384} e_{q, k} \cdot e_{c_i, k}$$
   Candidates failing the semantic relevance floor ($\text{Sim}_{\text{dense}} < 0.20$) are discarded.
2. **Okapi BM25 Lexical Ranking**:
   For stemmed query terms $T_q = \text{Stem}(\text{Tokenize}(q))$:
   $$\text{Score}_{\text{BM25}}(q, c_i) = \sum_{t \in T_q} \text{IDF}(t) \cdot \frac{\text{TF}(t, c_i) \cdot (k_1 + 1)}{\text{TF}(t, c_i) + k_1 \cdot \left(1 - b + b \cdot \frac{|c_i|}{\text{avgdl}}\right)}$$
   where $k_1 = 1.2$, $b = 0.75$, and:
   $$\text{IDF}(t) = \ln\left(1.0 + \frac{N - \text{DF}(t) + 0.5}{\text{DF}(t) + 0.5}\right)$$
3. **Reciprocal Rank Fusion (RRF)**:
   Dense and lexical ranked lists are merged into a unified score with smoothing parameter $k_{\text{RRF}} = 60$:
   $$\text{RRF}(c_i) = \sum_{q \in \{q_{\text{main}}\} \cup Q_{\text{sub}}} \left( \frac{1}{60 + \text{Rank}_{\text{dense}}(q, c_i)} + \frac{1}{60 + \text{Rank}_{\text{BM25}}(q, c_i)} \right)$$
4. **Spatial Neighbor Expansion**:
   For the top-$3$ seed chunks, adjacent chunks ($i-1, i+1$) from the same document are appended with discounted relevance ($0.5 \times \text{Score}$) to preserve broader surrounding narrative context.

---

## 5. Stage 4: Dynamic Context Assembly & Overlap Deduplication

### 5.1 Intent-Budgeted Token Allocation
To maximize generation throughput and prevent prompt evaluation latency bottlenecks on mobile CPUs, context is budgeted dynamically by query intent:
- **`GENERAL` / `PAGE_LOOKUP`**: 384 tokens
- **`COMPARISON`**: 640 tokens
- **`SUMMARY` / `ENUMERATION` / `PAGE_RANGE`**: 1152 tokens

### 5.2 Algorithmic Boundary Overlap Deduplication
When contiguous chunks from the same document ($c_i, c_{i+1}$) are selected, they share 1–2 overlapping sentences due to ingestion overlap. `ContextAssembler` identifies and strips the shared leading sentences from $c_{i+1}$ before prompt injection:

```
Algorithm 3: Boundary Overlap Stripping (joinRemovingOverlap)
-------------------------------------------------------------------------------------------------------
Input : Preceding chunk text Prev, Following chunk text Next
Output: Deduplicated concatenated text Merged

1: tail_text <- NormalizeWhitespace(TakeLastCharacters(Prev, 1200))
2: next_sentences <- SplitIntoSentences(Next)
3: drop_count <- 0
4: for sentence s in next_sentences (up to max 3 sentences) do
5:     s_norm <- NormalizeWhitespace(s)
6:     if Length(s_norm) >= 12 and tail_text contains s_norm then
7:         drop_count <- drop_count + 1
8:     else
9:         break
10:    end if
11: end for
12: if drop_count == 0 then
13:    return Prev + "\n\n" + Next
14: else
15:    remainder <- Join(Drop(next_sentences, drop_count), " ")
16:    return Prev + " " + remainder
17: end if
-------------------------------------------------------------------------------------------------------
```

---

## 6. Stage 5: Native Neural Inference & Citation Grounding

### 6.1 Native C++17 JNI Engine (`llama_jni.cpp`)
Inference is executed through a direct C++17 JNI bridge to `llama.cpp`:
- **SIMD Vectorization**: Compiled with ARM NEON / ARM FP16 on `arm64-v8a` and AVX2/FMA on `x86_64`.
- **In-Memory Context Persistence**: The `llama_model` and `llama_context` pointers are retained in RAM across conversational turns, avoiding multi-second model reload penalties.
- **Quantization Format**: Q4_K_M (4-bit Medium K-quantization), providing optimal perplexity-to-memory ratio.

### 6.2 Prompt Defense & Citation Sanitization
1. **Indirect Prompt Injection Defense**: Untrusted text extracted from documents is sanitized via `PromptBuilder.sanitizeUntrusted()`, converting control sequences (`<|im_start|>`, `[INST]`, `<start_of_turn>`, `<think>`) into safe escaped unicode glyphs (`‹|im_start|›`).
2. **DeepSeek Reasoning Token Budgeting**: For distilled reasoning models, thinking output inside `<think>...</think>` is budgeted (max 768 tokens). If budget expires while thinking, the system injects `</think>` and forces immediate answer synthesis.
3. **Citation Sanitization**: Regex patterns extract model-generated citations (`[Source N, Page P]`), verify that source index $N$ exists in the provided context blocks, and canonicalize page spans.
4. **Deterministic Closed-World Refusal**: When context contains no evidence or negative sentinels (`[NO_EVIDENCE_FOUND]`), the model is constrained to output:
   $$\text{"I couldn't find this in the selected document(s)."}$$
