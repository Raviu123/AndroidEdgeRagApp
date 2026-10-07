# OfflineDocAI — Critical Analysis of the RAG Pipeline

> ## Implementation status (2026-10-06)
>
> Sections 1–9 of the fix plan were implemented on `unified-rag-v2` (uncommitted working tree). The analysis below describes the code **before** these changes.
>
> | Area | Now |
> |---|---|
> | Storage | `deleteChunksForDocument` is real SQL (verified in generated `ChunkDao_Impl`). FK `ON DELETE CASCADE` + index on `chunks.documentId`. `embeddingModelId`/`embeddingDim` per chunk. DB v3 with `MIGRATION_2_3`, which drops orphan/duplicate chunks. `exportSchema = true` (`app/schemas/`). No destructive fallback. Single DB instance. |
> | Chunking / embedding | Token-based chunks measured with MiniLM's own WordPiece tokenizer: target 160 / max 200 tokens, 1–2 sentence overlap, new chunk per section, never above the 254-token embedding limit. Tokenizer matches HF BERT (accent stripping, Unicode punctuation, `[SEP]` kept). Dynamic sequence length. One shared engine. Atomic asset extraction. PDF word-separator fix. Scanned PDFs fail with a clear message. Imported files are copied privately. Stale documents are re-indexed automatically. |
> | Model files | Gradle `ensureEmbeddingModel` downloads + SHA-256-verifies the ONNX model on a fresh clone. LLM downloads are checked against Content-Length, catalog size, GGUF magic and published SHA-256. The non-functional embedding picker was removed. |
> | Query handling | `QueryReformulator` deleted (no suffixes, no `removePrefix("the")`). The original question is used. Routing happens once, with 6 document-agnostic intents, and the classification is passed to retrieval. |
> | Retrieval | Dense top-50 + BM25 (Porter stems, exact term match) top-50 → RRF (k=60) → top 20 (40 for enumerations) + neighbour chunks. Minimum-evidence floor. Cached per-document index. All bonuses, penalties and boilerplate lists are gone. |
> | Domain logic | `CitationValidator` (1,512 lines) deleted. No grafting/NASA/résumé/battery rules remain. `DocumentAgnosticCodeTest` fails the build if such vocabulary reappears in production code. |
> | Context | The budget is computed from the real `n_ctx` (4096 on devices with ≥ 3.4 GB RAM, else 2048, capped by the model's training context). It reserves the system prompt, question, template and answer tokens, counted with the model's tokenizer. Per-intent latency caps are 1800–2600 tokens. Greedy packing (typically 8–12 chunks). Adjacent chunks are merged with overlap removed. Context tags == citation format. |
> | Generation | Structured messages (no `parsePrompt`). The model's own GGUF chat template via `llama_chat_apply_template`, with family fallbacks. Control-token sanitising. `<think>` stripping. Repetition penalty. Stop on control tokens. UTF-8 byte-array JNI. Abort-callback cancellation plus a watchdog that really stops native generation. llama.cpp pinned to `e117148a`. |
> | Removed | Calculation aid, unconditional citations, canned answers, LLM-free extractive fallback, `cleanAndRenumberList`, the dead `librnllama` libs (~46 MB) and the dead modules. |
>
> **Verification:** 67/67 JVM tests pass (6 new RAG suites). `assembleDebug` builds for arm64-v8a and x86_64. 3/3 instrumented tests pass on an x86_64 emulator: real-embedding paraphrase retrieval across topics, the v2→v3 migration with cascade, and import/re-index/delete. The on-device LLM test (`groundedAnswerFromInstalledModel`) was **skipped**: no GGUF model was installed and the emulator lacked space. Run it on a phone with a model before release.
>
> **Still open (Phase 3/4 of §12):**
> - The batch question queue is still fire-and-forget (not sequential).
> - There is no user Stop button or token streaming.
> - Conversations store only one document id.
> - No OCR.
> - No learned cross-encoder reranker.
> - `app/src/main/java_original_preserved/` (not compiled) still holds old code.

**Branch:** `unified-rag-v2` (working tree incl. uncommitted changes) · **Date:** 2026-10-06
**Scope:** document ingestion → chunking → embedding → storage → query understanding → retrieval → reranking → context assembly → LLM generation → post-processing, plus app wiring, database, native layer and tests.

All paths below are relative to `app/src/main/kotlin/com/siddu/offlinedocai/` unless they start with `app/`. Line numbers refer to the current working tree.

---

## 0. Executive verdict

| Question you asked | Short answer |
|---|---|
| Is the document being **embedded** properly? | **Partially.** The ONNX MiniLM model and mean pooling are correct, but every chunk is **truncated to 256 tokens while chunks are 350–500 words (~450–900 tokens)**. Roughly half of every chunk is invisible to semantic search. |
| Is it **indexed / stored** properly? | **No.** Embeddings are stored correctly as BLOBs, but `deleteChunksForDocument` is a **no-op** (verified in the generated Room code). Re-indexing duplicates chunks and deleting a document leaves orphaned chunks forever. There is no FK, no index, and no record of which embedding model produced a vector. DOCX/TXT are stored as a single "page 1". |
| Is **retrieval** happening properly? | **It runs, but it is unreliable.** It is a brute-force scan of every chunk on every query, routed by ~40 hand-written regexes into 10 strategies. The **query rewriter corrupts questions** (`"what is thermodynamics"` → `"rmodynamics definition"`). The retriever and the context assembler can disagree on intent. Paraphrased questions that share no keywords with the document are actively **filtered out**. |
| Is **reranking** happening? | **There is no real reranker.** The "Cross-Reranker" is a hand-tuned score of BM25 + phrase bonuses + regex bonuses with magic constants. Dense similarity contributes at most **10 points** against lexical bonuses of 20–100+. A chunk with zero keyword overlap gets −35 and is dropped. |
| Can the AI **generate a response** from the retrieved items + question? | **Barely.** For normal questions the evidence budget is **384 tokens, smaller than a single chunk**, so the LLM receives **exactly one chunk** (the fallback). The prompt is built as a string, then **re-parsed by regex** inside `GGUFLLM`. If the document contains a line starting with `Question:`, the real question is replaced by document text. n_ctx is fixed at 2048. There is no streaming, no working cancellation, and the model loads on the main thread. |
| Can it **analyse questions correctly**? | **Only for the grafting test document.** Query understanding is regex-only, and the answer validator contains **grafting-specific vocabulary** (`scion`, `rootstock`, `cambium`, `21c`, `85%`, "weekly inspection"). On other documents those rules delete correct sentences. |

**Bottom line:** the architecture has the right *shape* (extract → chunk → embed → hybrid retrieve → assemble → generate → validate). However, a handful of sizing and wiring bugs mean the LLM usually sees one partially-embedded chunk. Large amounts of heuristic code are overfit to one test PDF. The fastest path to a "seamless" app is **removing and simplifying**, not adding more heuristics (see §9).

All 236 JVM unit tests pass (19 suites, `./gradlew testDebugUnitTest`). This does **not** contradict the above. The JVM tests use a hash-based mock embedding (`ONNXEmbeddingEngine.computeDeterministicMockEmbedding`) and are largely built around the grafting document, so they cannot detect these failures (§8).

---

## 1. Architecture as built

```
UI (MainActivity.kt, 2,775 lines, all screens)
  └─ MainViewModel  ── builds Room DB, ChatRepository, DocumentRepository, loads LLM (main thread!)
       ├─ DocumentRepository.importDocument  (unscoped CoroutineScope(IO))
       │     DocumentProcessor (PDFBox / POI / TXT)
       │     → TextChunker.chunkDocument (350/500/60 words)
       │     → ONNXEmbeddingEngine #1 (MiniLM-L6 INT8, 256 tok, mean-pool, L2)
       │     → chunkDao.insertAll (once, at the very end)
       └─ ChatRepository.sendMessage (fire-and-forget launch)
             QueryReformulator.classifyAndRewrite(userText)  → searchString (rewritten, lower-cased)
             QueryIntentRouter.route(userText)               → intent used for budget/assembly
             ONNXEmbeddingEngine #2 .getEmbedding(searchString)
             VectorStore.retrieveDetailed(searchString)      → routes AGAIN on searchString (10 strategies)
             ContextAssembler.assemble (budget 384 / 640 / 1152 tok)
             buildPrompt(...)  → string
             GGUFLLM.generate(string)  → parsePrompt() regex re-parse → ChatML / Llama-3 template
                 → LlamaRunner (JNI, llama.cpp @ master) → cleanAndRenumberList → citation appended
```

Component inventory:

| Component | File | Status |
|---|---|---|
| Extraction | `domain/rag/DocumentProcessor.kt` | Works for text PDFs; see §2 |
| Chunking | `ai/chunking/TextChunker.kt` | Structure-aware, but chunk sizes conflict with the embedder |
| Embedding | `ai/embeddings/ONNXEmbeddingEngine.kt`, `Tokenizer.kt` | Correct model math, wrong sequence length vs chunk size |
| Storage | `data/` (Room v2) | Delete bug, no FK/index, no model versioning |
| Query understanding | `ai/retrieval/QueryIntentRouter.kt`, `ai/reformulation/QueryReformulator.kt` | Two independent classifiers that disagree; rewriter bugs |
| Retrieval + "rerank" | `ai/retrieval/VectorStore.kt` (1,253 lines) | Brute force, heuristic, lexically dominated |
| Context assembly | `ai/retrieval/ContextAssembler.kt` | Budget too small → one chunk |
| Generation | `ai/llm/GGUFLLM.kt` (2,425 lines), `LlamaRunner.kt`, `cpp/llama_jni.cpp` | Works; fragile prompt plumbing, threading and cancellation bugs |
| Validation | `ai/validation/CitationValidator.kt` (1,512 lines) | Overfit to the grafting PDF; now bypassed for native output |
| Dead code | `GGUFEmbeddingEngine`, `BertWordPieceTokenizer`, `DocumentProfiler`, `SelectiveContentReducer`, `OllamaLLM`, `java/org/nehuatl/llamacpp/LlamaContext.java`, 8× `jniLibs/**/librnllama*.so` (~46 MB), `java_original_preserved/` | Remove |

---

## 2. Ingestion & extraction (`domain/rag/DocumentProcessor.kt`)

| Sev | Finding | Evidence |
|---|---|---|
| **High (verify)** | **PDF word spacing is probably lost.** The custom `PDFTextStripper` overrides `writeString(text, positions)`, `writeLineSeparator` and `writeParagraphSeparator`, but **not `writeWordSeparator()`**. PDFBox emits inter-word spaces through `writeWordSeparator()` into the stripper's internal writer, which this code ignores. PDFs that position words without literal space glyphs (LaTeX, many generators) produce `Thequickbrownfox`. That breaks BM25, the tokenizer, and the embedding. Verify by logging page text from a LaTeX PDF. Fix: override `writeWordSeparator() { currentPageText.append(' ') }`. | `DocumentProcessor.kt:85-114` |
| High | **Scanned/image PDFs import as "COMPLETED" with 0 chunks.** If no text is found, `pages` is empty, no exception is thrown, and the document is marked completed. Every question then answers "could not find". No OCR and no user warning. | `DocumentProcessor.kt:119-127`, `DocumentRepository.kt:97-127` |
| Medium | **DOCX and TXT are a single "page 1".** All chunks get `pageNumber = 1`, so page citations, PAGE_LOOKUP, PAGE_RANGE and neighbour page constraints are meaningless for these formats. | `DocumentProcessor.kt:169, 189` |
| Low | Legacy `.doc` (`application/msword`) is routed to the DOCX (XWPF) parser and always fails. | `DocumentProcessor.kt:67` |
| Low | No headers/footers/page-number stripping. Repeated running headers pollute chunks and BM25 IDF. | — |
| Medium | The original file is **not copied**; only the content URI is stored. `GetContent()` grants are temporary, so "Reprocess" fails after an app restart (`SecurityException`). | `MainActivity.kt:1752`, `DocumentRepository.kt:155-162` |

---

## 3. Chunking (`ai/chunking/TextChunker.kt`)

The structure-preserving block parser (headings, lists, tables, cross-page continuation) is the best-designed part of the pipeline. Problems:

| Sev | Finding | Evidence |
|---|---|---|
| **Critical** | **Chunk size is incompatible with both the embedder and the context budget.** `targetWords = 350`, `maxWords = 500`. MiniLM reads 256 tokens (≈170–190 English words), and the LLM evidence budget for normal questions is 384 tokens (≈110–130 words). The ingestion and generation sides were sized independently. | `TextChunker.kt:48-50` vs `ONNXEmbeddingEngine.kt:37` vs `ContextAssembler.kt:73` |
| Medium | **Overlap almost never happens.** Overlap only re-uses the previous block if it is ≤ 60 words, and normal prose paragraphs are longer. Sentence-split oversized blocks get no overlap at all. Facts spanning a boundary are split. | `TextChunker.kt:295-303, 262-289` |
| Medium | **Heading detection false positives.** Any line ≤ 90 chars starting with `N.N` (e.g. `3.5 million users were…`) becomes a heading. Any short Title-Case line after a blank line (names, table captions) becomes a section title. `sectionTitle` then drives SECTION_QUERY and blocks neighbour expansion. | `TextChunker.kt:346-373` |
| Low | `isTableRow` treats any line with ≥ 2 `|` or ≥ 2 tabs as a table. `isListItem` matches `A. Smith`, `I. Intro`. | `TextChunker.kt:378-387` |
| Low | A chunk spanning pages stores only its **start** page in `pageNumber`. PAGE_LOOKUP filters on `pageNumber ==`, so content that begins on the previous page is missed. `pageSpan` exists but is not used for filtering. | `TextChunker.kt:244-252`, `VectorStore.kt:165` |

---

## 4. Embedding (`ai/embeddings/`)

**What is correct:** all-MiniLM-L6-v2 (INT8 ONNX, 22.9 MB bundled), WordPiece tokenization, attention-mask mean pooling, L2 normalisation, the same model for query and documents, and a real exception if inference fails.

| Sev | Finding | Evidence |
|---|---|---|
| **Critical** | **Silent truncation at 256 tokens.** `padOrTruncate(tokens, 256)` cuts every 350–500-word chunk roughly in half. It also drops the trailing `[SEP]` token. Sentences in the second half of a chunk cannot be found semantically. | `ONNXEmbeddingEngine.kt:37, 129-130`, `Tokenizer.kt:94-105` |
| High | **Two separate embedding engines** (one in `DocumentRepository`, one in `ChatRepository`) mean two ORT sessions and double the memory. On first run both may extract the asset to the **same file concurrently**. Extraction is not atomic: a partial copy > 1 KB is never re-extracted and breaks all embeddings permanently. | `DocumentRepository.kt:39`, `ChatRepository.kt:57`, `ONNXEmbeddingEngine.kt:60-72` |
| High | **The embedding model picker does nothing.** The engine is hard-coded to `all-MiniLM-L6-v2.onnx`. `AvailableEmbeddingModels` offers BGE-small (needs CLS pooling + a query instruction prefix) and a GGUF embedder with no backend. Downloads go to `models/embeddings/` but the app checks `models/embedding/`. If this were ever wired up, existing vectors would silently mix models: no model ID or dimension is stored per chunk, and dimension mismatch returns cosine 0. | `AvailableEmbeddingModels.kt`, `MainViewModel.kt:112, 181-202`, `VectorStore.kt:1215` |
| Medium | Always pads to 256, so a 6-word query costs as much as a full chunk. Use dynamic length (`min(tokens, 256)`); the model accepts variable shape. | `ONNXEmbeddingEngine.kt:133` |
| Low | Tokenizer deviates from HF BERT-uncased: no accent stripping/NFD, `\p{Punct}` is ASCII-only (Unicode punctuation and CJK not split), and failed words emit partial pieces + `[UNK]` instead of a single `[UNK]`. These are minor quality losses. | `Tokenizer.kt:42-91` |
| Low | Embedding is done one chunk at a time, held in memory, and inserted once at the end. Killing the app mid-import loses everything and leaves the document stuck in `PROCESSING` forever (no recovery on startup). | `DocumentRepository.kt:110-127` |
| Info | `all-MiniLM-L6-v2.onnx` is **not tracked in git** (only `vocab.txt` is), so a fresh clone cannot embed anything. | `git ls-files app/src/main/assets` |

---

## 5. Storage & indexing (`data/`)

| Sev | Finding | Evidence |
|---|---|---|
| **Critical** | **`deleteChunksForDocument` does nothing.** It is declared `suspend fun deleteChunksForDocument(docId: Long) {}`, a Kotlin interface method *with a body*. Under kapt, Room's generated code delegates to that empty body: `ChunkDao_Impl.java:441 → ChunkDao.DefaultImpls.deleteChunksForDocument(...)`. Consequences: **Reprocess duplicates every chunk** (duplicates then compete in retrieval and eat the budget), and **deleting a document leaves its chunks forever**, despite the UI saying they are removed. Fix: remove `{}`. | `data/dao/ChunkDao.kt:29`, `app/build/generated/source/kapt/debug/.../ChunkDao_Impl.java:439-442` |
| High | **No foreign keys, cascades or indices.** `chunks.documentId` and `messages.conversationId` are unindexed, and every query does `WHERE documentId IN (...)` as a full scan. `deleteDocument` only deletes the document row. | `data/entities/ChunkEntity.kt`, `MainViewModel.kt:294-299` |
| High | **No embedding provenance.** There is no `embeddingModelId`/`dim` column, so the app cannot detect that vectors need re-embedding after a model change. | `ChunkEntity.kt` |
| High | `fallbackToDestructiveMigration()` plus `exportSchema = false`: the next schema bump without a migration **silently wipes all documents and chats**, and migrations cannot be tested. | `MainViewModel.kt:36`, `data/AppDatabase.kt:18` |
| Medium | **No vector index.** Every question loads every chunk row (text + 1.5 KB BLOB) into memory and deserialises every embedding. This is OK for a few hundred chunks and slow for books. | `VectorStore.kt:129` |
| Low | `sourcesJson` is built by string concatenation, escaping only `"` (not `\` or newlines). `ChatExporter` splits it on `|||`, so exports contain raw JSON. | `ContextAssembler.kt:515-521`, `ChatExporter.kt:61, 86` |

---

## 6. Query understanding, retrieval & "reranking"

### 6.1 Two classifiers that disagree

`ChatRepository` runs **both** `QueryReformulator.classifyAndRewrite(userText)` and `QueryIntentRouter.route(userText)` (`ChatRepository.kt:217-219`). Retrieval is called with the **rewritten, lower-cased** `searchString`. `VectorStore.retrieveDetailed` then **routes again on that string** (`VectorStore.kt:137`). The context budget and assembly use the intent from the **original** text (`ChatRepository.kt:241-250`).

Example: *"What are all the skills listed?"*
- Router on the original text → `EXHAUSTIVE_LIST` → 1,152-token budget, list assembly.
- Reformulator → `"all the skills listed definition"` → VectorStore routes this to `CONCEPTUAL_QUERY`. The exhaustive scan never runs.
- Lower-casing also disables `ENTITY_LOOKUP`, which requires capitalised names (`QueryIntentRouter.kt:124-126`).

### 6.2 Query rewriter bugs (`ai/reformulation/QueryReformulator.kt`)

| Sev | Finding | Line |
|---|---|---|
| **Critical** | `clean.removePrefix(prefix).trim().removePrefix("the")` strips the **letters** "the" from the start of the term: *"what is thermodynamics"* → `"rmodynamics definition"`, *"explain theory of change"* → `"ory of change definition"`, *"what are the themes"* → `"mes definition"`. | `:82` |
| High | Every "what is / what are / explain / describe / tell me about" question gets the word **`definition`** appended. Every "how/why/when/where" question gets **`procedure guide`** appended. Those words become BM25 keywords and phrase-match targets, biasing retrieval toward chunks containing "definition"/"guide"/"procedure" (and, through substring matching, "guidelines", "defined"...). | `:76-138` |
| Medium | Any 1–2 word input (e.g. "battery capacity", "Project Apollo") is forced into a definitional query. | `:108-119` |
| Medium | **No conversation history.** It is always called with `profile = null`, and previous turns are never used, so follow-ups ("what about the second one?") cannot work. | `ChatRepository.kt:217` |

### 6.3 Intent router (`ai/retrieval/QueryIntentRouter.kt`)

About 40 large regexes with first-match priority. They work for the phrasing they were written for and misroute anything else. Notable issues:

- **EXACT_FACT returns nothing on a regex miss.** If the router decides the question is a DATE/AMOUNT/PERCENT/ID fact, `retrieveExactFact` returns **an empty list** unless a chunk matches the value regex. For example, a date written "March 2024" does not match the DATE regex (`\d{1,2}[/-]...` or `Month DD, YYYY`), so the user is told the document has no information (`VectorStore.kt:240-243, 251`).
- `FACT_URL_REGEX` matches `link`/`repository`, and `FACT_AMOUNT_REGEX` matches `package`/`cost`. *"How does the package manager link libraries?"* becomes a URL-fact lookup.
- `FACT_TEMPERATURE_REGEX` contains `healing temperature`, a phrase from the grafting document.
- `EXHAUSTIVE_LIST_REGEX` matches any query ending in a plural word, e.g. *"which methods"* or a bare `^\w+s$`, and also `tell me about the …s`.
- PAGE_LOOKUP matches `p. 5` anywhere, e.g. "section 3, p. 5 of the appendix".

### 6.4 Retrieval & scoring (`ai/retrieval/VectorStore.kt`)

**Is there reranking?** Not in the ML sense. `scoreCandidates` (`:756-1053`) computes:

```
initialScore = dense*10 + BM25*8 per term + phraseBonus(20/25)
rerank       = initialScore + RRF*500 (≤ ~16)
             + 25 (phrase) + 30 (compound) + 50 (def regex) + 30 (lead regex)
             + up to 16 (entity/list/unit regex bonuses)
             + 20 / 10 / −35 (keyword coverage)
survived     = rerank ≥ 5 and not boilerplate
```

| Sev | Finding | Evidence |
|---|---|---|
| **Critical** | **Semantic similarity is nearly irrelevant.** Dense cosine (0–1) is multiplied by **10**, while lexical bonuses add 20–100+. A chunk sharing **no keyword** with the query gets **−35** (`:1015-1016`) and falls below the survival threshold. Paraphrased questions (the main reason to use embeddings) are filtered out, and `retrieveConceptual` then returns empty, so the user is told "could not find information". | `:849, 939, 1009-1021, 604-616` |
| **High** | **Substring keyword matching causes false positives.** `wordMatchesKeyword` accepts any doc word that *contains* a ≥ 4-char stem: `rate` ⊂ accu**rate**/gene**rate**/sepa**rate**, `test` ⊂ con**test**, `data` ⊂ **data**base, `cell` ⊂ ex**cell**ent. This feeds BM25, IDF, coverage and FIND_ALL. | `:1204-1212` |
| **High** | **Performance.** `wordMatchesKeyword` compiles `Regex("[^a-zA-Z0-9]")` on **every call**. It runs for chunks × words × keywords, three times (IDF, BM25, coverage), plus `getRootStems` allocations. For 1,000 chunks × 400 words × 5 keywords that is ~6 M regex compilations per query, before the LLM even starts. | `:795-798, 825-833, 906-911, 1205` |
| High | **Boilerplate filter is document-specific.** Any chunk containing "table of contents", "all rights reserved", "published by the", "distribution statement", "STI program" or "standard form 298" (NASA report phrases) is excluded from **every** retrieval path, including the first content chunk of most books. | `:1170-1183` |
| High | **Definition scoring has hard-coded domain penalties.** −80 for any chunk mentioning failure/defect/symptoms/troubleshooting, −40 for tools/supplies, −30 for "aftercare / post-graft / weekly inspection" (grafting vocabulary), and +30 for being on page 1. Definition candidates with a negative score are discarded entirely, so "What is a failure mode?" cannot retrieve its own definition. | `:1110-1128, 930` |
| Medium | **GLOBAL_SUMMARY is really "first N chunks".** Once a document has any heading, almost every chunk has `sectionTitle != null`, so the "heading chunks" loop fills `topK-1` slots with the first chunks in order. Stratified sampling never runs. | `:563-579` |
| Medium | EXHAUSTIVE_LIST, FIND_ALL, PAGE_LOOKUP and PAGE_RANGE return **unbounded** lists, which are then truncated to 1–2 chunks by the assembler (§7). The "exhaustive" guarantee is lost. | `:367-480` |
| Medium | Comparison: `candA.take(halfK)` takes candidates in **document order**, not score order. | `:526-530` |
| Low | Neighbour expansion only runs for results with score ≥ 20 and never crosses a section boundary, so it mostly does not trigger. Its "Failure" keyword filter is domain-specific. | `:699-734` |
| Low | Production code uses `println` + full query logging at INFO (privacy) and logs the top-10 previews on every query. | `:1036-1050` |

---

## 7. Context assembly (`ai/retrieval/ContextAssembler.kt`)

| Sev | Finding | Evidence |
|---|---|---|
| **Critical** | **The LLM usually receives exactly one chunk.** The budget for EXACT_FACT / PAGE_LOOKUP / ENTITY / CONCEPTUAL / SECTION is **384 tokens**. With the project's own `estimateTokens` + 25 overhead, measured costs are: 100 words ≈ 359, 150 ≈ 460, 200 ≈ 567, 350 ≈ 923. A normal 200–350-word chunk never fits. `selectEvidenceWithinBudget` then hits `if (selected.isEmpty()) selected.add(prioritized.first())`, so only the single top chunk is sent. The 1,152 budget (lists/summaries) fits about one or two chunks. **This is the main reason answers are incomplete or wrong.** | `:73-75, 83-98, 366-378`; `ChatRepository.kt:241-244` |
| High | The budget is calculated for a 2,048-token context while n_ctx is 2,048 and the model's real context (e.g. Qwen2.5 = 32K) is ignored. The 6 metadata lines per block (`[DOC_ID=] [SOURCE_ID=] [PAGE=] [PAGE_SPAN=] [SECTION=] [CHUNK_INDEX=]`) are not counted in the budget. | `:28-39`, `GGUFLLM.kt:51` |
| Medium | The context labels blocks `[SOURCE_ID=…][PAGE=…]` but the model is told to cite `[Source X, Page Y]`, a different syntax. Small models handle this mismatch poorly. | `:28-39` |
| Medium | Prefix-based dedup (first 40 chars) can drop distinct chunks that share a boilerplate opening (e.g. repeated headers). | `:169-187` |

---

## 8. Generation (`ai/llm/GGUFLLM.kt`, `LlamaRunner.kt`, `cpp/llama_jni.cpp`)

### 8.1 Prompt plumbing

- `ChatRepository.buildPrompt` (`:407-455`) builds a string with the **length instructions after `Answer:`**. `GGUFLLM.generate` then **re-parses** that string with `parsePrompt` (`GGUFLLM.kt:1165-1265`), keeps only the context and question, and builds a new template. All of ChatRepository's instructions are discarded.
- **High: question hijack.** `parsePrompt` takes the first line starting with `Question:` (or a line starting with `Answer`/`Instruction:`) **anywhere**, including inside the document context. FAQ documents, exam papers and interview transcripts therefore replace the user's question with document text and truncate the context (`GGUFLLM.kt:1170-1180`). Fix: pass `context` and `question` as separate parameters (a data class), never as a string to re-parse.
- **High: fabricated "Calculation Aid (verified arithmetic)".** For any question containing total/sum/difference/average/mean, the **first two numbers found anywhere in the context** (page numbers, years, section numbers) are added or subtracted. The result is injected into the prompt as *verified* (`GGUFLLM.kt:486-525`). Delete this.
- **Chat templates:** only ChatML and Llama-3 exist. The choice is made from a filename enum, and the `architecture` argument is never passed. DeepSeek-R1-Distill (in the catalog) gets ChatML, and its `<think>` block is shown to the user and consumes the 128–512 token budget. Imported Gemma/Phi/Mistral files get whichever template was used last. `tokenizer.chat_template` from the GGUF is never read (`GGUFLLM.kt:127-138, 214-217, 400-413`). Use `llama_chat_apply_template` with the model's own template.
- The Qwen system prompt says "Answer concisely" even in LONG mode (`:374`).

### 8.2 Runtime

| Sev | Finding | Evidence |
|---|---|---|
| **Critical** | **Native model load runs on the main thread** (ViewModel `init` → `checkModels` → `selectLlmModel` → `nativeInit`), and on model import/selection. It is `@Synchronized` with `inference`, so switching models during a generation **freezes the UI until generation ends** (ANR). | `MainViewModel.kt:93-97, 126, 136, 146`; `LlamaRunner.kt:93, 214` |
| **Critical** | **Timeouts and cancellation do not work.** `withTimeout(120s)` wraps a *blocking* JNI call, so it can only fire after native code returns. The finished answer is then thrown away as "timed out". There is no user Stop button wired to `nativeCancel`. A cancel issued before entry is reset by `g_cancel_generation.store(false)`. | `ChatRepository.kt:199-208, 271-280`; `llama_jni.cpp:160, 494` |
| **Critical** | **Batch question queue is not sequential.** `QuestionQueueManager` awaits a lambda that calls the fire-and-forget `sendMessage`, so all N questions launch ~500 ms apart, pile onto the native lock, and overwrite the single global `chatState`. Pause/Stop are ineffective. | `MainViewModel.kt:236-238`, `ChatRepository.kt:149` |
| High | n_ctx is hard-coded to 2048 with no check that prompt + max_tokens fits. Overflow is either an error or a silently truncated answer presented as complete (`llama_jni.cpp:231-235, 388-391`). SHORT = 128 max new tokens often cuts list answers. | `GGUFLLM.kt:51, 787-793` |
| High | **JNI UTF-8:** `NewStringUTF` with raw UTF-8 output aborts under CheckJNI (debug) or garbles text (release) when output is cut mid-character. Input via `GetStringUTFChars` is Modified UTF-8/CESU-8, so emoji and math symbols reach the tokenizer corrupted. Pass `byte[]` both ways. | `llama_jni.cpp:182, 459` |
| High | `llama.cpp` is fetched at **`GIT_TAG master`** (shallow): builds are non-reproducible, need network, and break whenever the upstream API changes. Pin a commit. | `app/src/main/cpp/CMakeLists.txt:35` |
| Medium | Sampling: no repetition penalty, fixed seed 1234, and **greedy for LONG mode**. Small (0.5–1.5B) models loop. `<|im_start|>` is not a stop string. | `llama_jni.cpp:342-351, 394-411`; `GGUFLLM.kt:1032` |
| Medium | `parse_special=true` on the whole prompt: a document containing `<|im_end|>` becomes a real control token (prompt injection). | `llama_jni.cpp:213` |
| Medium | Native model and ONNX sessions are never freed (no `onCleared`, `freeModel` or `close`). Recreating the ViewModel loads a second context. | `MainViewModel.kt` |
| Medium | `resolveModelFile` runs on **every** `generate`: it lists directories, scans `/data/local/tmp`, and may silently switch to "the first .gguf found". | `GGUFLLM.kt:250-326` |
| Low | x86_64 is built with forced `-mavx2 -mfma` (SIGILL on older emulators/hosts). No streaming. Every token is logged at INFO. | `CMakeLists.txt:14-20`, `llama_jni.cpp:438` |

### 8.3 Post-processing

- **Uncommitted change:** native LLM output now **bypasses `CitationValidator.validate` entirely** (`GGUFLLM.kt:1138-1139`). Fabricated citations like `[Source 99, Page 42]` pass through.
- If the output lacks "[Source" and "Page", a citation to the top block is **appended unconditionally**, including to refusals ("I couldn't find…" + `[Source 12, Page 3]`) (`GGUFLLM.kt:1061-1066`).
- `cleanAndRenumberList` runs twice. It renumbers inline "N. Word" patterns (corrupting "Chapter 3. Results"), deletes number runs, and **collapses all double spaces**, which breaks code indentation in answers (`CitationValidator.kt:872-960`; applied at `GGUFLLM.kt:882, 1040, 1070`).
- **LLM-free answers:** if the native lib fails to load, ~1,000 lines of extractive "synthesizers" produce an answer that is shown as if the LLM wrote it (`GGUFLLM.kt:1101-1131, 1345-2424`). `synthesizeGeneralAnswer` has **canned answers** for photosynthesis, relativity, Fibonacci, primes, etc., and a generic invented "pros & cons" template (`GGUFLLM.kt:596-768`).

---

## 9. Overfitting to the test document (the biggest generalisation risk)

The repo contains `Grafting_15_Page_Guide.pdf`, and the androidTests and many unit tests are built around it. The production code has been tuned until those tests pass:

| Location | Grafting/test-document-specific logic |
|---|---|
| `ai/validation/CitationValidator.kt:733-815` (`isClaimRelevantToQuery`) | Term lists containing `scion, rootstock, cambium, callus, sucker, cleft grafting, 21c, 27c, 70f, 80f, 85%, weekly, aftercare, wax, pruning shears`. An explicit `if (… qLower.contains("grafting"))` branch. **Any question containing "tool", "material", "temperature", "healing", "cause", "fail" or "check" deletes every LLM claim that lacks these terms.** `asksCheck` uses `qLower.contains("do")`, which matches "does" and "document". `"Ac"`/`"Af"` are compared against lower-cased text and never match. |
| `ai/retrieval/VectorStore.kt:1115-1128` | Penalties for `aftercare`, `post-graft`, `weekly inspection`, `troubleshooting`, `tools required`. |
| `ai/retrieval/VectorStore.kt:1170-1183` | NASA STI report boilerplate phrases. |
| `ai/retrieval/QueryIntentRouter.kt:99` | `healing temperature`. |
| `ai/reformulation/QueryReformulator.kt:108` | Comment and examples: "grafting", "rootstock". |
| `ai/llm/GGUFLLM.kt:282, 1877-1880, 2018-2025, 2047` | Résumé sections (`projects`, `skills`, `certifications`), battery/thermal bonuses, "standard form 298". |
| Tests | ~350 grafting references across `AnswerLengthControlTest` (163), the androidTests, `CitationValidatorTest`, `ChatExporterTest` and `QuestionQueueTest`. |

The code comments claim "Generic, deterministic … without using hardcoded document-specific keywords" (`QueryIntentRouter.kt:57-59`, `GGUFLLM.kt:37`). **That claim is false.** Every one of these rules reduces quality on any other document.

---

## 10. Tests

- `./gradlew testDebugUnitTest`: **236 tests, 0 failures** (19 suites).
- They do not validate real RAG quality:
  1. JVM tests use `computeDeterministicMockEmbedding` (word-hash vectors), so **semantic retrieval is never tested**. Only lexical overlap is.
  2. Most assertions encode grafting-document answers, so they lock in the overfitting.
  3. No test covers chunk size vs embedder length, the evidence budget, `deleteChunksForDocument`, re-indexing, or the reformulator `"the"` bug.
- The androidTests (real ONNX + llama) are the only real signal, and they also target the grafting PDF.

**Recommended:** build a small **document-agnostic evaluation set**: 3–5 different PDFs (résumé, technical manual, research paper, FAQ, scanned PDF) with 20–30 Q/A pairs each, including paraphrased questions. Measure **retrieval recall@k** (is the gold chunk in the context?) separately from **answer correctness**. Run it on-device with the real embedder.

---

## 11. App-level issues (summary)

| Sev | Issue | Evidence |
|---|---|---|
| High | Asking about a document still `PROCESSING` → "could not find" (chunks are inserted only at the end). The chat picker does not check status. | `DocumentRepository.kt:124`, `MainActivity.kt` doc picker |
| High | Selecting **"All documents"** in the chat picker sets `selectedDocId = null`, which **silently disables RAG** (plain LLM chat). | `MainActivity.kt:840` |
| Medium | `textInput`/`selectedDocId` use `remember` (not `rememberSaveable`): after a rotation the next question silently runs without the document. | `MainActivity.kt:775-778` |
| Medium | Conversations store a single `documentId`; multi-document chat is not persisted. | `ConversationEntity`, `MainViewModel.kt:273` |
| Medium | `chatState` is global: the spinner/error shows in all conversations, and errors are shown twice (message + card). | `ChatRepository.kt:60, 341`; `MainActivity.kt:1115` |
| Medium | `messagesForConversation` creates a new `stateIn(Eagerly)` flow per call (leaks one collector per visit/rotation). | `MainViewModel.kt:336` |
| Medium | `MemoryGovernor` measures only the JVM heap; llama and ORT memory is native, so it always reports SAFE. | `ai/governor/MemoryGovernor.kt:44-50` |
| Medium | Model downloads: no SHA-256/size check, no resume, no free-space check, no cancel. The manifest declares foreground-service permissions but **no service exists**. | `ModelDownloader.kt:110-121`, `AndroidManifest.xml:6-9` |
| Medium | `allowBackup="true"`: document text goes to cloud backup, and multi-GB model files break the 25 MB auto-backup quota. | `AndroidManifest.xml` |
| Low | `ModelManager` scans `/data/local/tmp/models/llm` (dev backdoor) and uses the provider's `DISPLAY_NAME` as a filename (path traversal). | `ModelManager.kt:132, 300-310` |
| Maint. | God files: `MainActivity.kt` 2,775 lines, `GGUFLLM.kt` 2,425, `CitationValidator.kt` 1,512, `VectorStore.kt` 1,253. No DI. Status is a raw string. `java_original_preserved/` (3k+ lines) and dead native libs (~46 MB) are still in the source tree. | — |

---

## 12. Prioritised fix plan

### Phase 1: make the core pipeline correct (≈ 1–2 days, highest return)

1. **Fix chunk delete:** remove `{}` from `ChunkDao.deleteChunksForDocument`. Add `@ForeignKey(onDelete = CASCADE)` + `@Index("documentId")` on `ChunkEntity` (with a migration). Run a one-off cleanup of duplicate/orphan chunks.
2. **Align chunk size with the embedder:** target **~120–180 words (≤ 230 WordPiece tokens), with 1–2 sentences (~30 words) of overlap**. Better still, chunk by **token count using the same WordPiece tokenizer**, so nothing is ever truncated. Re-index existing documents.
3. **Fix the evidence budget:** derive it from the real n_ctx: `budget = n_ctx − system − question − max_new_tokens − template overhead`, counting metadata lines. With n_ctx 4096 and ~150-word chunks, that is 6–10 chunks. Raise n_ctx to 4096 (Qwen2.5 and Llama-3.2 support it), memory permitting.
4. **Fix the reformulator:** delete the `"definition"` / `"procedure guide"` suffixes and the `removePrefix("the")` bug. Pass the **original** user text to retrieval and route **once**, passing the `QueryClassification` into `VectorStore` instead of re-routing.
5. **Make dense retrieval first-class:** replace the hand-tuned score with standard **hybrid retrieval**. Take the top-k from dense (cosine) and the top-k from BM25 (exact tokens + a proper stemmer, **no substring matching**), fuse with **RRF only**, take the top 20, and drop the −35 "no keyword" penalty. Precompile all regexes. Cache the tokenised words per chunk (or store them at index time).
6. **Fix PDF word spacing** (`writeWordSeparator`) and fail imports with zero extracted text with a clear "scanned PDF, OCR not supported" message.

### Phase 2: generation correctness (≈ 2–3 days)

7. Pass `(contextBlocks, question, answerLength)` to the LLM as **structured data**; delete `parsePrompt` and ChatRepository's string prompt.
8. Use the model's own chat template (`llama_chat_apply_template` / GGUF `tokenizer.chat_template`); strip `<think>…</think>`.
9. **Delete:** the calculation aid, unconditional citation appending, the double `cleanAndRenumberList`, the canned `synthesizeGeneralAnswer`, the extractive "LLM-free" fallback (show a clear error instead), and the grafting/NASA/résumé-specific rules in `CitationValidator`, `VectorStore` and `GGUFLLM`.
10. Make citations simple and checkable: label blocks `[1] (p. 5)` and ask for `[1]`-style citations. Validate only that cited ids exist; do not delete sentences by keyword lists.
11. Add a repetition penalty (~1.1). Use temperature ~0.2–0.3 for LONG instead of greedy. Add `<|im_start|>` and the model's EOG tokens as stops. Turn off `parse_special` for document and user text (tokenise the template separately).

### Phase 3: robustness & UX (≈ 2–4 days)

12. Move all model loading to `Dispatchers.IO`. Use a separate lock for load vs inference. Add a native **abort callback** wired to coroutine cancellation and a Stop button. Add **token streaming** to the UI.
13. Make `sendMessage` a `suspend` fun (or return the `Job`) and serialise requests with a `Mutex`, so the question queue becomes truly sequential. Use per-conversation chat state.
14. Use one shared `ONNXEmbeddingEngine` (singleton). Make asset extraction atomic (write to temp, then rename). Use dynamic sequence length. Commit the ONNX model (Git LFS) or download it with a checksum.
15. Store `embeddingModelId` + `dim` per chunk, and re-embed on model change. Wire up (or remove) the embedding picker; BGE needs CLS pooling + a query prefix.
16. Insert chunks in batches during embedding. Mark stale `PROCESSING` documents as `FAILED` on startup. Run imports under WorkManager. Copy imported files into app storage (or use `OpenDocument` + `takePersistableUriPermission`).
17. Block or queue questions on documents that are still indexing. Fix "All documents" to pass all completed doc IDs.
18. Pin llama.cpp to a commit, pass JNI strings as `byte[]`, and delete `LlamaContext.java`, `librnllama*.so`, `java_original_preserved/` and the other dead classes.
19. Database: `exportSchema = true`, remove `fallbackToDestructiveMigration`, add indices.

### Phase 4: quality headroom (optional, once Phase 1–3 are done)

20. A **real reranker**: a small cross-encoder (e.g. `ms-marco-MiniLM-L-6-v2` ONNX INT8, ~23 MB) over the top 20 fused candidates. It is the same runtime you already ship.
21. Use the LLM (already loaded) for **query rewriting with conversation history**, i.e. condense follow-up questions into standalone queries, instead of regex rewriting.
22. Use a vector index (sqlite-vec or an in-memory HNSW) once documents exceed ~5k chunks. Add OCR (ML Kit text recognition) for scanned PDFs. Use real page numbers for DOCX where possible.

---

## 13. Quick reference: top 12 defects by impact

| # | Defect | File:line | Impact |
|---|---|---|---|
| 1 | Evidence budget (384) < one chunk → LLM gets 1 chunk | `ai/retrieval/ContextAssembler.kt:73, 376-378` | Incomplete or wrong answers on most questions |
| 2 | Embedding truncates at 256 tokens; chunks are 350–500 words | `ai/embeddings/ONNXEmbeddingEngine.kt:37`, `ai/chunking/TextChunker.kt:48-50` | ~Half of each chunk unsearchable semantically |
| 3 | `deleteChunksForDocument {}` is a no-op | `data/dao/ChunkDao.kt:29` | Duplicate chunks on reprocess, orphans on delete |
| 4 | Dense score ×10 vs lexical bonuses; −35 for no keyword overlap | `ai/retrieval/VectorStore.kt:849, 1015` | Paraphrased questions return "not found" |
| 5 | `removePrefix("the")` and "definition"/"procedure guide" suffixes | `ai/reformulation/QueryReformulator.kt:82, 85, 131` | Corrupted search queries |
| 6 | Double routing on different strings | `domain/repository/ChatRepository.kt:217-237`, `VectorStore.kt:137` | Retrieval strategy ≠ assembly strategy |
| 7 | Grafting-specific claim filtering | `ai/validation/CitationValidator.kt:733-815` | Correct answers deleted on other documents |
| 8 | `parsePrompt` takes "Question:" lines from the document | `ai/llm/GGUFLLM.kt:1170-1180` | Wrong question answered for FAQ/exam docs |
| 9 | Fake "verified arithmetic" from the first two numbers | `ai/llm/GGUFLLM.kt:486-525` | Confidently wrong numeric answers |
| 10 | Model load on main thread + shared monitor with inference | `MainViewModel.kt:93-146`, `ai/llm/LlamaRunner.kt:93, 214` | ANR / UI freeze |
| 11 | Timeout/cancel cannot interrupt native inference; queue not sequential | `domain/repository/ChatRepository.kt:199-280`, `MainViewModel.kt:236-238` | Hung requests, overlapping generations |
| 12 | PDF `writeWordSeparator` not overridden (verify) | `domain/rag/DocumentProcessor.kt:85-114` | Glued words in many PDFs → retrieval collapse |
