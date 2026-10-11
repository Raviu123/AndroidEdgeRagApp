# How OfflineDocAI works (current `unified-rag-v2` code)

This is a read-only explainer: what happens, in which file, in which order. Nothing was changed to write it.
Line numbers are approximate pointers into the current files.

---

## 0. Your question first: "is the RAG answer just the retrieved chunk?"

**Short answer: no. The language model *does* generate every document answer, but the current settings push it to copy the source text almost word for word.** That is why a well-written passage comes back nearly unchanged, while plain chat (no document) reads naturally.

### 0.1 Proof that generation happens
In a document chat, the only texts that ever become the assistant's answer are:

| Answer text | Comes from | Where |
|---|---|---|
| The model's output | `llm.generateGrounded(...)` | `domain/repository/ChatRepository.kt` (~line 302) |
| "I couldn't find this in the selected document(s)." | Fixed message: nothing retrieved, or nothing relevant (relevance gate) | `ChatRepository.kt` ~279, ~284, ~299 |
| "The selected document is still being indexed…" | Fixed message: document not ready | `ChatRepository.kt` ~259 |
| "Page N has no extractable text…" | Fixed message: page question with no text on that page | `ChatRepository.notFoundMessage` |

**There is no code path that puts chunk text straight into the answer.** The chunks only go *into the prompt* (and into the hidden `sources` field of the message, which the chat screen does not display; only the export uses it).

You can confirm this on the phone with logcat while asking a document question:
```
adb logcat -s ChatRepository LlamaRunner LlamaJNI GGUFLLM
```
For every document answer you should see, in order:
1. `ChatRepository: intent=GENERAL results=… blocks=3 evidenceTokens=…/1500 …` (what was retrieved and put in the prompt);
2. `LlamaJNI: Generation finished: prompt=1234 tokens, generated=87 tokens, …` (the model really produced `generated` new tokens);
3. `LlamaRunner: Inference finished in … ms`.

If you instead see `ChatRepository: Relevance gate: no passage is related to "…"`, the answer was the fixed "not found" message and the model was not called.

### 0.2 Why it looks copied (cause → where)
These are all in the current code, and they add up:

1. **Greedy decoding for document answers.** Document answers use temperature 0 (always the single most likely next token). Plain chat uses temperature 0.6 with sampling. Greedy decoding on a 0.5–1.5B model strongly prefers continuing text it has just read in the prompt, which is the source passage. *Where:* `ai/llm/GGUFLLM.kt` ~line 225 (`temperature = 0f`) vs ~234 (`CHAT_TEMPERATURE = 0.6f`, defined ~55).
2. **The system prompt asks for copying.** It says "Use only information stated in the sources", "After each statement, cite the source", and "Copy numbers, names, dates, units and quotations **exactly** as they appear". Small models over-apply "copy exactly" to whole sentences. *Where:* `ai/llm/PromptBuilder.kt` `groundedSystemPrompt`, lines ~25–35.
3. **Citation after every statement** makes the model work sentence by sentence through a source block, which is effectively extractive. *Same place.*
4. **Small chunks (128–180 tokens) that are already clean prose.** When one chunk already answers the question neatly, the most faithful answer *is* that text. *Where:* `ai/chunking/TextChunker.kt` lines 24–26.
5. **Small instruction models are extractive by nature** in grounded QA. That is usually desirable (fewer hallucinations), but it reads like copy-paste.
6. **Weak repetition control.** The repeat penalty (1.1) only looks at the last 64 *generated* tokens, not the prompt, so copying from the prompt is never penalised. *Where:* `GGUFLLM.kt` ~57 and `cpp/llama_jni.cpp` (`llama_sampler_init_penalties(…, 64, …)`).

So "something missing" is half right. Nothing is missing in the pipeline, but nothing *asks the model to synthesise* (combine and rephrase in its own words), and greedy decoding removes the variation you see in plain chat.

### 0.3 What could be changed (ideas only, not done)
- Use mild sampling for grounded answers (temperature ~0.3, top-p 0.9) instead of greedy.
- Rewrite the rules: "Answer in your own words, combining the relevant sources; quote only numbers, names and short exact phrases." Cite at the end of each paragraph instead of each statement.
- Put the question **before** the sources as well as after, so the model reads with the goal in mind.
- For short-answer questions, ask for "a one or two sentence direct answer first, then supporting details".
- Optionally increase chunk size again for BGE (trained on 512 tokens) so answers combine more context.

---

## 1. The big picture

```
                         ┌──────────── MODEL MANAGEMENT ─────────────┐
 Settings screen ──► MainViewModel ──► ModelDownloadManager ──► ModelDownloadWorker (WorkManager)
 (MainActivity.kt)        │                      └─► ModelFileDownloader (HTTP, resume, SHA-256)
                          ├─► ModelManager      (which .gguf language models are installed / active)
                          ├─► GGUFLLM           (loads + runs the language model) ─► LlamaRunner ─► LlamaNative ─► llama_jni.cpp ─► llama.cpp
                          └─► EmbeddingManager  (3 search models, health check, switching)
                                   ├─► ONNXEmbeddingEngine  (MiniLM / BGE via ONNX Runtime)
                                   └─► LlamaEmbeddingEngine (MiniLM GGUF via llama.cpp)

                         ┌──────────── INDEXING (when you import a document) ────────────┐
 Library ─Import─► DocumentRepository.importDocument
                    1 copy file into app storage
                    2 DocumentProcessor      → text per page (PDF / DOCX / TXT)
                    3 TextChunker            → chunks of ~128–180 tokens with overlap
                    4 EmbeddingManager       → one 384-number vector per chunk (batched)
                    5 Room database          → chunks table (text + vector + page + model id)

                         ┌──────────── ANSWERING (when you ask) ────────────┐
 Chat ─Send─► ChatRepository.ask
                    1 QueryIntentRouter      → what kind of question (general, page, summary, compare, list)
                    2 EmbeddingManager       → vector of the question (+ comparison sides / follow-up)
                    3 VectorStore            → best chunks (vector similarity + keyword BM25, fused)
                    4 relevance gate         → nothing related? reply "not found" without the model
                    5 ContextAssembler       → numbered source blocks that fit the token budget
                    6 PromptBuilder + GGUFLLM→ prompt (rules + sources + question) → model generates (streamed)
                    7 PromptBuilder          → clean output, fix citations
                    8 Room database          → message saved (answer + hidden sources JSON)
```

---

## 2. Indexing a document, step by step

| Step | What happens | File |
|---|---|---|
| 1. Register | A `DocumentEntity` row is inserted with status `PROCESSING`. | `domain/repository/DocumentRepository.kt` `importDocument` |
| 2. Private copy | The picked file is copied to `filesDir/documents/<id>-<name>`, so re-indexing later never needs the original file permission. | `DocumentRepository.copyToPrivateStorage` |
| 3. Extract text | **PDF**: PDFBox-Android, page by page (16 MB in RAM, the rest in a temp file). **DOCX**: Apache POI; pages from Word's page-break marks, Heading styles become `# Heading`, tables become `\| a \| b \|` rows. **TXT**: encoding detected (UTF-8 / UTF-16 / Windows-1252), pages split on form-feed. Scanned image PDFs → error (no OCR). | `domain/rag/DocumentProcessor.kt` |
| 4. Chunk | Lines become blocks (heading, list item, table row, paragraph); paragraphs split across pages are re-joined. Blocks are packed into chunks of **~128 tokens (max 180)**, measured with the embedding model's own tokenizer, with 1–2 sentences of overlap inside a section. A new heading always starts a new chunk. Each chunk keeps `pageNumber`, `pageSpan` ("4" or "4-5") and `sectionTitle`. | `ai/chunking/TextChunker.kt` (`VERSION = 3`) |
| 5. Embed | Each chunk (with its section title prepended) becomes a **384-number vector** from the *active* search model, 16 chunks at a time. | `DocumentRepository.runIndexing`, `ai/embeddings/EmbeddingManager.kt` |
| 6. Store | All chunks of the document are replaced in one transaction. The document becomes `COMPLETED` with `indexSignature = "chunker-v3|<embedding model id>"`. | `data/dao/ChunkDao.replaceChunksForDocument`, `data/entities/ChunkEntity.kt` |
| Re-index | At app start and after switching search model, documents whose signature differs (older chunker or different embedding model) are re-indexed automatically. | `DocumentRepository.reindexStaleDocuments` |

**Tokenizer:** WordPiece (BERT uncased) using `assets/models/embedding/vocab.txt`. The same vocabulary serves all three search models. *File:* `ai/embeddings/Tokenizer.kt` (class `WordPieceTokenizer`).

---

## 3. Answering a question, step by step

All in `domain/repository/ChatRepository.kt` unless noted.

| Step | What happens | File / function |
|---|---|---|
| 0. Queue | Requests run one at a time (`requestMutex`). Each has its own cancel token (Stop button, timeout). | `ask`, `runRequest` |
| 1. Previous turn | The last question/answer pair of this chat is loaded (for follow-ups). | `previousTurn` |
| 2. Ready check | Only documents indexed with the **currently active** search model are used. Otherwise you get "still being indexed". | `answerFromDocuments`, `DocumentRepository.isSearchable` |
| 3. Classify | Regex rules decide the intent: `PAGE_LOOKUP` ("what does page 4 say"), `PAGE_RANGE`, `SUMMARY`, `COMPARISON` ("compare X and Y"), `ENUMERATION` ("list all…"), else `GENERAL`. | `ai/retrieval/QueryIntentRouter.kt` |
| 4. Embed question | The question becomes a vector (BGE adds its "Represent this sentence for searching relevant passages:" prefix). Comparison sides and, for short follow-ups ("and the second one?"), "previous question + this question" are embedded as extra searches. | `EmbeddingManager.embedQuery`, `isFollowUp` |
| 5. Retrieve | **Page intents:** the chunks of those pages, in order. **Summary:** an even spread of up to 24 chunks across the document. **Otherwise hybrid search:** cosine similarity of vectors (top 50, ignoring <0.20) + BM25 keyword score (stemmed, stop-words removed, top 50), merged with Reciprocal Rank Fusion; top 20 (40 for lists) plus the neighbouring chunks of the best 3. Only chunks embedded by the active model are compared. | `ai/retrieval/VectorStore.kt` `retrieve`, `ai/retrieval/TextAnalyzer.kt` |
| 6. Relevance gate | If no chunk is above the model's similarity floor (MiniLM 0.30, BGE 0.50) **and** no chunk contains half of the question's content words → reply "I couldn't find this…" **without calling the model**. | `passesRelevanceGate` |
| 7. Budget | The model is loaded if needed. Evidence budget = context window − prompt skeleton − answer allowance − margin, capped at 1500 (general) / 2000 (compare) / 2400 (list, summary, pages) tokens. | `GGUFLLM.availableEvidenceTokens`, `evidenceCap` |
| 8. Assemble | Duplicates are removed; chunks are added in priority order until the budget is full; consecutive chunks are merged (overlap removed); blocks are numbered. General/compare questions put the best block first, others keep document order. | `ai/retrieval/ContextAssembler.kt` `assemble` |
| 9. Prompt | System rules + numbered sources + question (+ "follows up on: …"), formatted with the model's own chat template. | `ai/llm/PromptBuilder.kt` `groundedMessages`, `GGUFLLM.render` |
| 10. Generate | llama.cpp generates the answer (greedy, max 192 / 448 / 900 tokens for Short / Default / Long), streamed to the screen. Reasoning models get an extra thinking budget. The timeout starts here. | `GGUFLLM.generateGrounded` → `runInference` → `ai/llm/LlamaRunner.kt` `inference` → `cpp/llama_jni.cpp` `nativeInference` |
| 11. Clean up | Remove `<think>` blocks and stray template tokens; rewrite citations to the exact `[Source N, Page P]` tags; drop citations to sources that don't exist. If the answer is "not found", no sources are attached. | `PromptBuilder.cleanOutput`, `sanitizeCitations`, `isNotFoundAnswer` |
| 12. Save | The message is stored with the answer and a hidden `sources` JSON (block number, page, section, chunk ids, score, text). | `MessageEntity.sources` |

### 3.1 What the model actually sees (example, Qwen / ChatML)
```
<|im_start|>system
You answer questions using only the numbered sources provided by the user.
Rules:
- Use only information stated in the sources. Do not use outside knowledge and do not guess.
- After each statement, cite the source it came from by copying its tag exactly, for example [Source 2, Page 7].
- If the sources do not contain the answer, reply exactly: "I couldn't find this in the selected document(s)."
- If the sources answer only part of the question, answer that part and say what is not covered.
- Copy numbers, names, dates, units and quotations exactly as they appear in the sources.
- Answer clearly and completely: a short paragraph, or a list when the question asks for several items.<|im_end|>
<|im_start|>user
Sources:

[Source 1, Page 4] (Section: Leave Policy)
Full-time staff receive twenty days of paid annual leave in their first year...

[Source 2, Page 5]
...

Question: How many days of paid leave do staff get in their first year?<|im_end|>
<|im_start|>assistant
```
With greedy decoding and those rules, a 1.5B model's most likely continuation is close to "Full-time staff receive twenty days of paid annual leave in their first year [Source 1, Page 4]", which is the chunk's own sentence. That is the behaviour you noticed.

### 3.2 Document chat vs plain chat

| | Plain chat (no document) | Document chat (RAG) |
|---|---|---|
| Code path | `GGUFLLM.generateChat` | `GGUFLLM.generateGrounded` |
| System prompt | "helpful, honest assistant…" | strict grounding rules (above) |
| Decoding | temperature 0.6, top-p 0.9, top-k 40 (varied, natural) | **temperature 0 (greedy)**, deterministic |
| Context | previous Q/A turn | numbered source passages + previous question |
| Post-processing | think-block removal | + citation rewriting, not-found detection |

---

## 4. Model management

### 4.1 Language models (SLM)
| What | File |
|---|---|
| Catalog: Qwen 0.5B, Llama 3.2 1B, Qwen 1.5B, DeepSeek-R1 1.5B (URL, exact size, SHA-256) | `ai/llm/AvailableLlmModels.kt` |
| Installed models, active model (saved in prefs), import from file, delete | `data/ai/ModelManager.kt` |
| Reads model info (architecture, name, context length, quantisation) from the GGUF header | `ai/llm/GGUFMetadata.kt` |
| One per process: loads the active model, load state (Loading / Ready / Failed), prompt rendering, generation, reasoning budget | `ai/llm/GGUFLLM.kt` |
| Thread-safe wrapper: load/free, token counting, chat template, inference with cancel token and streaming | `ai/llm/LlamaRunner.kt` |
| Loads `libllama_jni.so`, picks the CPU variant, compatibility mode, crash sentinel, all JNI declarations | `ai/llm/LlamaNative.kt` |
| Native code: backend loading, model load, chat template, token-by-token generation, GGUF embeddings | `cpp/llama_jni.cpp` |
| Native build: llama.cpp pinned commit, 7 ARM CPU variants (`GGML_BACKEND_DL` + `GGML_CPU_ALL_VARIANTS`) | `cpp/CMakeLists.txt` |
| Prompts, chat templates fallback, output cleaning, citations | `ai/llm/PromptBuilder.kt` |
| Interfaces: `LocalLLM`, `GenerationRequest`, `ChatTurn`, `GenerationCancelledException` | `ai/llm/LocalLLM.kt` |
| Short / Default / Long | `ai/llm/AnswerLength.kt` |

### 4.2 Search (embedding) models
| What | File |
|---|---|
| Catalog of 3: MiniLM ONNX (built-in), BGE-small ONNX (CLS pooling, query prefix), MiniLM GGUF (llama.cpp) | `ai/embeddings/AvailableEmbeddingModels.kt` |
| Active model, install status, health check on this phone, switching, delete | `ai/embeddings/EmbeddingManager.kt` |
| ONNX Runtime engine (batched, mean or CLS pooling, L2 normalised) | `ai/embeddings/ONNXEmbeddingEngine.kt` |
| llama.cpp engine for the GGUF model | `ai/embeddings/LlamaEmbeddingEngine.kt` |
| `EmbeddingModel` interface (`getEmbedding`, `embedQuery`, `embedDocuments`, `modelId`) | `ai/embeddings/EmbeddingModel.kt` |
| WordPiece tokenizer | `ai/embeddings/Tokenizer.kt` |
| Bundled files | `assets/models/embedding/all-MiniLM-L6-v2.onnx` (downloaded and SHA-checked by Gradle at build time), `assets/models/embedding/vocab.txt` |

### 4.3 Downloads
| What | File |
|---|---|
| Per-model status, queue (one at a time), cancel, free-space check, cleanup of `.part` files | `data/download/ModelDownloadManager.kt` |
| Background foreground worker with notification, retries on network errors | `data/download/ModelDownloadWorker.kt` |
| HTTP download with resume (`Range`), redirects, size + SHA-256 + format check, atomic install | `data/download/ModelFileDownloader.kt` |

---

## 5. Storage (Room database `offline_docai.db`, schema v3)

| Table | Holds | Files |
|---|---|---|
| `documents` | name, type, page count, status (`PROCESSING`/`COMPLETED`/`FAILED`), private copy path, `indexSignature`, error | `data/entities/DocumentEntity.kt`, `data/dao/DocumentDao.kt` |
| `chunks` | text, page, page span, section, position, **vector (float32 BLOB)**, embedding model id + dimension; deleted with their document | `data/entities/ChunkEntity.kt`, `data/dao/ChunkDao.kt`, `data/FloatArrayConverter.kt` |
| `conversations` | chat title, linked document | `data/entities/ConversationEntity.kt`, `data/dao/ConversationDao.kt` |
| `messages` | role, text, timestamp, hidden `sources` JSON | `data/entities/MessageEntity.kt`, `data/dao/MessageDao.kt` |
| Migrations | 1→2→3, schemas exported to `app/schemas/` | `data/AppDatabase.kt` |

Files on disk (app private storage): `files/models/llm/*.gguf` (language models), `files/models/embeddings/*` (downloaded search models), `files/models/embedding/*` (extracted built-in model), `files/documents/*` (copies of imported documents).

---

## 6. Screens and glue

| What | File |
|---|---|
| All screens: chat, library, document detail, settings (model cards), drawer, dialogs | `MainActivity.kt` |
| App state and actions (startup, downloads, switching, delete, chat, queue) | `MainViewModel.kt` |
| Batch questions: parse a list, run one after another (each waits for its answer) | `ai/queue/QuestionQueueManager.kt`, `QuestionQueueParser.kt`, `QuestionQueueState.kt` |
| Answer rendering (Markdown) | `ui/components/MarkdownMessage.kt` |
| Export chat to PDF / JSON (JSON includes the retrieved chunks) | `ui/PdfExporter.kt`, `ai/export/ChatExporter.kt` |

**Tip for checking "copied or generated":** export a document chat as JSON (`ChatExporter`). Each answer is listed with its `retrievedChunks`, so you can put the answer and the source text side by side.

---

## 7. Useful logcat tags

| Tag | Tells you |
|---|---|
| `LlamaNative` | which CPU variant was loaded, compatibility mode |
| `LlamaJNI` | model loaded (n_ctx, threads), `Generation finished: prompt=… generated=…` |
| `GGUFLLM` | model ready, template family, reasoning model yes/no |
| `ChatRepository` | intent, number of results/blocks, evidence tokens, relevance gate, total time |
| `VectorStore` | top-5 chunks with fused/dense/BM25 scores |
| `EmbeddingManager` / `ONNXEmbeddingEngine` / `LlamaEmbeddingEngine` | search model load, health check result |
| `DocumentRepository` | indexing progress and failures |
| `ModelDownloadManager` / `ModelDownloadWorker` | downloads |
