# OfflineDocAI — End-to-End Evaluation Report

**Branch:** `unified-rag-v2` (working tree, uncommitted changes included)
**Date:** 2026-10-07
**Scope:** SLM catalog & download → SLM storage/registry → SLM load & inference → embedding model & its delivery → text extraction → chunking → embedding → storage → retrieval → context assembly → response generation.
**Mode:** Planning only. No source code was changed. The only file added is this report.

> **How this was evaluated.** I read every file on the RAG and model path, traced the runtime flow through the ViewModel and the Compose UI, checked the model catalog against the Hugging Face API, and inspected the llama.cpp build that CMake actually configured (`app/.cxx/Debug/...`). I did **not** run the app on a device. Every finding comes from reading the code, from the build configuration, or from verified external data. Each one cites a file and line.

---

## 1. Executive summary

| Stage | Verdict | Notes |
|---|---|---|
| SLM catalog (URLs, sizes, SHA-256) | ✅ Correct | All four URLs, byte sizes and SHA-256 values match Hugging Face exactly (checked live). Two size labels shown to users are wrong. |
| SLM download | ⚠️ **Buggy** | Validation is solid, but the download lifecycle and UI state are not. Two downloads can write to the same `.part` file and install a truncated model (**Bug D1**). There is no foreground service, no resume, no cancel and no free-space check. |
| SLM registry (ModelManager) | ⚠️ Works with gaps | GGUF metadata is parsed on the **main thread**. The selection state is split between two sources of truth. |
| SLM load / inference | ⚠️ **Buggy** | The "Ready" status is not linked to the real load result (**Bug L1**). Deleting a loaded model frees neither RAM nor disk (**Bug L2**). llama.cpp is built for **baseline ARMv8.0** (no dotprod/i8mm), so it is much slower and hits the 180 s timeout (**Bug L3**). DeepSeek-R1 often returns an "empty answer" (**Bug L4**). |
| Embedding model & delivery | ⚠️ **Regression** | The one bundled model is solid (SHA-verified, atomic extraction). But the **3-model choice from the `ravi` branch (download / select / delete) was removed**, so phones that can't run ONNX Runtime have no alternative. Restoring it properly is planned in **§8**. |
| Text extraction | ⚠️ OK for PDF | DOCX and TXT become a single "page 1", so page citations and page queries are meaningless for them. There is no OCR, and large PDFs are loaded fully into RAM. |
| Chunking | ✅ Good | Structure-aware, measured in embedding tokens, with sentence overlap. Chunks of 160–200 tokens are longer than MiniLM's 128-token training length. |
| Embedding | ✅ Correct, slow | Mean pooling and L2 normalisation match sentence-transformers. Chunks are embedded one at a time (no batching). |
| Storage (Room) | ✅ Good | v3 schema with FK cascade, index, migrations and exported schema. `allowBackup=true` can copy private documents to cloud backup. |
| Retrieval | ✅ Good design | Hybrid dense + BM25 with RRF, intent routing and neighbour expansion. The weak relevance floor means "not found" is mostly left to the LLM. |
| Context assembly | ✅ Good | Token-budgeted, de-duplicated, with overlap-aware merging and canonical citation tags. |
| Response generation | ⚠️ Works with gaps | Uses the native chat template with good prompt-injection hygiene and citation sanitising. There is **no conversation memory**, and the **question queue doesn't wait for answers** (**Bug Q1**, high). |

**Overall:** The RAG core (chunk → embed → hybrid retrieve → budget → grounded prompt) is well engineered and document-agnostic. The weak areas are the **model lifecycle**: download, selection, load, release and status reporting. Native build performance and two concurrency bugs (download and question queue) are the next biggest problems.

---

## 2. End-to-end flow (as implemented)

```
                       ┌──────────────────────── MODEL SIDE ────────────────────────┐
 Settings UI ──Download──▶ MainViewModel.downloadLlmModel ─▶ ModelDownloader (HTTP, .part,
 (MainActivity:2533)       (MainViewModel.kt:123)            size + GGUF magic + SHA-256)
                                    │ checkModels() on Main thread
                                    ▼
                           ModelManager.refreshModels  ── GGUFMetadataParser.parse (main thread!)
                                    │ active model (SharedPreferences)
                                    ▼
                           selectLlmModel ─▶ GGUFLLM.selectModel ─▶ warmUp() [result ignored]
                                                     └─▶ LlamaRunner.loadModel ─▶ JNI nativeInit (llama.cpp)

                       ┌──────────────────────── DOCUMENT SIDE ─────────────────────┐
 Library ─Import─▶ DocumentRepository.importDocument
                    1. copy to filesDir/documents        (DocumentRepository.kt:218)
                    2. DocumentProcessor: PDF/DOCX/TXT    (DocumentProcessor.kt)
                    3. TextChunker (WordPiece-token sized)(TextChunker.kt)
                    4. ONNXEmbeddingEngine (MiniLM-L6 int8, 384-d, one chunk at a time)
                    5. Room: chunks(BLOB) replace-in-transaction, doc → COMPLETED + signature

                       ┌──────────────────────── QUERY SIDE ────────────────────────┐
 Chat ─▶ ChatRepository.sendMessage (fire-and-forget coroutine, 180 s watchdog)
          ├─ no docs  → GGUFLLM.generateChat
          └─ docs     → QueryIntentRouter.route → embed question (+ comparison sides)
                        → VectorStore.retrieve (dense cosine + BM25, RRF, neighbours /
                          page fetch / summary sampling)
                        → GGUFLLM.availableEvidenceTokens (loads model if needed)
                        → ContextAssembler.assemble (token budget, merge, number)
                        → GGUFLLM.generateGrounded (native template, fit-guard, inference)
                        → PromptBuilder.cleanOutput + sanitizeCitations → message + sources JSON
```

---

## 3. The SLM download & load bugs (primary request)

The catalog itself is **not** the problem. I verified it against `huggingface.co/api/models/.../tree/main`:

| Model | Catalog size | HF size | Catalog SHA-256 | HF SHA-256 |
|---|---|---|---|---|
| qwen2.5-0.5b-instruct-q4_k_m | 491,400,032 | 491,400,032 ✅ | 74a4da8c… | 74a4da8c… ✅ |
| Llama-3.2-1B-Instruct-Q4_K_M | 807,694,464 | 807,694,464 ✅ | 6f85a640… | 6f85a640… ✅ |
| qwen2.5-1.5b-instruct-q4_k_m | 1,117,320,736 | 1,117,320,736 ✅ | 6a1a2eb6… | 6a1a2eb6… ✅ |
| DeepSeek-R1-Distill-Qwen-1.5B-Q6_K | 1,464,178,784 | 1,464,178,784 ✅ | c9c8aff5… | c9c8aff5… ✅ |

All four URLs return `302 → us.aws.cdn.hf.co/xet-bridge…` with `X-Linked-Size` matching. `ModelDownloader` follows redirects manually (≤ 8), so this works.

The real bugs are in the **lifecycle and state handling** around download and load.

### 🔴 Bug D1 — Two downloads can write to the same `.part` and install a truncated model

**Where:**
- `MainActivity.kt:2538` — `isDownloading = currentlyDownloadingLlm == model.fileName && llmDownloadState is Progress`
- `MainViewModel.kt:123-140` — `downloadLlmModel` has no guard, keeps no `Job`, and shares one `llmDownloader`
- `ModelDownloader.kt:45-47` — deletes any existing `.part` at start
- `ModelDownloader.kt:127-152` — validates the **streamed** byte count and digest, then renames whatever file is now at the `.part` path

**What goes wrong:**
1. There is only **one** `ModelDownloader` and **one** `currentlyDownloadingModel`, but every catalog card's *Download* button stays enabled.
2. Start downloading model A, then tap Download on model B. `currentlyDownloadingModel` becomes B, so card A **loses its progress bar and shows "Download" again**, even though A is still downloading in the background. Both coroutines also write progress into the same `downloadState`, so the progress jumps between A and B.
3. When either one finishes, `currentlyDownloadingModel` is set to `null` (`MainViewModel.kt:136-138`). The card of the one still running then shows "Download" again too.
4. Tap "Download" on a card that is still downloading. The new coroutine **deletes the `.part` file** (the first writer keeps writing to the now-unlinked inode) and starts a new `.part`. When the first coroutine finishes, its checks pass because they use the bytes it streamed, not the file on disk. It then **renames the second, half-written `.part` to the final `.gguf`**.
5. `checkModels()` → `refreshModels()` sees a GGUF with a valid header, marks it INSTALLED and auto-selects it. `warmUp()` → llama.cpp then fails ("tensor data is not within file bounds") → the user sees *"Unable to initialize the local LLM"*. Meanwhile the second coroutine reports *"The downloaded file is not a valid GGUF model"*, because its `.part` was moved away.

**Repro:** Settings → Download *Qwen 0.5B* → while it runs, tap Download on *Llama 1B* → the Qwen card now shows "Download" → tap it → wait for both to finish.

**Fix plan:**
- Allow only one download per file, and ideally one at a time overall. Keep the `Job` in a `Map<fileName, Job>` and ignore or deny a second start.
- Use a **per-download state map** (`Map<String, DownloadState>`) instead of one shared `StateFlow`.
- Disable every Download button while any download is active, and add a Cancel button.
- Give each attempt its own temp file name (for example `name.<uuid>.part`). Before renaming, re-check the **on-disk** file length against the expected size.

### 🔴 Bug L1 — "Ready" / "Active" is not tied to whether the model actually loaded

**Where:**
- `MainActivity.kt:2429` — the badge shows `"Ready"` whenever `activeModel != null`
- `MainViewModel.kt:114` — `viewModelScope.launch(Dispatchers.Default) { chatRepository.warmUpModel() }`; the result is discarded
- `GGUFLLM.kt:82` — `warmUp()` returns a `Boolean` that nothing reads
- `LlamaRunner.runnerState` (LOADING / READY / ERROR) and `ModelManager.updateModelStatus` / `ModelStatus.LOADING/LOADED` are **never used** (`grep` finds no callers)

**Effect:** When a load fails (truncated file, too little RAM, unsupported architecture), Settings still shows **Ready**. The user only finds out when the first question fails after a long wait. Load progress is also invisible: the first question after selecting a model silently pays the full load time inside the "Generating" spinner.

**Related — selection can point at two different models:**
- `GGUFLLM.selectModel` (`GGUFLLM.kt:63-68`) stores `selectedModelFile` **even when** `ModelManager.setActiveModel` refuses the model, which it does for any model whose status is not INSTALLED (`ModelManager.kt:205`).
- The UI marks a card as selected when *either* `activeModel` *or* `selectedLlmModel` matches (`MainActivity.kt:2537`).
- So a model in ERROR state can show as "Active" and be the one GGUFLLM tries to load, while ModelManager's active model is a different file.

**Fix plan:**
- Expose a single `ModelLoadState` (Idle / Loading(name) / Ready(name, nCtx) / Failed(name, reason)) from GGUFLLM, built from `LlamaRunner.runnerState`. Drive the badge and a "Loading model…" indicator from it.
- Make ModelManager the **only** source of truth for the active model. `GGUFLLM.resolveModelFile()` should read it and never keep its own `selectedModelFile`. Refuse to select models in ERROR state in the UI.
- Show the warm-up failure reason in Settings and offer *Retry* or *Delete & re-download*.

### 🔴 Bug L2 — Deleting a loaded model frees neither RAM nor disk

**Where:** `ModelManager.removeModel` (`ModelManager.kt:223-269`) deletes the file but never calls `GGUFLLM.release()` or `LlamaRunner.freeModel()`. `MainViewModel` has no `onCleared()` that releases the native model either.

**Effect:** llama.cpp loads the model with `mmap` (`llama_model_default_params()`, `llama_jni.cpp:91`). Deleting an mmapped file unlinks it, but its **disk blocks stay allocated until the mapping is released**. So "Delete" on the active 1.1–1.5 GB model does not free storage. If no other model is installed, nothing reloads, so the old model stays mapped (and its KV cache allocated) until the process dies. The same leak happens when the Activity finishes while the process stays alive: the next `MainViewModel` creates a new `GGUFLLM` / `LlamaRunner` and loads a second copy.

**Fix plan:** Route deletion through a coordinator that (1) refuses while generating, (2) calls `llm.release()` if this model is the loaded one, then (3) deletes the file. Add `MainViewModel.onCleared()` → `release()`. Better still, make `GGUFLLM` a process-wide singleton like `ONNXEmbeddingEngine`.

### 🟠 Bug L3 — llama.cpp is built for baseline ARMv8.0 (no dotprod / fp16 / i8mm)

**Where:** `app/src/main/cpp/CMakeLists.txt:14-28`. In the configured build (`app/.cxx/Debug/.../arm64-v8a/CMakeCache.txt`), `GGML_NATIVE=OFF` (always the case when cross-compiling) and `GGML_CPU_ARM_ARCH` is empty. In `ggml-cpu/CMakeLists.txt` that branch adds **no `-march` flag**, so the NDK default `armv8-a` is used.

**Effect:** The quantised matmul kernels fall back to plain NEON. On phones from about the last 6 years (all support dotprod, many support i8mm), prompt processing and generation are typically **2–4× slower** than they could be. The previously shipped prebuilt libraries (`librnllama_v8_2_dotprod_i8mm.so` and others, now deleted) had these optimisations. With evidence caps of 1800–2600 tokens plus up to 900 new tokens, a 1.5B model on this build can exceed `GENERATION_TIMEOUT_MS = 180_000` (`ChatRepository.kt:50`). Users then see *"The model took too long to answer…"*, which looks like a model problem.

**Fix plan (revised after reviewing `ravi`, see §8.5):** Do **not** simply hard-code `GGML_CPU_ARM_ARCH=armv8.2-a+dotprod`. The `ravi` branch commit `dc9cbf7` fixed a **SIGABRT on MediaTek Helio (MT676x/677x/678x) phones with the dotprod build**, so a dotprod-only build would crash those phones. Instead:
- Choose the CPU variant **at runtime**, as `ravi` did, using `GGML_BACKEND_DL` + `GGML_CPU_ALL_VARIANTS`.
- Keep a **compatibility (safe) mode** that forces the baseline variant on blocklisted chips, or automatically after a crash during load.
- Make sure release builds use `Release`/`RelWithDebInfo` for the native code; the current cache shows `CMAKE_BUILD_TYPE=Debug` for the debug variant.
- Measure tokens/s before and after.

### 🟠 Bug L4 — DeepSeek-R1-Distill often produces an "empty answer" error

**Where:** `GGUFLLM.maxNewTokens` (`GGUFLLM.kt:110-117`: 192 / 448 / 900), `PromptBuilder.cleanOutput` (`PromptBuilder.kt:160-168`), `GGUFLLM.runInference` (`GGUFLLM.kt:162`).

**Effect:** R1 distills write a long `<think>…</think>` block before answering. When the token limit is reached before `</think>`, `cleanOutput` cuts everything from `<think>`, leaving `""`. `runInference` then throws *"Local neural model produced an empty answer"*, and the user sees *"Local generation failed"*. This is very likely with SHORT and DEFAULT lengths. R1 also performs poorly with a system prompt (llama.cpp's `deepseek3` template just prepends it).

**Fix plan:** Give reasoning models a separate, larger thinking budget (for example +1024 tokens). Or drop R1 from the catalog for RAG and replace it with an instruct model (for example Qwen2.5-3B or Llama-3.2-3B for 6 GB+ devices). If it stays, fold the system prompt into the user turn for this family.

### 🟡 Bug D2 — GGUF metadata is parsed on the main thread after every download or import

**Where:** `MainViewModel.checkModels()` runs on `Dispatchers.Main` (called from `init`, `importLlmModel`, `downloadLlmModel`, `deleteModelItem`) → `ModelManager.refreshModels()` (`@Synchronized`) → `GGUFMetadataParser.parse`. The parser walks **every** metadata entry, including the 128k–152k-element token arrays and 150k–280k merges, using **unbuffered `RandomAccessFile` reads plus seeks** (about 0.6–0.8 M syscalls). `ModelManager.getInstance` also calls `refreshModels()` from the ViewModel constructor.

**Effect:** A UI freeze of roughly 0.5–2 s right after each download, import or first launch, with ANR risk on slow storage. The result is cached in `llm_metadata.json` afterwards.

**Fix plan:** Move `refreshModels()` to `Dispatchers.IO`. Stop parsing once the needed keys are found: the token arrays come *after* `general.*` and `<arch>.context_length`, but the early-exit condition needs `general.parameter_count`, which most GGUFs don't have. Read through a buffered stream.

### 🟡 Bug D3 — Download lifecycle is not robust for 0.5–1.5 GB files

| Problem | Where | Effect |
|---|---|---|
| `FOREGROUND_SERVICE` / `FOREGROUND_SERVICE_DATA_SYNC` declared, but **no service or WorkManager exists** | `AndroidManifest.xml:6-9` | The comment claims "Bug #10 fix", but downloads still run in `viewModelScope`. They die when the Activity finishes and are at risk when the app is backgrounded or the device dozes. |
| No resume (`Range:` header) | `ModelDownloader.kt:45-47` deletes `.part` | One network drop restarts a 1.4 GB download from zero. |
| Orphaned `.part` after process death | — | Up to 1.5 GB of hidden waste in `filesDir/models/llm` until the same model is downloaded again. |
| No free-space check | — | Fails with ENOSPC only after downloading most of the file. |
| No cancel button | `MainActivity.kt:2697-2747` | The user cannot stop a download. |
| Wrong size labels | `AvailableLlmModels.kt:24, 37` | "390 MB" is really 491 MB and "700 MB" is really 808 MB. |
| A newly downloaded model is never auto-activated when another is active | `MainViewModel.kt:135` — `selectedLlmModel.value == null` is never true after `checkModels()` | Dead code; the intended behaviour is unclear. |
| Progress only when `Content-Length > 0` | `ModelDownloader.kt:112` | Progress stays at 0% if a CDN uses chunked encoding. |

**Fix plan:** Move the download into a `CoroutineWorker` with `setForeground` (dataSync), add `Range` resume, check `StatFs` first, clean up `*.part` on startup, and add a cancel action.

---

## 4. Stage-by-stage evaluation

### 4.1 SLM catalog & registry
- **Catalog** (`AvailableLlmModels.kt`): verified correct (see §3). Note that `GGUFLLM.SupportedModel` duplicates the catalog list. Keep a single source.
- **ModelManager**: imports from SAF with a `.part` file, header validation and atomic rename. It persists metadata JSON, remembers the active model in prefs, and falls back sensibly. Issues:
  - It also scans `/data/local/tmp/models/llm` (`ModelManager.kt:132`), a dev convenience that should be debug-only.
  - SAF import keeps the source's display name. Importing a file with the same name as an installed model silently overwrites it, even if it is the loaded model, which goes through the same mmap issue as L2.
  - `GGUFMetadataParser` fallback values are optimistic: an unknown file type is reported as "Q4_K_M" and an unknown parameter count as "1.5B" (`GGUFMetadata.kt` `fileTypeToQuant`, `extractParamFromFilename`). This mislabels models in the UI.
  - `skipValue` for strings longer than 10 MB does not advance the pointer, so every later read is misaligned (rare).

### 4.2 SLM load & inference (JNI, `llama_jni.cpp`, `LlamaRunner`, `GGUFLLM`)
**Good:**
- UTF-8 byte arrays across JNI, so emoji and non-BMP text work, and incomplete UTF-8 at the end is trimmed.
- Uses the model's own chat template, with a family fallback.
- Prompt token counting is exact (BOS plus parsed specials).
- `n_ctx` is capped at the model's training context, and the window size depends on device RAM.
- KV cache is cleared for every request.
- Cancel works through an abort callback plus a flag.
- Generation stops on EOG or any control token.
- Pinned llama.cpp commit (`e117148a…`). I checked that the JNI calls (`llama_sampler_init_penalties` 5-arg, `llama_memory_clear`, `llama_vocab_is_control`, `llama_chat_apply_template` without a model) match the header that was fetched.

**Issues** (beyond L1–L4):
- `nativeInference` **resets the cancel flag at entry** (`llama_jni.cpp:225`). A cancel requested during model load or retrieval is lost, so the watchdog's timeout cancel can be overwritten and generation runs to completion only to be thrown away as "timed out".
- `cancelInference()` reads `handle` without the lock while `freeModel()` may be freeing it (`LlamaRunner.kt:177`), which risks use-after-free if a model switch races a cancel.
- `loadModel` reloads whenever `chooseContextSize()` changes, and that depends on `MemoryInfo.lowMemory` (`GGUFLLM.kt:105`). So a transient low-memory flag can trigger a full reload between warm-up and the first question.
- The prompt is rendered and counted between `ensureLoaded()` and `inference()` without holding the runner. A model switch in that window would send a prompt formatted with the wrong template to the new model (`GGUFLLM.kt:136-149`).
- Sampling is fixed (temperature 0.2, top-p 0.9, top-k 40, repeat penalty 1.1 over 64 tokens) with a random seed, so answers are not reproducible. For grounded QA, consider greedy decoding or a fixed seed.
- `THREADS = 4` is hard-coded. Use the count of big cores where it can be detected.

### 4.3 Embedding model & its delivery
**Good:**
- `all-MiniLM-L6-v2` int8 ONNX (Xenova `model_quantized.onnx`). The Gradle task `ensureEmbeddingModel` (`app/build.gradle.kts`) downloads it at build time and **checks its SHA-256** (`afdb6f1a…`).
- It is bundled in assets and extracted to `filesDir` with temp-file + rename and a version marker.
- One process-wide instance shares the tokenizer between chunk sizing and the model input.
- The WordPiece tokenizer faithfully mirrors `BertTokenizer(do_lower_case=True)`.
- Mean pooling over all positions, which is correct because there is no padding, followed by L2 normalisation.

**Issues:**
- The model is stored twice on disk: about 23 MB in the APK and again in `filesDir`. ORT can create a session from bytes or a memory-mapped asset fd instead.
- `ASSET_VERSION` (`ONNXEmbeddingEngine.kt:40`) is a hand-maintained string. If the asset changes and nobody bumps it, the old model stays in use and the **index signature does not change**, so stored vectors silently mismatch query vectors. Derive the marker from the asset hash at build time (BuildConfig field).
- A fresh clone **cannot build offline**, because `preBuild` must reach Hugging Face. Document this or vendor the file via Git LFS.
- MiniLM is English-centric. Retrieval over non-English documents will be weak. If needed, consider `multilingual-e5-small` or `paraphrase-multilingual-MiniLM-L12-v2`.
- `isCjk` checks code points above 0xFFFF on a 16-bit `Char`, so supplementary CJK characters are never split (minor). Zero-width and format characters (Cf) are not stripped as HF does (minor).

### 4.4 Text extraction (`DocumentProcessor.kt`)
**Good:** Single-pass per-page PDF extraction with an explicit word separator fix, DOCX tables rendered as pipe rows (which the chunker's table detection then recognises), and clear errors for scanned PDFs, `.doc` files and empty files. The file is copied into private storage, so re-indexing does not depend on the URI grant.

**Issues:**
- **DOCX and TXT always become a single page 1** (`DocumentProcessor.kt:208-209, 230-231`). All citations read "Page 1", `PAGE_LOOKUP` / `PAGE_RANGE` queries are meaningless, and the "page" UI is misleading. Use logical sections instead (DOCX heading styles, explicit page breaks, or every N characters) and label them "Section n" rather than "Page".
- DOCX heading styles (`Heading1…`) are not passed to the chunker, which then has to guess headings from text shape. The style is free, reliable signal.
- DOCX headers, footers, footnotes, text boxes and content controls (SDT) are ignored.
- `PDDocument.load(stream)` loads the **whole PDF into memory**. Use `MemoryUsageSetting.setupTempFileOnly()` for large files.
- No OCR for image-only PDFs (stated in the error message). Consider ML Kit Text Recognition as an optional fallback.
- TXT is decoded as UTF-8 only and a BOM is not stripped. Other encodings (UTF-16, Windows-1252) are garbled.

### 4.5 Chunking (`TextChunker.kt`)
**Good:**
- Parses structural blocks (headings, lists, tables, paragraphs) and joins paragraphs across page breaks, including hyphen repair.
- Measures chunks in the **embedder's own tokens**, so nothing is truncated at embed time.
- Oversized blocks split at sentences, then words.
- 1–2 sentence overlap within a section, and a new section always starts a new chunk.
- Records `pageSpan`, `sectionTitle` and `paragraphIndex`, and versions the chunker (`VERSION = 2`) as part of the index signature.

**Issues:**
- Target and max sizes are 160 / 200 tokens (`TextChunker.kt:24-25`), but all-MiniLM-L6-v2 was **trained on ≤128-token inputs**. Quality degrades on longer inputs. Benchmark target sizes of 110–128.
- The embedding input also prepends `sectionTitle` (`DocumentRepository.kt:194-197`). This is good for recall, but it adds tokens beyond the chunk budget (still under 256).
- Overlap sentences carried from the previous page do not move the chunk's `pageNumber` and `pageSpan` (they come from the new units only), so a citation can point at page N for text printed on page N−1 (minor).
- `splitByWords` re-tokenises the growing window for every word, which is O(n²) on long run-on text (tables extracted as one line, minified text). Bounded by window size, so acceptable, but easy to make incremental.
- Heading heuristics (ALL CAPS, Title Case between blank lines) rarely fire on PDFBox output, because blank lines are inconsistent. That is fine by design ("conservative"), but `sectionTitle` will be null for most PDFs.

### 4.6 Embedding pass (`DocumentRepository.runIndexing`)
- Chunks are embedded **one at a time** (`DocumentRepository.kt:165-182`) with `setIntraOpNumThreads(2)`. Padded batches of 8–16 sequences with an attention mask would be 2–4× faster on large documents.
- All `ChunkEntity` objects are held in memory before a single transaction. That is fine for typical documents, but a 1000-page PDF uses about 1.5 KB of vector plus the text per chunk.
- One shared `importState`: two imports at once overwrite each other's progress in the UI. Indexing is serialised by `indexMutex`, but file copying is not.
- Deleting a document while it is being indexed: `replaceChunksForDocument` then fails on the foreign key, so the work is wasted but the database stays consistent. Cancelling the indexing job on delete would be cleaner.
- Re-index at startup (`reindexStaleDocuments`) correctly handles interrupted imports and signature upgrades, and keeps legacy chunks when the source is gone. Good.

### 4.7 Storage (Room)
**Good:**
- Schema v3 with FK `ON DELETE CASCADE`, an index on `documentId` and an index on `messages.conversationId`.
- Explicit migrations 1→2→3, with no destructive fallback, and the exported schema in `app/schemas`.
- Embeddings stored as a float32 BLOB with `embeddingModelId` and `embeddingDim`, and an atomic per-document chunk replace.

**Issues:**
- `android:allowBackup="true"` (`AndroidManifest.xml:12`) with no backup rules. Private document copies, chunk text and chat history in the DB can go to cloud backup, which conflicts with an "offline / private" product promise. Add `dataExtractionRules` and `fullBackupContent` that exclude `documents/`, `models/` and the DB, or set `allowBackup=false`.
- Messages store `sources` JSON that includes full chunk text for every answer, so the DB grows quickly. Store chunk IDs and resolve the text when needed.
- `embeddingModelId` is stored but retrieval only checks the vector dimension (`VectorStore.kt:285`). A future 384-d model would be compared silently. Filter by model ID.

### 4.8 Retrieval (`VectorStore.kt`, `QueryIntentRouter.kt`, `TextAnalyzer.kt`)
**Good:**
- A clean, document-agnostic hybrid design: dense cosine (top 50, floor 0.20) plus BM25 (Porter-stemmed, stop-worded, exact-stem matching), fused with RRF (k = 60) across the main query and the comparison sides, then neighbour expansion.
- Deterministic intents for page lookup, page range, summary (evenly spread sample where any prefix is balanced), comparison and enumeration.
- A per-document in-memory index, invalidated by a (count, maxId) signature.

**Issues:**
- **Weak "nothing relevant" gate:** any single BM25 term hit survives (`VectorStore.kt:129`). In practice retrieval almost always returns 20 chunks, so deciding "not found" is left entirely to a 0.5–1.5B model, which is where small models hallucinate. Add a calibrated gate, for example best dense < 0.30 **and** best BM25 below a threshold → return "not found" without calling the LLM.
- No re-ranking. A small cross-encoder (for example `ms-marco-MiniLM-L-6-v2` int8, about 23 MB) on the top 20 would noticeably improve precision for small context budgets.
- **SUMMARY** samples at most 24 chunks and the budget then keeps about 10–13 of them. For a 300-page document that covers about 3% of the text. Consider a map-reduce summary (per-section mini-summaries, cached) for long documents.
- **No conversation context:** follow-ups such as "and what about the second one?" are embedded alone and retrieve nothing relevant. Add lightweight query condensation using the previous turn, or include the last Q/A pair in retrieval.
- Brute-force scoring of every chunk on each query is fine up to about 10–20k chunks. Beyond that, consider an ANN index or pre-filtering by BM25.
- The ENUMERATION regex (`what are the …s`) is broad. Many ordinary questions become ENUMERATION (top-K 40, document-order evidence), which loses best-first ordering.
- Page intents need a "page …" phrase at the start or after a question verb, so "explain page 4" works but "what's on p4" does not (`QueryIntentRouter.kt:48-50`).

### 4.9 Context assembly (`ContextAssembler.kt`)
**Good:**
- Greedy priority packing with exact LLM token counts and a worst-case header cost per block.
- Document-order merging of consecutive chunks with overlap removal.
- Best-first block order for open questions.
- A canonical `[Source N, Page P]` tag that is exactly what the prompt asks the model to copy.
- Reports `isBudgetLimited`, which changes the prompt ("do not claim the list is complete").

**Issues:**
- `evidenceCap` (1800–2600 tokens, `ChatRepository.kt:57-61`) combined with the slow ARM build (L3) is the main latency driver. Prefill is roughly linear in prompt length, so tune these caps per model size after fixing L3.
- Merged groups report the max relevance of their members. One strong chunk can pull a long weak neighbour group to position #1. Minor.

### 4.10 Response generation (`ChatRepository.kt`, `PromptBuilder.kt`, `GGUFLLM.kt`)
**Good:**
- Strict grounded system prompt with an exact not-found phrase, partial-answer rule, copy-numbers-exactly rule and budget-limited caveat.
- Untrusted text is neutralised (`<|…|>`, `<｜…｜>`, `[INST]`, `<think>`…) before templating, which is real prompt-injection hygiene.
- Final fit-guard drops blocks until the prompt fits.
- Citation sanitiser rewrites or removes citations to sources that don't exist.
- No fabricated fallback: errors surface as messages.
- Watchdog cancels native generation.

**Issues:**
- 🔴 **Bug Q1 — the question queue does not wait for answers.**
  - `QuestionQueueManager.startQueue` calls `onExecuteQuestion(...)` and treats it as finished when it returns (`QuestionQueueManager.kt:77`). But the callback is `ChatRepository.sendMessage`, which is **fire-and-forget**: it just `scope.launch`es and returns at once (`ChatRepository.kt:100`).
  - Result: all N questions start about 500 ms apart and run at the same time. They queue up behind the `@Synchronized` inference lock, and each one's 180 s watchdog starts **at launch time**. Later questions time out while waiting, and their watchdog calls `llm.cancel()`, which **aborts whichever question is generating at that moment**.
  - The shared `chatState` flickers between them, and the queue reports "completed" right away.
  - **Fix:** add `suspend fun ask(...)` that runs the same pipeline and returns when the answer is stored. Use it from both `sendMessage` (wrapped in a launch) and the queue. Give each request its own cancel token, so the watchdog of one request cannot cancel another, and start the watchdog only when the request reaches the model.
- **No chat history** in any prompt. Each turn is single-shot (see §4.8).
- The 180 s timeout is fixed. It does not scale with model size, answer length or measured device speed, and it includes model load and retrieval time.
- Output is not streamed. The UI shows a spinner for up to 3 minutes. Streaming tokens via a JNI callback would improve perceived speed a lot.
- `isNotFoundAnswer` matches only the exact phrase. Paraphrased refusals ("The document does not mention…") still get a sources block attached.

---

## 5. Testing status

- **Unit test run (`./gradlew testDebugUnitTest --offline`, 2026-10-07): ✅ 67 tests, 0 failures, 0 errors, 0 skipped** across 11 classes. The classes are the `rag/*` suite (chunker, tokenizer and stemmer, router, hybrid retrieval, context and prompt, document-agnostic guard), `GGUFMetadataParserTest`, `ModelManagerTest`, `AvailableLlmModelsTest`, `QuestionQueueTest` and `ChatExporterTest`.
- **Instrumented:** `RagPipelineAndroidTest.kt` (one file, not run here because it needs a device). The older benchmark and E2E tests were deleted in this branch.
- **Why the bugs above pass the tests:**
  - `QuestionQueueTest.testSequentialExecutionOrder` passes a *suspending* callback that really waits. Production passes `ChatRepository.sendMessage`, which returns at once, so Bug Q1 is invisible to the test.
  - No test covers `ModelDownloader` (redirects, validation, concurrent starts: D1), `GGUFLLM` selection and load state (L1), delete-while-loaded (L2), or Room migrations (`MigrationTestHelper` can use the now-exported schemas).
  - Two test source roots exist (`src/test/java` and `src/test/kotlin`). Consolidate them to avoid confusion.

---

## 6. Prioritised fix plan

### Phase 1 — Model management redesign: SLM **and** 3 embedding models  ⏱ ~4–5 days
Full design in **§8**. Summary:
1. One shared **`ModelDownloadManager`** for SLMs and embedding models:
   - per-model state, one download at a time, Cancel;
   - a unique `.part` file per attempt, size + SHA-256 + format checks, then an atomic install;
   - a free-space check and cleanup of orphaned `.part` files (fixes **D1**, part of **D3**).
2. Restore the **3 embedding options** (MiniLM-L6 ONNX, BGE-small ONNX, MiniLM-L6 GGUF) through an **`EmbeddingManager`** with:
   - per-model pooling and query prefixes;
   - a device health check before activating a model;
   - a confirmed, automatic **re-index** when switching.
3. A single source of truth for the active SLM and the active embedding model. Each card shows **real** states (Not downloaded → Downloading x% → Installed → Loading → Ready / Failed(reason)). Fixes **L1**.
4. Release native models before deleting their files; `onCleared()` release; process-wide singletons (**L2**).
5. Move `refreshModels()` and GGUF parsing off the main thread (**D2**).
6. Correct the size labels, and make the "download → use" behaviour explicit with a "Use this model" button and an "Use now?" prompt after a download finishes.

### Phase 2 — Generation robustness  ⏱ ~2 days
6. **Q1:** a suspend `ask()` API, the queue awaits each answer, per-request cancellation, and the watchdog starts at inference.
7. Don't reset the native cancel flag at the start of inference; take the cancel token per request.
8. **L4:** a reasoning-token budget for R1, or replace R1 in the catalog.
9. A relevance gate before calling the LLM ("not found" without generation).

### Phase 3 — Performance  ⏱ ~2–3 days
10. **L3:** runtime CPU-variant selection (`GGML_BACKEND_DL` + `GGML_CPU_ALL_VARIANTS`) plus a MediaTek-Helio / crash-triggered **compatibility mode** (§8.5); use a release native build type; benchmark tokens/s.
11. Stream tokens to the UI.
12. Batch embeddings during indexing.
13. Re-tune `evidenceCap` and the timeout per model size after measuring.

### Phase 4 — Download durability  ⏱ ~2 days
14. Run the Phase-1 `ModelDownloadManager` inside a WorkManager `CoroutineWorker` with a foreground `dataSync` notification, and add HTTP `Range` resume. Downloads then survive leaving the app and network drops, for both SLMs and embedding models.

### Phase 5 — Retrieval and extraction quality  ⏱ ongoing
15. Chunk-size experiment (≈128 tokens) against the current 160/200 on the existing retrieval tests.
16. DOCX: use heading styles as sections and logical sections instead of "Page 1"; TXT: encoding detection and logical sections.
17. Optional cross-encoder re-ranker; map-reduce summaries for long documents; follow-up query condensation.
18. Filter by `embeddingModelId`; derive `ASSET_VERSION` from the hash.

### Phase 6 — Privacy, hygiene and tests
19. `allowBackup` / backup rules to exclude documents, models and the DB.
20. Remove the `/data/local/tmp` scan from release builds, and remove (or actually implement) the `FOREGROUND_SERVICE` permissions.
21. Add tests for the downloader (MockWebServer, including two concurrent starts of the same file), GGUFLLM load state and delete-while-loaded, a queue test that uses the real fire-and-forget `sendMessage` contract, and Room migrations.

---

## 7. Severity index

| ID | Severity | Area | One-line |
|---|---|---|---|
| D1 | 🔴 High | Download | Two downloads can share one `.part` and install a truncated model; UI state is shared across models |
| L1 | 🔴 High | Load | "Ready" is not tied to the real load; warm-up failures are silent; selection can point at two different models |
| L2 | 🔴 High | Load / Delete | Deleting a loaded model frees neither RAM nor disk (mmap); no release on `onCleared` |
| Q1 | 🔴 High | Generation | Question queue fires every question at once; watchdogs cancel each other's inference |
| L3 | 🟠 Medium-High | Performance | llama.cpp built for baseline ARMv8.0, so 2–4× slower and causes timeouts |
| L4 | 🟠 Medium | Generation | DeepSeek-R1 thinking runs past the token limit, producing an "empty answer" error |
| D3 | 🟡 Medium | Download | No foreground worker or resume; orphan `.part` files; no space check; no cancel; wrong size labels |
| D2 | 🟡 Medium | Registry | GGUF metadata parsed on the main thread (jank / ANR risk) |
| G1 | 🟡 Medium | Generation | Native cancel flag reset at inference start; cancel races with free |
| R1 | 🟡 Medium | Retrieval | No relevance gate, so "not found" is left to a small LLM |
| R2 | 🟡 Medium | Chat | No conversation memory; follow-up questions fail |
| X1 | 🟡 Medium | Extraction | DOCX and TXT are a single "Page 1"; page citations and page queries are meaningless |
| P1 | 🟡 Medium | Privacy | `allowBackup=true` without exclusions |
| E1 | 🟢 Low | Embedding | `ASSET_VERSION` maintained by hand; asset stored twice on disk |
| C1 | 🟢 Low | Chunking | Chunk size above MiniLM's 128-token training length |
| M1 | 🔴 High | Embedding | The 3-embedding-model choice was removed; phones that can't run ONNX Runtime have no working search model (see §8) |

---

## 8. Addendum — Model download & switching redesign (3 embedding models + SLMs), based on the `ravi` branch

> Requirement: let the user **download, switch and delete three different embedding models**, because not every phone can run every backend. Make SLM and embedding download and switching **clear and correct**. Everything below is a plan only; nothing is implemented yet.

### 8.1 What the `ravi` branch did (commit `a238055`, reviewed file by file)

| Area | `ravi` implementation | Files (on `ravi`) |
|---|---|---|
| Embedding catalog | 3 options: **all-MiniLM-L6-v2 ONNX int8**, **BGE-small-en-v1.5 ONNX int8**, **all-MiniLM-L6-v2 GGUF Q4_K_M** | `ai/embeddings/AvailableEmbeddingModels.kt` |
| Embedding engines | `ONNXEmbeddingEngine(file)` (ONNX Runtime) and `GGUFEmbeddingEngine`, a wrapper that tries ONNX → GGUF (via prebuilt `librnllama` `LlamaContext.embedding`) → **hash vectors** | `ai/embeddings/*.kt`, `java/org/nehuatl/llamacpp/LlamaContext.java` |
| Download / select / delete | `MainViewModel.downloadEmbeddingModel / selectEmbeddingModel / deleteEmbeddingModel`; selection stored in prefs `selected_embedding_model`; the default model is auto-downloaded on first launch | `MainViewModel.kt:99-186` |
| Settings UI | "Installed Embedding Models" (tap to select, delete) + "Embedding Model Catalog (Download Options)" with progress / complete / error cards | `MainActivity.kt:1936-2060` |
| SLM loading | **Runtime CPU-variant selection** among 6 prebuilt `librnllama_*.so` (v8, v8.2, dotprod, i8mm, dotprod+i8mm, x86_64) using `/proc/cpuinfo`, with a **MediaTek Helio blocklist** that forces the baseline build (fix for a dotprod SIGABRT, commit `dc9cbf7`) | `LlamaContext.java:28-115` |
| SLM selection | prefs `selected_llm_model`; GGUFLLM reads it on every call and reloads when the path changes | `ai/llm/GGUFLLM.kt:28-66` |

### 8.2 Why it *seemed* to work on `ravi`, and what must not be copied

| `ravi` behaviour | Consequence |
|---|---|
| Downloads write **directly to the final file** (no `.part`), with no size, SHA or format check and a 30 s read timeout (`ModelDownloader.kt:24-92`) | An interrupted download leaves a partial `.onnx`/`.gguf` that is listed as installed and selectable. |
| `ONNXEmbeddingEngine` returns an **all-zero vector** on any failure; `GGUFEmbeddingEngine` then silently falls back to **768-d hash vectors** | Search "works" but is effectively random or keyword-hash. Chunks can mix 384-d and 768-d vectors. The failure is never shown to the user. |
| `GGUFLLM` silently falls back to an **extractive "synthesizer"** when the native model fails to load (`GGUFLLM.kt` ~line 120 onwards) | The app *appears* to answer even when no SLM is loaded. **This is the main reason it seemed to work.** |
| Engines are created **once** when `DocumentRepository` and `ChatRepository` are constructed (two separate `GGUFEmbeddingEngine(context)` instances) | Selecting another embedding model only writes prefs. It takes effect after an app restart, and documents are **not re-indexed**, so queries from model B are compared with chunks from model A. |
| BGE-small run with **mean pooling** and no query instruction | BGE is trained for **CLS pooling** (verified: `BAAI/bge-small-en-v1.5/1_Pooling/config.json` → `pooling_mode_cls_token: true`) and a query prefix. Its retrieval quality was degraded. |
| Startup marks every `PROCESSING` document as `FAILED`; DB built with `fallbackToDestructiveMigration()` | Interrupted imports are lost; a schema change wipes user data. |

**What we keep from `ravi`:** the 3-model catalog and the download / select / delete UX; the runtime CPU-variant selection and the MediaTek-Helio safe mode; offering an embedding model when none is usable.
**What we replace:** every silent fallback (errors are always shown), direct-to-final downloads, selection that needs a restart, and missing re-indexing.

### 8.3 Embedding catalog — verified on Hugging Face (2026-10-07)

| # | Option | Backend | Source file | Exact size (bytes) | SHA-256 | Dim | Pooling | Query prefix | Why offer it |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **all-MiniLM-L6-v2 (INT8)** — *default* | ONNX Runtime | `Xenova/all-MiniLM-L6-v2` → `onnx/model_quantized.onnx` | 22,972,370 | `afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1` | 384 | mean | — | Fast (6 layers), small, good general quality. Also bundled in the APK, so first launch works offline. |
| 2 | **BGE-small-en-v1.5 (INT8)** — *best retrieval* | ONNX Runtime | `Xenova/bge-small-en-v1.5` → `onnx/model_quantized.onnx` | 34,014,426 | `6c9c6101a956d62dfb5e7190c538226c0c5bb9cb27b651234b6df063ee7dbfe4` | 384 | **CLS** | `Represent this sentence for searching relevant passages: ` (queries only) | Higher retrieval accuracy; 12 layers, so about 2× slower to index. |
| 3 | **all-MiniLM-L6-v2 (GGUF Q4_K_M)** — *compatibility* | **llama.cpp** (our `llama_jni`) | `leliuga/all-MiniLM-L6-v2-GGUF` → `all-MiniLM-L6-v2.Q4_K_M.gguf` | 20,999,104 | `53533e550397f2ba4e627bd5833d2c791097372d861a52022a8586282c2178cc` | 384 | mean (GGUF has `pooling_type`; verified) | — | For phones where **ONNX Runtime fails**. It uses the same native engine as the SLM, so if the SLM runs, this runs. |

Notes:
- Options 1 and 2 use the **identical** WordPiece vocab: both `vocab.txt` files have git blob `fb140275c155…`, the same as ravi's `bert_vocab.txt`. The bundled `vocab.txt` and the existing `WordPieceTokenizer` serve both, so no extra tokenizer download is needed.
- Option 3 tokenizes inside llama.cpp. For chunk sizing, the shared WordPiece counter stays valid because it is the same MiniLM vocab.
- Backup source for option 3 if the leliuga file fails to load on the pinned llama.cpp: `second-state/All-MiniLM-L6-v2-Embedding-GGUF` → `all-MiniLM-L6-v2-Q4_K_M.gguf` (20,999,104 B, SHA `2ec4cee28a27a9c973d5f5230930d6ef6e52694bd2bc71be26a9bef5b1d755e6`).
- **Decision to confirm:** keep option 1 bundled in the APK (recommended: the app works offline from first launch; it adds about 23 MB) **or** make all three download-only (smaller APK, but nothing works until a download finishes). In the bundled case, card 1 shows "Built-in" instead of "Download", and it can't be deleted.

### 8.4 Target design

**A. One catalog model for both kinds**
```
ModelSpec(id, kind = LLM | EMBEDDING, backend = LLAMA_CPP | ONNX, displayName, fileName,
          url, sizeBytes, sha256, ramHint, badge, description,
          // embedding-only:
          dimension, pooling = MEAN | CLS, maxSeqLen, queryPrefix, documentPrefix)
```
`AvailableLlmModels` and a restored `AvailableEmbeddingModels` both produce `ModelSpec`. The duplicate `GGUFLLM.SupportedModel` enum is deleted.

**B. `ModelDownloadManager` (process singleton, shared by SLMs and embedding models)**
- `StateFlow<Map<modelId, DownloadStatus>>` with the states `Queued / Downloading(bytes, total, %) / Verifying / Installed / Failed(reason) / Cancelled`. Each card reads **its own** entry, so no shared state and no wrong card showing progress.
- **One download at a time.** Further requests queue, and tapping Download again on the same model does nothing (fixes D1).
- Per-attempt temp file `models/<kind>/<file>.<attemptId>.part`, plus a `StatFs` free-space check (size + 10%) before starting.
- Verify, in order: on-disk length = catalog size → SHA-256 → format check (`GGUF` magic for `.gguf`; for `.onnx`, a successful ORT session-create in the health check below). Then an atomic rename.
- Cancel button. At startup, delete `*.part` files that don't belong to an active download.
- Phase 4: runs inside a WorkManager foreground worker with `Range` resume.

**C. `EmbeddingManager` (process singleton, replaces direct `ONNXEmbeddingEngine.getInstance` use)**
- Engines behind one interface: `OnnxEmbeddingEngine(spec, file)` (mean **or CLS** pooling from the spec, query prefix applied only by `embedQuery()`) and `LlamaEmbeddingEngine(spec, file)`.
  - `LlamaEmbeddingEngine` uses new JNI functions in `llama_jni.cpp`: a **separate handle** from the SLM, `embeddings = true`, `pooling_type = MEAN`, `n_ctx = n_batch = n_ubatch = 512` (non-causal models need the whole input in one ubatch), and `llama_get_embeddings_seq`.
- API: `embedDocument(text)`, `embedQuery(text)`, `activeSpec`, `state: StateFlow<EmbeddingState>`.
- **Health check before activating** (the "does this phone support it" test):
  1. load the engine;
  2. embed a fixed probe sentence;
  3. check dimension = spec, no NaN/zero, norm ≈ 1;
  4. for options 1 and 3, check that the cosine to a stored reference vector is ≥ 0.95. This catches a broken backend on that chip.

  If any check fails, mark the model **"Not supported on this device: <reason>"**, keep the previous active model, and suggest the other backend.
- **First launch:** health-check the bundled option 1. If ONNX Runtime fails, show a one-time prompt: "Your phone can't run the default search model. Download the compatible version (21 MB)?" That download is option 3.
- **No silent fallbacks.** No zero vectors and no hash vectors. Failures are thrown and shown.

**D. Switching the embedding model = re-indexing**
- The index signature already contains the embedding model id (`DocumentRepository.CURRENT_INDEX_SIGNATURE`). It becomes `chunker-v<N>|<activeEmbeddingSpec.id>`, so every document indexed with another model is detected as stale.
- When the user taps **Use** on another embedding model: health check → confirmation dialog: *"Switching to BGE-small will re-index 7 documents (~3 min). Documents are unavailable for chat until re-indexed."* → on confirm, activate and **re-index in the background**, showing per-document progress in the Library.
- Retrieval loads only chunks whose `embeddingModelId == active id` (not just a dimension check), so vectors from different models are never mixed. Documents still re-indexing show "Updating search index…" in chat instead of wrong answers.
- Deleting the **active** embedding model is blocked. The user must switch first. The bundled option can't be deleted.

**E. SLM side (`ModelManager` + `GGUFLLM`)**
- `ModelManager` is the **only** source of the active SLM. `GGUFLLM` keeps no `selectedModelFile`.
- `GGUFLLM` becomes a process singleton exposing `llmState: Loading / Ready(nCtx) / Failed(reason)`, built from `LlamaRunner.runnerState`. Selecting a model triggers a warm-up whose **result is shown**.
- Delete flow: refuse while generating → `release()` if loaded → delete the file → pick the next model (or none) and show it.
- After a download finishes: card shows "Downloaded ✓ — [Use this model]". If no SLM is active it is activated automatically; otherwise the user is asked.
- No extractive fallback (already true on this branch; keep it).

**F. Settings screen (clear states)**
```
Language model (answers)                     Search model (embeddings)
┌───────────────────────────────────┐        ┌───────────────────────────────────────┐
│ Qwen 2.5 1.5B   ● Active · Ready   │        │ MiniLM-L6 (ONNX) ● Active · Built-in   │
│ Llama 3.2 1B    [Use] [Delete]     │        │ BGE-small (ONNX)  Downloading 42% [✕]  │
│ Qwen 2.5 0.5B   [Download 491 MB]  │        │ MiniLM-L6 (GGUF)  [Download 21 MB]     │
│ DeepSeek R1 1.5B ⚠ Failed: <why>   │        │   "Use if search fails on your phone"  │
│                 [Retry] [Delete]   │        │                                         │
└───────────────────────────────────┘        └───────────────────────────────────────┘
Header badge: "Model: Loading…" / "Ready" / "Failed – tap for details" (from real load state)
```
- Every action is disabled while it is invalid (for example Delete on the active or in-use model, or Download while another download runs; those buttons show "Queued").
- Errors stay on the card that failed, never in a shared banner.

### 8.5 Native library: speed **and** device compatibility (applies to the SLM and the GGUF embedder)
- Reintroduce `ravi`'s idea with our own JNI. Build llama.cpp with `GGML_BACKEND_DL=ON`, `GGML_CPU_ALL_VARIANTS=ON`. Then ggml ships `libggml-cpu-*` variants (armv8.0 … armv8.6+i8mm) and picks the best one using kernel HWCAP feature bits, which is more reliable than matching `/proc/cpuinfo` strings.
- **Compatibility mode** (keeps the `ravi` Helio fix):
  - (a) Use `ravi`'s MT676x/677x/678x/"helio" blocklist to load only the baseline CPU variant.
  - (b) Crash sentinel: write a flag before `nativeInit`/first decode and clear it on success. If the flag is still set at the next launch, enable compatibility mode automatically and tell the user.
  - (c) Add a manual toggle in Settings → Advanced.
- Verify on at least: one Helio G-series phone (crash check), one Snapdragon 7/8-series phone (speed check), and the x86_64 emulator.

### 8.6 Tests to add with this work
- `ModelDownloadManager` with MockWebServer: redirect chain, wrong size, wrong SHA, cancel mid-way, a second start of the same model (must not start), two different models (queued), orphaned `.part` cleanup.
- `EmbeddingManager`:
  - CLS vs mean pooling selection;
  - the query prefix is applied only to queries;
  - a failed health check keeps the previous model;
  - switching marks documents stale and retrieval ignores chunks from other models.
- Instrumented: embed the same sentence with options 1 and 3, cosine ≥ 0.95; load/unload each catalog SLM; delete-while-loaded frees the file (free-space check).

### 8.7 Files expected to change (implementation phase)
| New | Changed | Removed |
|---|---|---|
| `data/ai/ModelSpec.kt`, `data/ai/ModelDownloadManager.kt`, `ai/embeddings/AvailableEmbeddingModels.kt` (restored with verified data), `ai/embeddings/EmbeddingManager.kt`, `ai/embeddings/LlamaEmbeddingEngine.kt` | `ONNXEmbeddingEngine.kt` (spec-driven, CLS/mean, prefixes), `llama_jni.cpp` + `CMakeLists.txt` (embedding JNI, CPU variants), `LlamaRunner.kt` / `GGUFLLM.kt` (singleton, load state, release), `ModelManager.kt` (IO thread, single source), `DocumentRepository.kt` (signature from active embedder, re-index on switch), `VectorStore.kt` (filter by model id), `ChatRepository.kt` (`embedQuery`), `MainViewModel.kt`, `MainActivity.kt` Settings, `AndroidManifest.xml` | `domain/repository/ModelDownloader.kt` (replaced), `GGUFLLM.SupportedModel` enum |


---

## 9. Fix status (implemented 2026-10-07)

**Verification done here:**
- `./gradlew testDebugUnitTest`: ✅ **85 tests, 0 failures** (67 existing + 18 new).
- `./gradlew assembleDebug`: ✅ builds. The APK contains all 7 Android ARM CPU variants (`libggml-cpu-android_armv8.0_1` … `armv9.2_2`).
- **Not verified:** nothing was run on a phone (no device was connected), and the instrumented test `RagPipelineAndroidTest` was compiled but not executed. See "What to test on a phone" below.

### 9.1 Bug-by-bug

| ID | Status | What changed | Where |
|---|---|---|---|
| D1 | ✅ Fixed | New `ModelDownloadManager`: per-model status map; a second start of the same model is refused; one transfer at a time (others show "Queued"); Cancel. The final file is installed only after on-disk size + SHA-256 + format checks, so a corrupt file can never be installed. | `data/download/*` |
| D2 | ✅ Fixed | Model scan and GGUF parsing run on `Dispatchers.IO`. The parser uses a buffered little-endian reader and stops before the tokenizer tables. | `ModelManager.refresh()`, `GGUFMetadata.kt` |
| D3 | ✅ Fixed | WorkManager foreground worker (dataSync, notification with progress) that survives leaving the app, waits for network and resumes with HTTP `Range`. Also: free-space check, cleanup of orphaned `.part` files at startup, Cancel button, size labels derived from exact byte counts (491 MB / 808 MB / 1.1 GB / 1.5 GB), and an explicit "Use it now?" prompt after a download. | `ModelDownloadWorker`, `ModelFileDownloader`, `MainViewModel` |
| L1 | ✅ Fixed | `GGUFLLM.loadState` (NoModel / NotLoaded / Loading / Ready(n_ctx, CPU variant) / Failed(reason)) comes from the real native load and drives the Settings card, the drawer badge and Retry/Delete. ModelManager is the only source of the active model; models in ERROR state cannot be selected. | `GGUFLLM`, `MainActivity` |
| L2 | ✅ Fixed | Delete and import-overwrite release the native model first, so RAM and disk are freed immediately. `GGUFLLM` is a process singleton, so the model is never loaded twice. | `MainViewModel.deleteModelItem/importLlmModel` |
| L3 | ✅ Fixed | `GGML_BACKEND_DL` + `GGML_CPU_ALL_VARIANTS` on arm64: the best variant the CPU supports is loaded at runtime. **Compatibility mode** forces ARMv8.0 on MediaTek Helio (ravi's blocklist), automatically after a native crash (crash sentinel), or manually in Settings → Advanced. Threads use the performance-core count. | `CMakeLists.txt`, `llama_jni.cpp`, `LlamaNative.kt` |
| L4 | ✅ Fixed | Reasoning models get a separate thinking budget. If they are still thinking when it runs out, the reasoning is closed (`</think>`) and the answer is generated. DeepSeek's system prompt is folded into the user turn. Reasoning is hidden while streaming. | `GGUFLLM.runInference`, `PromptBuilder` |
| Q1 | ✅ Fixed | `ChatRepository.ask()` is a suspend function that returns when the answer is stored. The queue uses it, so questions really run one after another. Requests are serialised with a mutex. | `ChatRepository`, `MainViewModel.startQuestionQueue` |
| G1 | ✅ Fixed | Per-request `GenerationToken`. The native cancel flag is no longer reset inside inference, and a cancel racing the start is re-delivered. Cancel and free are synchronised. The watchdog starts when generation starts and stops only its own request. A Stop button keeps the partial answer. | `LlamaRunner`, `llama_jni.cpp`, `ChatRepository` |
| R1 | ✅ Fixed | Relevance gate: if no passage is above the model's similarity floor (MiniLM 0.30, BGE 0.50) **and** none contains half of the question's content words, the app replies "not found" without running the LLM. Paraphrased "not found" answers no longer get sources attached. | `ChatRepository.passesRelevanceGate`, `PromptBuilder.isNotFoundAnswer` |
| R2 | ✅ Fixed | Follow-up questions ("and the second one?", "why is that?") are also searched together with the previous question, and the grounded prompt mentions the previous question. Plain chat keeps the previous turn. | `ChatRepository.isFollowUp`, `PromptBuilder` |
| X1 | ✅ Fixed | DOCX pages come from Word's page-break marks (`lastRenderedPageBreak`, page breaks), so page citations and "page N" questions work. Heading styles become Markdown headings for the chunker. TXT: BOM / UTF-16 / Windows-1252 detection and form-feed pages. PDF: buffered through a temp file (16 MB in RAM). | `DocumentProcessor` |
| P1 | ✅ Fixed | `allowBackup="false"`: documents, chats and models never leave the device. | `AndroidManifest.xml` |
| E1 | ✅ Fixed | The bundled asset marker is the build-time SHA-256 (`BuildConfig.EMBEDDING_ASSET_SHA256`). | `build.gradle.kts`, `ONNXEmbeddingEngine` |
| C1 | ✅ Changed | Chunks are now 128 target / 180 max tokens (MiniLM's training length). The chunker version moved to 3, so documents re-index automatically on next start. Overlap carried from the previous page is included in the page span. The word-window split is no longer quadratic. | `TextChunker` |
| M1 | ✅ Fixed | **3 search models**: MiniLM-L6 ONNX (built-in), BGE-small ONNX (CLS pooling + query instruction, as the model expects), MiniLM-L6 GGUF via llama.cpp (for phones where ONNX Runtime fails). Download, Use and Delete for each, with a per-device health check before activating. Switching asks for confirmation and re-indexes in the background. Retrieval only compares vectors from the active model. There are no silent fallback vectors. | `ai/embeddings/*`, `DocumentRepository`, `VectorStore` |

Smaller items fixed along the way:
- Embeddings are batched (8 per ONNX run, padded with an attention mask).
- Deleting a document cancels its indexing.
- A context-size change no longer causes model reloads.
- Model switches can't interleave with prompt preparation (one mutex covers load, prompt and generation).
- Grounded answers use greedy decoding.
- The `/data/local/tmp` scan runs only in debug builds.
- Unknown quantisation or size is shown as blank instead of a fake "Q4_K_M / 1.5B".
- The duplicate `SupportedModel` enum was removed.

### 9.2 Not done (intentionally or needs a decision)
- **Cross-encoder re-ranker and map-reduce summaries for very long documents.** Both need another model download and an on-device quality evaluation first.
- **Room migration tests** (`MigrationTestHelper`) and merging the two unit-test folders (`src/test/java` + `src/test/kotlin`).
- Message `sources` still store the passage text, which the source viewer uses.
- The ENUMERATION intent regex and page-question phrasing ("what's on p4") are unchanged.
- `importState` is still one shared progress line when several documents are imported at the same time.

### 9.3 What to test on a phone
1. **Settings → Download a language model:**
   - While it downloads, the other cards say *Queued* if you tap them, and the same card can't be started twice.
   - Leave the app: the notification shows progress.
   - Turn off Wi-Fi mid-way, then back on: it resumes.
2. **First model:** it becomes active automatically and the card goes *Loading… → Ready* (with the CPU variant shown under Advanced).
3. **Second model:** after download, the "Use it now?" prompt appears. Switch, ask a question, then delete the old model; storage is freed immediately.
4. **Search model → Download BGE → Use:** the check runs, the confirmation shows the document count, the Library shows re-indexing, and answers work afterwards.
5. **Search model → MiniLM GGUF → Use:** this exercises the llama.cpp embedding path.
6. **DeepSeek-R1:** with the Default length, ask a question. It should show "Thinking…" and then an answer, not "empty answer".
7. **Question queue with 3 questions:** they are answered one after another, each in full.
8. **Stop button during an answer:** the partial text is kept with "_Stopped._".
9. **On a MediaTek Helio phone:** Settings → Advanced shows compatibility mode on and the `armv8.0` variant.
