# Section 2: Related Work & Comparative Positioning

This document provides a comprehensive academic survey of related literature across **On-Device RAG**, **Edge Machine Learning**, **Vector Indexing Optimization**, **Document Chunking**, and **Small Language Models (SLMs)**. It details what OfflineDocAI borrows from prior works, what it refines, and where it establishes distinct scientific novelty.

---

## 1. Taxonomic Landscape of Edge Retrieval-Augmented Generation

In recent years, the intersection of Retrieval-Augmented Generation and Edge Computing has garnered intense research focus. The table below provides a systematic taxonomy of the 8 leading edge RAG publications in literature and positions OfflineDocAI against each.

| Publication | Venue / Year | Target Hardware | Core Technique | Key Limitations Addressed by OfflineDocAI |
| :--- | :--- | :--- | :--- | :--- |
| **EdgeRAG**<br>*(Seemakhupt et al.)* | arXiv (UVA/Waterloo/Google, Dec 2024) | Edge Gateway / Server | Online indexing with pruned IVF vector clustering & on-demand re-embedding. | Evaluates vector retrieval in isolation; assumes pre-extracted text and lacks on-device document ingestion, structural chunking, and SLM generation. |
| **MobileRAG**<br>*(Park et al.)* | arXiv (KAIST, Jul 2025) | Smartphone CPU | `EcoVector` partitioned graph search + Selective Content Reduction (SCR). | Focuses on vector search and token pruning; ignores hierarchical document layouts (tables, lists, headers) and multi-turn context assembly. |
| **MobileRAG (Agent)**<br>*(Loo et al.)* | arXiv (UESTC/PolyU/Tsinghua, Sep 2025) | Mobile OS | Agentic GUI automation via `InterRAG`, `LocalRAG`, and `MemRAG`. | Built specifically for Android OS UI navigation and app automation; not designed for unstructured document parsing and factual QA. |
| **SRAS**<br>*(Muttur)* | arXiv (Dayananda Sagar, Jan 2026) | Edge CPU | Sparse Reward-Aware Selector: RL-based (PPO, 0.76MB) policy for top-$k$ document selection. | Requires domain-specific RL policy training; introduces latency overhead for policy inference. OfflineDocAI achieves dynamic budgeting deterministically with zero training cost. |
| **Pocket RAG**<br>*(Kang et al.)* | arXiv / Elsevier (Hanyang/UT Austin, Feb 2026) | Android Phone ($<2\text{GB}$ RAM) | Domain-specific offline RAG for emergency first-aid medical guidance; 3.7s TTFT. | Application-specific to fixed medical triage; lacks general-purpose multi-format ingestion (DOCX/PDF) and dynamic prompt injection defenses. |
| **RAGdb**<br>*(Khalid)* | arXiv (SKAS IT, Feb 2026) | Intel Core i7 Laptop | Monolithic SQLite container with sublinear TF-IDF + exact substring boosting. | Evaluated only on x86 consumer laptops; lacks ARM64 NEON optimizations, ONNX Runtime edge embeddings, and mobile-native memory lifecycle management. |
| **HeRo**<br>*(Li et al.)* | ACM/IEEE DAC '26 (Peking Univ, Mar 2026) | Heterogeneous Mobile SoC (CPU/GPU/NPU) | Hardware-aware scheduling, shape-aware partitioning, and memory bandwidth contention control. | Operates at the hardware scheduling layer; assumes pre-existing embeddings and does not address document layout parsing, lexical fusion, or citation validation. |
| **SmartRAG**<br>*(Jiang et al.)* | arXiv (Nanjing Univ/HUST, Jul 2026) | Commodity Smartphone | Native 3-layer Knowledge Graph (`MRGraph`) and continual entity recognition (`EvoNER`). | Heavy on-device Knowledge Graph construction overhead and high cold-start ingestion latency on budget phones. OfflineDocAI uses lightweight structural chunking and fast BM25/Dense RRF. |

---

## 2. Deep Dive: Related Works and Scientific Refinements

### 2.1 Online Indexing & Edge Vector Search
* **Prior Works (EdgeRAG, MobileRAG KAIST, RAGdb)**:
  - *EdgeRAG* addresses edge RAM limitations by clustering embeddings with Inverted File (IVF) indexes, pruning cluster members, and re-computing embeddings on-demand during query execution.
  - *MobileRAG (KAIST)* proposes *EcoVector*, a partitioned vector index that loads only required index partitions into RAM, combined with *Selective Content Reduction (SCR)* to eliminate irrelevant sentences before LLM evaluation.
  - *RAGdb* embeds all documents into a portable SQLite database and proposes a Hybrid Scoring Function (HSF) combining sublinear TF-IDF with substring boosting to bypass GPU inference.
* **OfflineDocAI Refinement & Advancement**:
  - OfflineDocAI integrates ONNX Runtime (`all-MiniLM-L6-v2`) with direct zero-copy `ByteBuffer` BLOB serialization (`FloatArrayConverter.kt`) in SQLite.
  - Rather than relying solely on pure vector search or pure lexical search, OfflineDocAI implements a **Multi-Signal Reciprocal Rank Fusion (RRF)** pipeline. It calculates exact Okapi BM25 scores over dynamically stemmed inverted term frequencies while simultaneously computing cosine similarities over dense embeddings, establishing a relevance floor ($\text{MIN\_DENSE} \ge 0.20$) to eliminate semantic noise.

### 2.2 Document Layout & Semantic Chunking
* **Prior Works**:
  - Traditional edge RAG pipelines rely on fixed character-length or word-length sliding windows (e.g., LangChain `RecursiveCharacterTextSplitter` or fixed 400-word windows).
  - Naive splitting frequently fractures tabular data across chunks, severs bullet points from introductory clauses, and loses section headers.
* **OfflineDocAI Refinement & Advancement**:
  - OfflineDocAI formulates a **Hierarchical Structure-Preserving Chunking algorithm (`TextChunker.kt`)**.
  - It models documents as a sequence of semantic blocks (`HEADING`, `LIST_ITEM`, `TABLE_ROW`, `PARAGRAPH`).
  - It detects cross-page split sentences and joins them, preserves entire table rows and list items as atomic units, associates parent heading titles as metadata to all descendant chunks, and bounds chunk sizes directly in **WordPiece tokens** (128 target, 180 ceiling) to match the embedding model's exact input capacity.

### 2.3 Query Routing and Document Selection Policies
* **Prior Works (SRAS)**:
  - *SRAS* trains a compact Reinforcement Learning policy (0.76MB PPO model) to dynamically select which top-$k$ documents to forward to the generator under token constraints.
* **OfflineDocAI Refinement & Advancement**:
  - While RL document selection is innovative, it requires training separate policies across document domains and consumes inference cycles on CPU.
  - OfflineDocAI implements a **deterministic 6-class Query Intent Router (`QueryIntentRouter.kt`)** that classifies user queries into structural intents (`PAGE_LOOKUP`, `PAGE_RANGE`, `SUMMARY`, `COMPARISON`, `ENUMERATION`, `GENERAL`) in $<1\text{ ms}$ using prioritized heuristic and regex rules.
  - This allows the system to bypass dense vector search entirely for deterministic queries (e.g., retrieving exact pages) and dynamically scale token budgets and candidate counts based on query complexity.

### 2.4 On-Device Neural Execution & Grounding
* **Prior Works (Pocket RAG, SmartRAG)**:
  - *Pocket RAG* optimizes Time-to-First-Token (TTFT) for emergency medical triage within Android's 2GB memory boundary using quantized Small Language Models.
  - *SmartRAG* organizes mobile intelligence into Perception, Memory, Focus, and Thinking modules backed by a 1.7B parameter model.
* **OfflineDocAI Refinement & Advancement**:
  - OfflineDocAI implements a native **C++17 JNI bridge (`llama_jni.cpp`)** to `llama.cpp` with ARM NEON and AVX2 SIMD acceleration.
  - It provides runtime model switching between **Qwen 2.5 (1.5B and 0.5B)** and **Llama 3.2 (1B)** with persistent in-memory context caching.
  - It introduces **Adversarial Control Token Sanitization (`PromptBuilder.kt`)**, which neutralizes untrusted chat-control sequences (`<|im_start|>`, `[INST]`, `<think>`) embedded in raw user documents to prevent indirect prompt injection.
  - It introduces **Algorithmic Boundary Overlap Deduplication (`ContextAssembler.kt`)**, which identifies and strips shared sentences across adjacent chunks, boosting prompt information density.

---

## 3. Positioning Summary Matrix

The following matrix summarizes the architectural capabilities of OfflineDocAI in direct comparison with existing systems:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ ARCHITECTURAL COMPARISON MATRIX                                                                                 │
├────────────────────────────────┬──────────┬───────────┬────────────┬──────────┬────────┬───────────┬────────────┤
│ Dimension / Feature            │ EdgeRAG  │ MobileRAG │ Pocket RAG │  RAGdb   │  SRAS  │ SmartRAG  │OfflineDocAI│
├────────────────────────────────┼──────────┼───────────┼────────────┼──────────┼────────┼───────────┼────────────┤
│ 100% On-Device & Air-Gapped    │    ❌    │    ✅     │     ✅     │    ✅    │   ✅   │    ✅     │     ✅     │
│ Multi-Format Parsing (PDF/DOCX)│    ❌    │    ❌     │     ❌     │    ✅    │   ❌   │    ❌     │     ✅     │
│ Hierarchical Structural Chunk  │    ❌    │    ❌     │     ❌     │    ❌    │   ❌   │    ❌     │     ✅     │
│ Dense + BM25 Hybrid RRF Search │    ❌    │    ❌     │     ❌     │    ✅    │   ❌   │    ✅     │     ✅     │
│ Deterministic Intent Routing   │    ❌    │    ❌     │     ❌     │    ❌    │   ❌   │    ❌     │     ✅     │
│ Boundary Overlap Deduplication │    ❌    │    ❌     │     ❌     │    ❌    │   ❌   │    ❌     │     ✅     │
│ Native C++17 JNI SIMD Inference│    ❌    │    ❌     │     ✅     │    ❌    │   ❌   │    ✅     │     ✅     │
│ Multi-SLM Dynamic Switching    │    ❌    │    ❌     │     ❌     │    ❌    │   ❌   │    ❌     │     ✅     │
│ Prompt Injection Sanitization  │    ❌    │    ❌     │     ❌     │    ❌    │   ❌   │    ❌     │     ✅     │
│ Verifiable Page-Level Citation │    ❌    │    ❌     │     ✅     │    ❌    │   ❌   │    ❌     │     ✅     │
└────────────────────────────────┴──────────┴───────────┴────────────┴──────────┴────────┴───────────┴────────────┘
```
