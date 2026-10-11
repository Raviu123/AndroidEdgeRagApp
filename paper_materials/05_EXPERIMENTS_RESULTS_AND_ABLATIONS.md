# Section 5: Experimental Evaluation, Results & Ablation Studies

This document provides complete benchmark data, experimental setups, quantitative accuracy comparisons, hardware resource profiling, and isolated ablation study results for the OfflineDocAI research paper.

---

## 1. Experimental Methodology & Evaluation Setup

### 1.1 Document Test Corpora
The system was evaluated across a diverse benchmark suite comprising heterogeneous unstructured document genres:
1. **Technical & Agricultural Manuals** (e.g., *Grafting 15-Page Technical Guide*): Multi-page hierarchical sections, complex step sequences, and fine-grained numerical metrics (temperatures, ratios).
2. **Standard Question Answering Benchmarks**:
   - **SQuAD 2.0 Subset**: Single-document fact extraction and negative unanswerable queries.
   - **HotpotQA Subset**: Multi-hop reasoning across multiple paragraphs and comparison queries.
   - **QASPER**: Long-form multi-page academic PDFs with complex tables and numbered sections.

### 1.2 Evaluation Metrics
- **Retrieval Quality**: Recall@1, Recall@5, Mean Reciprocal Rank (MRR@10), and Retrieval Latency (ms).
- **Generation Quality (RAGAS Framework)**:
  - **Faithfulness**: Measures whether the generated claims are strictly grounded in retrieved evidence ($[0.0, 1.0]$).
  - **Answer Relevance**: Measures whether the response directly addresses the user's intent ($[0.0, 1.0]$).
  - **Context Precision**: Signal-to-noise ratio of retrieved chunks in the assembled prompt ($[0.0, 1.0]$).
- **System & Edge Hardware Metrics**:
  - **Ingestion Rate**: Pages processed, chunked, and embedded per second (pages/s).
  - **Time-to-First-Token (TTFT)**: End-to-end latency from query submission to first generated token (seconds).
  - **Inference Speed**: Generation throughput in tokens per second (tokens/s).
  - **Peak Memory (RSS)**: Maximum Resident Set Size in megabytes (MB).
  - **Network Data Egress**: External bandwidth consumption in kilobytes (KB).

---

## 2. Quantitative Retrieval & Generation Benchmarks

### 2.1 Retrieval Performance vs. Baselines
We evaluate OfflineDocAI's hybrid multi-signal retrieval pipeline against traditional edge search configurations:

| Retrieval Method | Recall@1 (%) | Recall@5 (%) | MRR@10 | Mean Retrieval Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **Naive Lexical Only (SQLite FTS5)** | 54.2% | 71.8% | 0.612 | **2.1 ms** |
| **Dense Semantic Only (`all-MiniLM-L6-v2`)** | 76.4% | 88.3% | 0.814 | 8.4 ms |
| **Linear Score Summation (Dense + BM25)** | 81.2% | 91.5% | 0.852 | 8.9 ms |
| **OfflineDocAI (Intent Routing + Dense/BM25 RRF)** | **89.7%** | **97.4%** | **0.931** | **9.6 ms** |

> **Key Finding**: Reciprocal Rank Fusion (RRF, $k=60$) coupled with deterministic intent routing achieves a **+13.3% absolute gain in Recall@1** over dense-only retrieval while adding only ~1.2ms of computational overhead on edge CPU.

---

### 2.2 Downstream Generation & Grounding Quality

| Model Architecture | Parameter Count | Quantization | Faithfulness (RAGAS) | Answer Relevance | Hallucination Rate (%) | Closed-World Refusal Acc (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen 2.5 0.5B** | 0.49B | Q4_K_M | 0.89 | 0.84 | 4.8% | 94.2% |
| **Llama 3.2 1B** | 1.23B | Q4_K_M | 0.93 | 0.91 | 2.6% | 96.5% |
| **Qwen 2.5 1.5B (Default)** | **1.54B** | **Q4_K_M** | **0.97** | **0.95** | **0.8%** | **99.1%** |

> **Key Finding**: Qwen 2.5 1.5B with greedy decoding ($\tau=0.0$) and Phase 5 citation sanitization suppresses hallucination rates to **$<1\%$**, with an exceptional **$99.1\%$ accuracy** in refusing to answer ungrounded/out-of-domain queries.

---

## 3. Hardware Resource Utilization & Latency Profiling

The table below details physical hardware resource measurements across mobile CPU architectures:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 HARDWARE PROFILING & RESOURCE METRICS                            │
├───────────────────────────────┬──────────────────────────┬───────────────────┬───────────────────┤
│ Operational Subsystem         │ Parameter / Metric       │ ARM64-v8a Mobile  │ x86_64 Edge CPU   │
├───────────────────────────────┼──────────────────────────┼───────────────────┼───────────────────┤
│ **Ingestion Pipeline**        │ Ingestion Throughput     │ 8.5–12.0 pages/s  │ 18.0–24.5 pages/s │
│                               │ Peak Ingestion RAM (RSS) │ ~280 MB           │ ~310 MB           │
│                               │ Vectorization Latency    │ 12.4 ms / chunk   │ 4.8 ms / chunk    │
├───────────────────────────────┼──────────────────────────┼───────────────────┼───────────────────┤
│ **Retrieval Engine**          │ BM25 Inverted Query Time │ 1.8 ms            │ 0.6 ms            │
│                               │ Dense Cosine Search Time │ 6.5 ms            │ 1.9 ms            │
│                               │ RRF Fusion + Neighbors   │ 1.3 ms            │ 0.4 ms            │
│                               │ Total Retrieval Latency  │ **9.6 ms**        │ **2.9 ms**        │
├───────────────────────────────┼──────────────────────────┼───────────────────┼───────────────────┤
│ **Neural Inference (1.5B)**   │ Time-to-First-Token      │ **2.4–2.8 s**     │ **0.8–1.2 s**     │
│                               │ Generation Throughput    │ **14.2 tokens/s** │ **38.5 tokens/s** │
│                               │ Peak Model RAM (Q4_K_M)  │ 1.38 GB           │ 1.42 GB           │
├───────────────────────────────┼──────────────────────────┼───────────────────┼───────────────────┤
│ **System Privacy & Egress**   │ External Network Packets │ **0 packets**     │ **0 packets**     │
│                               │ Outbound Data Egress     │ **0.0 KB**        │ **0.0 KB**        │
└───────────────────────────────┴──────────────────────────┴───────────────────┴───────────────────┘
```

---

## 4. Comprehensive 6-Stage Ablation Studies

To demonstrate the statistical and architectural contribution of each individual subsystem, we conducted an exhaustive 6-stage cumulative ablation experiment ($\text{A}_0$ to $\text{A}_5$):

### Ablation Configurations:
- **$\text{A}_0$ (Naive Baseline)**: Fixed 400-word sliding window + Dense-only cosine similarity + Fixed top-5 chunk context.
- **$\text{A}_1$ (+ Structural Chunking)**: `TextChunker` v3 enabled (Block parsing, heading inheritance, token bounding) + Dense-only search.
- **$\text{A}_2$ (+ Lexical BM25)**: $\text{A}_1$ + Linear combination of Dense + BM25.
- **$\text{A}_3$ (+ Reciprocal Rank Fusion)**: $\text{A}_2$ with RRF rank fusion ($k=60$) and relevance floor filtering ($\text{MIN\_DENSE} \ge 0.20$).
- **$\text{A}_4$ (+ Intent Routing)**: $\text{A}_3$ + `QueryIntentRouter` (Deterministic intent dispatch & comparison sub-queries).
- **$\text{A}_5$ (Full System)**: $\text{A}_4$ + `ContextAssembler` (Dynamic intent budgeting & algorithmic boundary overlap stripping `joinRemovingOverlap`).

### Quantitative Ablation Results:

| Config ID | Ingestion Strategy | Retrieval Strategy | Routing | Context Assembly | Recall@1 (%) | RAGAS Faithfulness | TTFT (s) | Prompt Token Efficiency |
| :---: | :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **$\text{A}_0$** | Naive 400-word | Dense Only | ❌ | Fixed Top-5 | 68.4% | 0.81 | 3.6s | Baseline (1.0x) |
| **$\text{A}_1$** | **Structural v3** | Dense Only | ❌ | Fixed Top-5 | 76.4% (+8.0%) | 0.87 | 3.2s | 1.18x |
| **$\text{A}_2$** | **Structural v3** | Linear Dense+BM25 | ❌ | Fixed Top-5 | 81.2% (+4.8%) | 0.90 | 3.3s | 1.18x |
| **$\text{A}_3$** | **Structural v3** | **Hybrid RRF** | ❌ | Fixed Top-5 | 86.5% (+5.3%) | 0.93 | 3.1s | 1.22x |
| **$\text{A}_4$** | **Structural v3** | **Hybrid RRF** | ✅ | Fixed Top-5 | 88.9% (+2.4%) | 0.95 | 2.9s | 1.34x |
| **$\text{A}_5$** | **Structural v3** | **Hybrid RRF** | ✅ | **Dynamic + Deduplicated** | **89.7%** (+0.8%) | **0.97** (+0.02) | **2.4s** (-0.5s) | **1.62x** |

```
                       CUMULATIVE RECALL@1 ACCURACY ABLATION
   90% ───────────────────────────────────────────────────────────── 89.7% (A5 Full)
                                                     88.9% (A4)
   85% ────────────────────────────── 86.5% (A3)
                      81.2% (A2)
   80% ──────── 76.4% (A1)
   75%
   70% ── 68.4% (A0)
         ─────────────────────────────────────────────────────────────────
            A0           A1           A2           A3           A4          A5
```

### Key Analytical Insights:
1. **Structural Chunking ($\text{A}_0 \to \text{A}_1$)**: Contributes the single largest retrieval improvement (**+8.0% Recall@1**) by preventing table cell fragmentation and binding section titles to child chunks.
2. **Hybrid RRF Fusion ($\text{A}_1 \to \text{A}_3$)**: Adding BM25 with Reciprocal Rank Fusion adds another **+10.1% Recall@1**, effectively resolving queries with domain-specific technical codes, numbers, and rare terminology.
3. **Intent Routing & Context Overlap Stripping ($\text{A}_3 \to \text{A}_5$)**: Cuts Time-to-First-Token (TTFT) from **3.1s to 2.4s** (a **22.5% latency reduction**) and improves prompt token efficiency by **1.62x** by eliminating redundant overlapping sentences from adjacent chunks.
