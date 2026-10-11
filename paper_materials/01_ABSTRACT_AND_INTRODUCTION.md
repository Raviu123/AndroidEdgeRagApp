# Section 1: Abstract & Introduction

## Paper Metadata
- **Title**: OfflineDocAI: An Efficient End-to-End On-Device Retrieval-Augmented Generation System for Edge Hardware
- **Keywords**: On-Device AI, Edge Computing, Retrieval-Augmented Generation (RAG), Small Language Models (SLMs), Privacy-Preserving AI, Air-Gapped Systems, Reciprocal Rank Fusion, Structural Chunking.

---

## Abstract

Retrieval-Augmented Generation (RAG) has emerged as the de facto standard for grounding Large Language Models (LLMs) in domain-specific knowledge and mitigating factual hallucinations. However, prevailing enterprise RAG architectures rely heavily on centralized cloud infrastructure, distributed vector databases, and multi-billion-parameter cloud APIs. This paradigm incurs substantial recurring cloud hosting costs, demands constant high-bandwidth network connectivity, and introduces severe data sovereignty and privacy risks when processing sensitive, confidential, or proprietary enterprise documents.

In this paper, we present **OfflineDocAI**, a fully decentralized, 100% air-gapped, zero-data-egress on-device RAG system engineered from the ground up for resource-constrained edge hardware. OfflineDocAI establishes an efficient end-to-end processing pipeline operating entirely within local edge device boundaries:
1. An on-device **Hierarchical Structure-Preserving Chunking algorithm** that parses raw unstructured multi-format documents (PDF, DOCX, TXT) into structural units (headings, table rows, lists, paragraphs), joins cross-page sentence continuations, and enforces embedding-token boundaries.
2. An embedded **Zero-Copy Vector Storage and Inverted Index** executing 384-dimensional dense semantic vectorization via ONNX Runtime and storing binary byte buffers with sub-millisecond retrieval overhead in SQLite.
3. A deterministic **Query Intent Classification and Multi-Signal Hybrid Retrieval engine** that routes user queries across 6 intent classes and fuses dense semantic cosine similarities with Okapi BM25 lexical scores using Reciprocal Rank Fusion (RRF, $k=60$).
4. A **Dynamic Intent-Budgeted Context Assembler** that greedily packs evidence into constrained token budgets while algorithmically eliminating redundant boundary sentences across contiguous chunks.
5. A native **C++17 SIMD-accelerated neural inference engine** (`llama.cpp`) executing quantized Small Language Models (0.5B to 1.5B parameters) with in-memory context caching, prompt injection sanitization, and verifiable page-level citation validation.

Extensive empirical evaluations across multiple document genres demonstrate that OfflineDocAI achieves an ingestion throughput of **8–15 pages per second**, delivers sub-second to 2.8s Time-to-First-Token (TTFT) across mobile CPU architectures, maintains a peak operational RAM footprint below **1.5 GB**, and operates with strictly **0.0 KB external network data egress**. The system demonstrates that production-grade, highly grounded document intelligence can be achieved on commodity edge hardware without sacrificing user privacy or requiring cloud connectivity.

---

## 1. Introduction & Motivation

Large Language Models (LLMs) have demonstrated extraordinary capabilities in natural language reasoning, synthesis, and conversational interaction. Despite these advancements, standard LLMs suffer from two fundamental operational bottlenecks:
1. **Parametric Knowledge Cutoffs**: LLMs cannot access private, newly created, or proprietary internal documents not present in their pretraining corpora.
2. **Hallucination Risk**: When queried on fine-grained factual entities, domain-specific numerical parameters, or obscure technical specifications, generative models frequently fabricate plausible-sounding but factually false responses.

Retrieval-Augmented Generation (RAG) effectively resolves these challenges by separating parametric reasoning from non-parametric knowledge storage. A retriever searches an external corpus for passages relevant to a user query, and an LLM synthesizes a grounded response conditioned strictly on the retrieved evidence.

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 The Enterprise RAG Trilemma             │
                  └────────────────────────────┬────────────────────────────┘
                                               │
               ┌───────────────────────────────┼───────────────────────────────┐
               ▼                               ▼                               ▼
    [1. Data Privacy Risk]         [2. Connectivity Latency]       [3. Infrastructure Cost]
  Sending confidential PDFs       High round-trip latency &       Continuous GPU cloud servers,
   to cloud APIs violates          total failure in remote or       vector DB subscriptions, &
  GDPR, HIPAA, and IP secrecy     air-gapped field environments     API token costs scale linearly
```

### The Cloud RAG Dilemma
While cloud-hosted RAG solutions are widespread, deploying RAG through cloud APIs is unacceptable in mission-critical, regulated, or privacy-sensitive domains:
- **Data Sovereignty & Legal Compliance**: In healthcare (HIPAA), legal discovery (attorney-client privilege), financial auditing, and defense sectors, uploading unredacted documents to third-party cloud servers is legally impermissible.
- **Air-Gapped & Degraded Environments**: First responders in disaster zones, maritime crews, field engineers in remote mining facilities, and personnel in high-security SCADA facilities operate with zero or intermittent internet access.
- **Infrastructure Bloat & Recurring Cost**: Standard server RAG stacks entail complex distributed infrastructure—Dockerized microservices, high-memory vector databases (e.g., Milvus, Pinecone), GPU inference clusters, and per-token API billing.

### Challenges of Edge RAG Deployment
Porting the complete RAG lifecycle to edge hardware (e.g., commodity ARM-based mobile devices, embedded Linux boards, edge gateways) introduces severe systems-level and algorithmic constraints:
- **Strict RAM Envelopes**: Mobile operating systems enforce hard memory budgets per application (often $\le 2\text{ GB}$). A multi-gigabyte vector index or an unquantized language model triggers Out-Of-Memory (OOM) process termination.
- **Compute & Thermal Throttling**: Edge CPUs lack high-wattage active cooling and multi-GPU VRAM. Inefficient token evaluation causes severe battery drain and thermal throttling.
- **Information Density Loss in Small Windows**: Small Language Models (SLMs) (0.5B–1.5B parameters) operate with constrained effective context lengths ($2048$–$4096$ tokens). Naive context concatenation introduces severe token clutter and distracts attention mechanisms.
- **Indirect Prompt Injection**: Processing arbitrary untrusted user documents on edge devices exposes the local model to adversarial formatting sequences, jailbreaks, and delimiter hijacking embedded within text.

---

## 2. Research Questions (RQs) Addressed

This research systematically formulates and answers the following core research questions:

- **RQ1 (Ingestion & Semantic Integrity)**: *How can multi-format unstructured enterprise documents be decomposed, chunked, and embedded entirely on-device without losing structural hierarchy (headings, tables, lists) and without exceeding embedding token ceilings?*
- **RQ2 (Retrieval Accuracy & Latency)**: *Can a hybrid lexical (BM25) and dense semantic vector retrieval pipeline using dynamic Reciprocal Rank Fusion (RRF) match or exceed server-grade retrieval precision under tight sub-millisecond edge CPU constraints?*
- **RQ3 (Context Optimization)**: *How can dynamic intent-driven token budgeting and algorithmic boundary overlap deduplication maximize the information density of small context windows for Small Language Models (SLMs)?*
- **RQ4 (Grounding & Execution Safety)**: *How can on-device neural execution enforce zero hallucinations, verifiable page-level citation tagging, and adversarial prompt defense in a strictly air-gapped environment?*

---

## 3. Summary of Scientific & Engineering Contributions

OfflineDocAI makes the following primary contributions to edge computing and applied natural language processing:

1. **End-to-End Vertically Integrated Edge Architecture**: We design, implement, and validate a 100% air-gapped, zero-data-egress on-device RAG system that executes the complete pipeline—document ingestion, structural chunking, ONNX embedding, SQLite vector indexing, intent-based hybrid retrieval, dynamic context assembly, native C++ JNI neural inference, and citation verification—entirely on commodity edge hardware.
2. **Hierarchical Structure-Preserving Chunking Algorithm**: We formulate a token-bounded document decomposition algorithm that parses structural elements (headings, table rows, lists, paragraphs), joins cross-page paragraph continuations, associates parent section hierarchies to child units, and aligns chunk lengths with WordPiece tokenizers to guarantee zero truncation during embedding.
3. **Intent-Routed Multi-Signal Hybrid Retrieval with RRF**: We introduce a deterministic 6-class intent router coupled with dual dense semantic search and Okapi BM25 lexical ranking on dynamic inverted indexes, fused via Reciprocal Rank Fusion ($k=60$) with relevance floor filtering ($\text{MIN\_DENSE} \ge 0.20$) and spatial neighbor context expansion.
4. **Dynamic Context Assembler with Algorithmic Overlap Stripping**: We formulate a greedy priority-based token packing mechanism that enforces intent-specific budgets (384/640/1152 tokens), deduplicates candidate chunks, and algorithmically identifies and removes overlapping sentence boundaries across contiguous chunks (`joinRemovingOverlap`).
5. **Adversarial Prompt Defense & Verifiable Citation Grounding**: We implement strict control-marker sanitization against indirect prompt injection, reasoning token budget management for distilled reasoning models (DeepSeek-R1 style), and a deterministic multi-tier citation verification engine ensuring verifiable page-level source attribution.
6. **Empirical Validation on Physical Hardware**: We provide extensive benchmarks measuring ingestion throughput (8–15 pages/s), retrieval accuracy (Recall@k, MRR), generation latency (TTFT $<2.8\text{s}$, 12–22 tokens/s), and memory footprint ($<1.5\text{ GB}$ peak RSS), accompanied by an exhaustive 6-stage ablation study.
