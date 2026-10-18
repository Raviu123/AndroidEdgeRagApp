# OfflineDocAI — Research Paper Drafting Kit & Materials

Welcome to the **Research Paper Preparation Kit** for the academic paper:  
**Title**: `OfflineDocAI: An Efficient End-to-End On-Device Retrieval-Augmented Generation System for Edge Hardware`  
**Target Venue**: SACAIM (*International Conference on Advanced IT, Engineering, and Management*) / IEEE / ACM Edge Computing Conferences.

This directory is designed as a self-contained, modular knowledge repository for human authors and autonomous writing agents to draft, format, and review the full paper manuscript.

---

## Directory Structure & Resource Map

| File | Purpose & Contents | Primary Paper Sections |
| :--- | :--- | :--- |
| [`01_ABSTRACT_AND_INTRODUCTION.md`](01_ABSTRACT_AND_INTRODUCTION.md) | Problem statement, privacy & edge computing motivation, research questions, core contributions, and paper roadmap. | **Section 1 (Introduction)**, **Abstract** |
| [`02_RELATED_WORK_AND_POSITIONING.md`](02_RELATED_WORK_AND_POSITIONING.md) | Exhaustive review of edge RAG literature (*EdgeRAG, MobileRAG, Pocket RAG, RAGdb, SRAS, HeRo, SmartRAG*), what was refined, and explicit positioning. | **Section 2 (Related Work & Background)** |
| [`03_SYSTEM_ARCHITECTURE_AND_PIPELINE.md`](03_SYSTEM_ARCHITECTURE_AND_PIPELINE.md) | Comprehensive 5-stage pipeline design, mathematical formulations, formal algorithms, pseudocode, and dataflow descriptions. | **Section 3 (System Architecture & Methodology)** |
| [`04_SPECIFICATIONS_AND_HYPERPARAMETERS.md`](04_SPECIFICATIONS_AND_HYPERPARAMETERS.md) | Hardware-agnostic system parameters, model quantizations (Q4_K_M), prompt engineering templates, security defense rules, and token budgets. | **Section 4 (System Configuration & Hyperparameters)** |
| [`05_EXPERIMENTS_RESULTS_AND_ABLATIONS.md`](05_EXPERIMENTS_RESULTS_AND_ABLATIONS.md) | Quantitative experimental tables, multi-metric evaluations (Recall@k, MRR, RAGAS, TTFT, TPS, RAM), and 6-stage ablation studies ($A_0$ to $A_5$). | **Section 5 (Empirical Evaluation & Ablation Studies)** |
| [`06_DISCUSSION_LIMITATIONS_AND_CONCLUSION.md`](06_DISCUSSION_LIMITATIONS_AND_CONCLUSION.md) | Edge trade-offs, theoretical implications, failure analysis, future directions, and concluding summary. | **Section 6 (Discussion & Limitations)**, **Section 7 (Conclusion)** |
| [`07_BIBTEX_REFERENCES.bib`](07_BIBTEX_REFERENCES.bib) | Standardized, camera-ready BibTeX citations for all related papers, foundational SLM papers, and indexing literature. | **References / Bibliography** |
| [`08_POST_DRAFT_V4_RAG_CHANGES_AND_EXPERIMENTS.md`](08_POST_DRAFT_V4_RAG_CHANGES_AND_EXPERIMENTS.md) | Exhaustive post-v4 RAG pipeline upgrades (ONNX Cross-Encoder, model tiers, temporal routing), MASVS & LiHua benchmarks, and 1.5B model placeholders. | **Section 3, 4 & 5 (Post-Draft v4 Updates)** |

---

## Visual Figures Available in Root

The paper utilizes 5 publication-quality high-resolution figures stored in [`figures/`](../figures/):

1. **Figure 1**: `fig1_system_architecture.png` — End-to-End On-Device RAG Processing Pipeline (5 Stages).
2. **Figure 2**: `fig2_chunking_comparison.png` — Hierarchical Structure-Preserving Chunking vs. Naive Sliding Window.
3. **Figure 3**: `fig3_retrieval_flow.png` — Query Intent Routing & Multi-Signal Reciprocal Rank Fusion.
4. **Figure 4**: `fig4_validation_ladder.png` — Dynamic Token Budgeting & Multi-Tier Grounding Validation Ladder.
5. **Figure 5**: `fig5_performance_benchmarks.png` — System Performance, Ingestion Throughput, TTFT, and RAM Footprint.

---

## Guide for Autonomous Drafting Agents

When generating the manuscript:
1. Preserve all mathematical notations ($L_{\text{target}}$, $\text{RRF}(d)$, $\text{Sim}_{\text{dense}}$).
2. Cite all 8 baseline edge RAG papers using standard academic attribution.
3. Keep the language objective, precise, and academically formal.
4. Emphasize the **100% Air-Gapped / Zero-Egress** edge privacy guarantees alongside the **low-latency algorithmic optimizations**.
