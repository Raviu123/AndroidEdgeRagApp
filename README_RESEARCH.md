# Offline Doc AI App - Research & Versioning History

This directory tracks the architectural iterations, experimental setups, RAG pipeline changes, and performance benchmarking for research publications.

## Index of Versioned States

- **[State 001 - Initial Offline RAG Architecture & Baseline Mechanics](docs/state_001.md)**: Details the baseline document parsing, sliding-window chunking, vector storage with Room DB, cosine similarity calculation, and fallback retrieval strategies.
- **[State 002 - Enhanced Retrieval & SLM Lifecycle Management](docs/state_002.md)**: Implements dynamic query routing, uniform stratified sampling for generic queries (*"What is this doc about?"*), TF-IDF lexical search, clean summary synthesis, and on-device model management (switching, downloading, and deleting SLMs).
- **[Working Model Version 1.0 - Full Native On-Device RAG Architecture](docs/working_model_v1.md)**: Full production-grade offline pipeline featuring native `llama.cpp` JNI execution on Android ARM64/x86_64, Microsoft ONNX Runtime Android (`all-MiniLM-L6-v2`), structure-preserving hierarchical chunking, deterministic query intent routing, hybrid vector/lexical retrieval, dynamic token budgeting, production SLM deployment (`Qwen 2.5 1.5B Instruct`), and Phase 6 citation grounding.

