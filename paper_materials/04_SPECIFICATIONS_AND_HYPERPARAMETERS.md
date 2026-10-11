# Section 4: System Specifications, Configurations & Hyperparameters

This document compiles the formal system parameters, hyperparameter tables, model configurations, prompt templates, and security rules of the OfflineDocAI architecture.

---

## 1. System Hyperparameters & Operational Specifications

The following table provides the exact numerical parameters used throughout the OfflineDocAI pipeline.

| Subsystem | Parameter Name | Symbol / Variable | Value | Rationale / Academic Basis |
| :--- | :--- | :---: | :---: | :--- |
| **Ingestion** | Target Chunk Length | $L_{\text{target}}$ | **128 tokens** | Aligned with WordPiece token distribution; fits fine-grained factual entities. |
| **Ingestion** | Hard Chunk Ceiling | $L_{\text{max}}$ | **180 tokens** | Guarantees chunk stays well within embedding model maximum sequence limit ($256$). |
| **Ingestion** | Chunk Overlap | $O_{\text{tokens}}$ | **32 tokens** | Preserves semantic continuity across chunk boundaries (~$20\%$ overlap ratio). |
| **Ingestion** | Heading Token Reserve | $H_{\text{res}}$ | **24 tokens** | Guarantees headroom for section title injection without exceeding $L_{\text{max}}$. |
| **Embedding** | Vector Dimension | $D_{\text{embed}}$ | **384 float32** | Native dimensionality of `all-MiniLM-L6-v2` transformer backbone. |
| **Embedding** | Max Sequence Length | $N_{\text{seq}}$ | **256 tokens** | Standard BERT WordPiece input buffer limit. |
| **Embedding** | Vector BLOB Size | $S_{\text{blob}}$ | **1536 bytes** | Direct byte serialization ($384 \times 4$ bytes/float) for zero-copy SQLite storage. |
| **Retrieval** | Dense Candidate Pool | $K_{\text{dense}}$ | **50 chunks** | Top candidate depth for dense semantic search. |
| **Retrieval** | BM25 Candidate Pool | $K_{\text{bm25}}$ | **50 chunks** | Top candidate depth for lexical Okapi BM25 search. |
| **Retrieval** | Dense Relevance Floor | $\text{MIN\_DENSE}$ | **0.20** | Empirically determined cosine similarity noise floor for MiniLM English embeddings. |
| **Retrieval** | BM25 Term Weight ($k_1$) | $k_1$ | **1.2** | Controls term frequency saturation in BM25 scoring. |
| **Retrieval** | BM25 Document Normalization ($b$) | $b$ | **0.75** | Standard document length penalty ratio. |
| **Retrieval** | RRF Smoothing Factor | $k_{\text{RRF}}$ | **60.0** | Cormack et al. reciprocal rank fusion constant to balance dense and lexical ranks. |
| **Retrieval** | Spatial Neighbor Seeds | $N_{\text{seed}}$ | **3 chunks** | Number of top primary hits expanded with adjacent ($\pm 1$) chunks. |
| **Retrieval** | Summary Stratified Samples | $N_{\text{sample}}$ | **24 chunks** | Evenly spaced samples across the document for global summary queries. |
| **Context** | Exact Fact Token Budget | $B_{\text{fact}}$ | **384 tokens** | Sub-second prompt evaluation for simple factual queries. |
| **Context** | Comparison Token Budget | $B_{\text{comp}}$ | **640 tokens** | Balanced budget to accommodate dual entity evidence blocks. |
| **Context** | Summary / List Token Budget| $B_{\text{enum}}$ | **1152 tokens** | Broad context budget for multi-page aggregations. |
| **Inference** | Context Window Size | $N_{\text{ctx}}$ | **2048 / 4096** | Sized dynamically based on system RAM ($<3.4\text{GB} \to 2048$, $\ge 3.4\text{GB} \to 4096$). |
| **Inference** | Max New Tokens (Default) | $T_{\text{gen}}$ | **448 tokens** | Production budget for synthesized responses. |
| **Inference** | Greedy Sampling Temperature | $\tau_{\text{rag}}$ | **0.0** | Greedy deterministic decoding for zero hallucinations in RAG mode. |
| **Inference** | Top-P Nucleus Sampling | $p_{\text{chat}}$ | **0.90** | Applied during open chat interactions. |
| **Inference** | Repetition Penalty | $r_{\text{pen}}$ | **1.10** | Suppresses repetitive n-gram loops during generation. |
| **Inference** | CPU Worker Threads | $N_{\text{threads}}$ | **4 threads** | Matches typical big/performance core cluster on mobile SoCs. |

---

## 2. On-Device Model Fleet Specifications

OfflineDocAI supports dynamic on-device switching across four quantized Small Language Model architectures in GGUF format:

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   ON-DEVICE SLM MODEL FLEET                                       │
├──────────────────────┬─────────────┬──────────────┬──────────────┬──────────────┬─────────────────┤
│ Model Name           │ Parameters  │ Architecture │ Quantization │ File Size    │ Primary Role    │
├──────────────────────┼─────────────┼──────────────┼──────────────┼──────────────┼─────────────────┤
│ Qwen 2.5 1.5B        │ 1.54B       │ Qwen2        │ Q4_K_M       │ 986 MB       │ Production RAG  │
│ Qwen 2.5 0.5B        │ 0.49B       │ Qwen2        │ Q4_K_M       │ 491 MB       │ Ultra-Low RAM   │
│ Llama 3.2 1B         │ 1.23B       │ Llama        │ Q4_K_M       │ 807 MB       │ Balanced Alt    │
│ DeepSeek-R1-1.5B-Qwen│ 1.54B       │ Qwen2        │ Q4_K_M       │ 992 MB       │ Complex Math/QA │
└──────────────────────┴─────────────┴──────────────┴──────────────┴──────────────┴─────────────────┘
```

---

## 3. Dynamic Prompt Formatting Templates

The inference engine dynamically detects the model's native chat template or applies standard fallbacks based on architecture family:

### 3.1 ChatML Format (Qwen 2.5)
```text
<|im_start|>system
You answer questions using only the numbered sources provided by the user.
Rules:
- Use only information stated in the sources. Do not use outside knowledge and do not guess.
- After each statement, cite the source it came from by copying its tag exactly, for example [Source 2, Page 7].
- If the sources do not contain the answer, reply exactly: "I couldn't find this in the selected document(s)."
- Copy numbers, names, dates, units and quotations exactly as they appear in the sources.
<|im_end|>
<|im_start|>user
Sources:

[Source 1, Page 4] (Document: Technical_Spec.pdf | Section: Thermal Envelopes)
The maximum operating temperature for the sensor module is 85°C (185°F). Operating above this threshold triggers automatic shutdown within 15 seconds.

Question: What is the maximum operating temperature?
<|im_end|>
<|im_start|>assistant
```

### 3.2 Llama 3 Format (Llama 3.2 1B)
```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You answer questions using only the numbered sources provided by the user...<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Sources:
[Source 1, Page 4] ...

Question: What is the maximum operating temperature?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

---

## 4. Adversarial Prompt Defense & Sanitization Rules

To prevent indirect prompt injection when parsing unvetted user documents, `PromptBuilder.sanitizeUntrusted()` enforces strict regex neutralization:

$$\text{Sanitize}(T): \text{ControlMarker} \to \text{EscapedUnicodeMarker}$$

1. **Delimiter Escaping**:
   - `<|` is transformed to `‹|`
   - `|>` is transformed to `|›`
   - `<｜` is transformed to `‹｜`
   - `｜>` is transformed to `｜›`
2. **Control Token Neutralization**:
   - Matches `<(?:\/?)(?:start_of_turn|end_of_turn|s|bos|eos|think|tool_call|tool_response)>` and wraps in safe angle delimiters `‹...›`.
   - Matches `\[(?:\/?)(?:INST|SYS)\]` and wraps in safe brackets `⟦...⟧`.
3. **Citation Sanitization**:
   - `sanitizeCitations()` uses regex `\[\s*Sources?\s*:?\s*([^\[\]]{1,80})\]` to match all generated citations and replaces invalid indices with verified canonical tags.
