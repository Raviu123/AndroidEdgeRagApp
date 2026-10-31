# Multi-Device Experimental Protocol: On-Device RAG Benchmarking

**Project:** OfflineDocAI (`unified-rag-v2`)  
**Objective:** Empirically evaluate on-device RAG performance, latency, memory consumption, and citation grounding across heterogeneous Android hardware tiers (4GB, 6GB, 8GB RAM) for research paper publication.  
**Target Output:** Publication-ready empirical tables and comparative figures for ACM MobiSys / IEEE TMC / EMNLP submissions.

---

## 1. Experimental Matrix

### A. Device Hardware Tiers
| Tier | Target Specs | Typical SoCs | Role in Experiment | Context Window (`n_ctx`)* |
|---|---|---|---|---|
| **Tier 1 (Budget)** | **4 GB RAM** | MediaTek Helio G-series / Snapdragon 680 | Stress test: OOM vulnerability, low-memory stability, CPU throttling | 4,096 if the phone reports ≥ 3.4 GB (most "4 GB" phones), else 2,048 |
| **Tier 2 (Mid-Range)** | **6 GB RAM** | Snapdragon 7s Gen 2 / Dimensity 7050 | Sweet spot: Daily commodity device performance | 4,096 tokens |
| **Tier 3 (Flagship)** | **8 GB+ RAM** | Snapdragon 8 Gen 2/3 / Dimensity 9200+ | Peak capability: Optimal TTFT and token generation speed | 4,096 tokens |

### B. Evaluated Small Language Models (SLMs)
1. **Qwen 2.5 0.5B Instruct** (Q4_K_M GGUF, ~398 MB) — Ultra-lightweight edge baseline.
2. **Qwen 2.5 1.5B Instruct** (Q4_K_M GGUF, ~986 MB) — Primary balanced production model.
3. **Llama 3.2 1B Instruct** (Q4_K_M GGUF, ~750 MB) — Alternative architecture comparison.

### C. Embedding & Ingestion Engine
* **ONNX Runtime Mobile:** `all-MiniLM-L6-v2` (384-dimensional normalized dense vectors).
* **Chunking Policy:** WordPiece tokenizer-aligned (128 target tokens / 180 max tokens, ≤ 32-token sentence overlap).

\* `n_ctx` is chosen from total RAM (`GGUFLLM.chooseContextSize`). The **evidence budget** inside it is device-adaptive: `min(context fit, intent cap, measured prompt speed × Speed-Mode seconds)`, floored at 600 tokens. Record it per run (`Evidence Budget` row / `budget_tokens`).

### D. Device-Adaptive Inference (record for every device)
* **Thread counts** are calibrated once per phone + model + CPU kernels (`PerformanceTuner`): generation threads and prompt-processing threads are measured separately from candidates derived from the CPU layout (`CpuTopology`, e.g. `2+6`, `1+3+4`). Log `threads_decode` / `threads_batch`, `cpu_topology`, `cpu_variant`.
* **Prompt prefix reuse:** the KV cache of the shared prompt start is reused between questions (`reused_prompt_tokens`). For cold-start measurements, restart the app before each run.
* **Speed Mode** (Settings → Answer speed) sets the prompt-processing target: Fastest 20 s, Balanced 35 s, Thorough 75 s. Keep it fixed (Balanced) across devices.
* **Thermal / battery saver** reduce threads and budget automatically (`thermal` field); note the phone's temperature state and keep battery saver off.
* Use a **release build**; debug builds are not representative.
* **Retrieval Policy:** Dual-channel Dense Top-50 + BM25 Top-50 $\rightarrow$ Reciprocal Rank Fusion (RRF, $k=60$).

---

## 2. Standardized Benchmark Query Set

To ensure statistical rigor across all 3 devices and 3 models, evaluate the following standard batch of questions across different intent categories:

```json
[
  {
    "id": "Q1",
    "intent": "FACTUAL",
    "query": "What is the optimal temperature and humidity for healing grafted plants?",
    "gold_page": 4,
    "ground_truth_fact": "Optimal healing requires 21-24°C (70-75°F) with 85-95% relative humidity."
  },
  {
    "id": "Q2",
    "intent": "PROCEDURAL",
    "query": "What are the step-by-step instructions for performing a cleft graft?",
    "gold_page": 7,
    "ground_truth_fact": "Cut rootstock horizontally, split center 2-3cm, shape scion into wedge, insert matching cambium layers, seal with tape/wax."
  },
  {
    "id": "Q3",
    "intent": "COMPARISON",
    "query": "Compare whip-and-tongue grafting with cleft grafting regarding branch thickness requirements.",
    "gold_page": "6-8",
    "ground_truth_fact": "Whip-and-tongue requires matching diameters (0.5-1.5cm), whereas cleft grafting allows scions to be thinner than rootstock."
  },
  {
    "id": "Q4",
    "intent": "ENUMERATION",
    "query": "List all rootstock compatibility issues mentioned in the document.",
    "gold_page": "11-12",
    "ground_truth_fact": "Delayed graft incompatibility, phloem breakdown, viral hypersensitivity, overgrowth at union."
  },
  {
    "id": "Q5",
    "intent": "SUMMARY",
    "query": "Give a complete summary of the grafting guide.",
    "gold_page": "Multi-page",
    "ground_truth_fact": "Overview of plant propagation through scion-rootstock union, tool sterilization, healing chamber management, and aftercare."
  },
  {
    "id": "Q6",
    "intent": "OUT_OF_DOMAIN",
    "query": "What are the maintenance steps for a lithium-ion battery?",
    "gold_page": "None",
    "ground_truth_fact": "The document does not contain information about lithium-ion batteries."
  }
]
```

---

## 3. Telemetry Metrics to Measure

For every test run (Device $\times$ Model $\times$ Query), record the following:

### A. Latency & Throughput Metrics
* **Document Ingestion Time ($T_{\text{ingest}}$):** Total seconds to extract, chunk, and embed document vectors into Room DB.
* **Retrieval Time ($T_{\text{ret}}$):** Milliseconds to compute Dense cosine + BM25 scores + RRF ranking.
* **Time-to-First-Token ($\text{TTFT}$):** Milliseconds from the start of the generation call to the first sampled token (includes prefill). Also report **prefill speed** (`prefill_tokens_per_second` = new prompt tokens / prefill time).
* **Decode Speed ($S_{\text{gen}}$):** Output generation speed measured in `tokens/second`.
* **Total Response Latency ($T_{\text{total}}$):** Wall-clock time from query submission to generation completion.

### B. Hardware Resource Metrics
* **Peak PSS Memory ($\text{RAM}_{\text{peak}}$):** Total memory in MB logged via `dumpsys meminfo`.
  * *Native C++ Memory:* Memory used by `llama.cpp` weights and KV cache.
  * *JVM Heap:* Memory used by Android UI, Room DB, and bitmaps.
* **CPU Core Load (%):** Thread utilization during active prefill/decode phases.
* **Thermal / Throttling Status:** Logged from Android battery/thermal services.

### C. Grounding & Answer Quality Metrics
* **Citation Accuracy (%):** Percentage of generated citations (`[Source N, Page P]`) matching gold reference pages.
* **Factual Correctness:** 0–5 scale evaluated against the ground-truth answer.
* **Hallucination Rate (%):** Percentage of ungrounded or fabricated statements (especially for Q6 Out-of-Domain).

---

## 4. Step-by-Step USB Execution Workflow

When you are ready to test, follow these steps with each phone:

### Step 1: Connect Phone & Verify ADB
```powershell
# 1. Check device connection
adb devices

# 2. Query device specs (CPU architecture, RAM, Android OS version)
adb shell getprop ro.product.model
adb shell getprop ro.product.cpu.abi
adb shell "cat /proc/meminfo | grep MemTotal"
```

### Step 2: Build & Install Fresh APK
```powershell
# Build and install debug APK onto the connected device
./gradlew installDebug
```

### Step 3: Push Test Documents and Models
```powershell
# Push sample PDF to phone documents
adb push Grafting_15_Page_Guide.pdf /sdcard/Documents/

# (Optional) Push pre-downloaded GGUF models directly to avoid in-app download time
adb push qwen2.5-1.5b-instruct-q4_k_m.gguf /sdcard/Android/data/com.offlinedocai/files/models/llm/
```

### Step 4: Run Batch Queue & Stream Telemetry
```powershell
# 1. Clear previous logcat buffers
adb logcat -c

# 2. Launch App
adb shell am start -n com.offlinedocai/.MainActivity

# 3. Stream real-time logs for LLM generation & retrieval metrics
adb logcat -s "VectorStore" "LlamaRunner" "ContextAssembler" "ChatRepository"

# 4. Monitor Peak Memory in background loop during generation
adb shell dumpsys meminfo com.offlinedocai
```

### Step 5: Pull Exported Benchmark Session
```powershell
# Pull the exported JSON benchmark log from the app
adb pull /sdcard/Android/data/com.offlinedocai/files/exports/ ./benchmark_results/
```

---

## 5. Paper Results Table Templates

The collected data will directly populate these publication tables:

### Table 1: Hardware Scaling & Latency Comparison
| Device Tier | RAM | Ingestion Time (s) | Avg TTFT (ms) | Avg Decode (tok/s) | Peak PSS RAM (MB) |
|---|---|---|---|---|---|
| **Tier 1 (Budget)** | 4 GB | *[Measured]* | *[Measured]* | *[Measured]* | *[Measured]* |
| **Tier 2 (Mid-Range)** | 6 GB | *[Measured]* | *[Measured]* | *[Measured]* | *[Measured]* |
| **Tier 3 (Flagship)** | 8 GB+ | *[Measured]* | *[Measured]* | *[Measured]* | *[Measured]* |

### Table 2: Model Quality vs Size on Edge Hardware
| Model | Size (MB) | Avg TTFT (ms) | Speed (tok/s) | Citation Accuracy (%) | Hallucination Rate (%) |
|---|---|---|---|---|---|
| **Qwen 2.5 0.5B** | ~398 MB | *[Measured]* | *[Measured]* | *[Measured]* | *[Measured]* |
| **Llama 3.2 1B** | ~750 MB | *[Measured]* | *[Measured]* | *[Measured]* | *[Measured]* |
| **Qwen 2.5 1.5B** | ~986 MB | *[Measured]* | *[Measured]* | *[Measured]* | *[Measured]* |

### Table 3: Intent-Specific Retrieval & Grounding Performance
| Query Intent | Retrieval Time (ms) | Chunks Packed | Citation Precision (%) | Gold Fact Retained? |
|---|---|---|---|---|
| **Factual** | *[Measured]* | *[Measured]* | *[Measured]* | Yes/No |
| **Procedural** | *[Measured]* | *[Measured]* | *[Measured]* | Yes/No |
| **Comparison** | *[Measured]* | *[Measured]* | *[Measured]* | Yes/No |
| **Enumeration** | *[Measured]* | *[Measured]* | *[Measured]* | Yes/No |
| **Summary** | *[Measured]* | *[Measured]* | *[Measured]* | Yes/No |
| **Out-of-Domain** | *[Measured]* | *[Measured]* | 100% (Refusal) | N/A |
