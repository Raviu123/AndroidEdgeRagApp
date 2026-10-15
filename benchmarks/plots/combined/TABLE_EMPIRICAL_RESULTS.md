# Empirical Multi-Tier Evaluation Tables (Publication-Ready)

### Table 1: Cross-Tier Generation & Memory Scaling

| Hardware Tier | Device Model | SLM Evaluated | Decode Speed (tok/s) | Avg TTFT (s) | Peak PSS (MB) | Native Heap (MB) |
|---|---|---|---|---|---|---|
| **4GB Tier** | Redmi Note 9 | Qwen 2.5 0.5B | 8.57 ± 1.50 | 29.64 ± 8.16 | 766.9 (max 785.1) | 327.8 |
| **4GB Tier** | Redmi Note 9 | Llama 3.2 1B | 3.67 ± 0.77 | 54.11 ± 12.66 | 1199.4 (max 1226.3) | 261.1 |
| **6GB Tier** | Realme RMX3943 | Qwen 2.5 0.5B | 8.18 ± 1.90 | 25.19 ± 5.16 | 788.4 (max 795.3) | 70.0 |
| **6GB Tier** | Realme RMX3943 | Llama 3.2 1B | 3.79 ± 0.74 | 50.16 ± 14.58 | 1259.2 (max 1273.7) | 82.9 |
| **8GB Tier** | Moto g67 5G | Qwen 2.5 0.5B | 19.42 ± 0.70 | 14.70 ± 2.72 | 770.6 (max 777.3) | 98.6 |
| **8GB Tier** | Moto g67 5G | Llama 3.2 1B | 10.66 ± 0.88 | 15.71 ± 5.42 | 1990.2 (max 1993.9) | 999.9 |
| **8GB Tier** | Moto g67 5G | Qwen 2.5 1.5B | 9.07 ± 1.43 | 19.16 ± 6.38 | 1885.6 (max 2173.2) | 1122.5 |

### Table 2: Document Ingestion Performance

| Hardware Tier | Device Model | Document | Page Count | Chunks | Ingest Time (s) | Pages/sec | Chunks/sec |
|---|---|---|---|---|---|---|---|
| **4GB Tier** | M2003J15SC | OWASP_MASVS.pdf | 35 | 172 | 14.11 | 2.48 | 22.08 |
| **8GB Tier** | moto g67 power 5G | OWASP_MASVS.pdf | 35 | 172 | 5.88 | 5.95 | 50.77 |
| **6GB Tier** | RMX3943 | OWASP_MASVS.pdf | 35 | 172 | 8.71 | 4.02 | 32.08 |

### Table 3: On-Device Hybrid Retrieval Latency Breakdown

| Hardware Tier | Query Embedding (ms) | Dense Cosine (ms) | BM25 + RRF (ms) | Assembly (ms) | Total Retrieval (ms) |
|---|---|---|---|---|---|
| **4GB Tier** | 47.7 | 26.6 | 26.0 | 81.1 | **186.1** |
| **6GB Tier** | 32.2 | 19.1 | 18.7 | 62.7 | **136.1** |
| **8GB Tier** | 18.4 | 7.7 | 7.1 | 43.5 | **78.7** |
