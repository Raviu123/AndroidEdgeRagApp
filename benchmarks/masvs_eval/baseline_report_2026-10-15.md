# MASVS baseline: Qwen2.5-0.5B on Redmi Note 9 (4 GB), no reranker

- Build installed 2026-10-15 22:15; 18 questions run through the in-app batch queue.
- Raw data: `runs/baseline-norerank-1015.json`.
- Scores are graded against `gold.json` with one fixed rubric. They are not comparable to the DeepSeek sheets.

| Q | Question (short) | Right pages reached model | Junk in context | Score /10 | What went wrong |
|---|---|---|---|---|---|
| 2 | What is MASVS | yes (p11) | TOC p3-4, p26/28/31 | 5 | Called a "model"; partial category list |
| 3 | Main security areas | yes | - | 7 | 7 of 8 (missing PRIVACY) |
| 4 | Secure data storage | yes (p17-18) | TOC p3-4 | 1 | Output only "MASVS-STORAGE: Storage" (6 tokens) |
| 5 | Network measures | yes | Foreword p5 | 8 | Good: TLS, encryption, pinning |
| 6 | Purpose of crypto | yes | TOC p3, p13 | 9 | Good |
| 7 | User authentication | yes | - | 6 | Copied AUTH intro; no AUTH-1/2/3 specifics |
| 8 | Protect sensitive data | yes | p6, p34 | 6 | Crypto only; no storage controls |
| 9 | Validate external data | yes (p27) | p1, p3, p11, p15, p33 | 6 | Partly right; wrong example (memory corruption) |
| 10 | Session management | n/a (not in doc) | - | 1 | Answered with crypto text instead of "not covered" |
| 11 | Improper storage risks | yes | p5, p20 | 5 | Correct but vague |
| 12 | Data over a network | yes (p23-24) | p11, p18 | 1 | "securely stores sensitive data" (wrong topic) |
| 13 | Privacy | yes | TOC p4, p11 | 6 | Baseline/GDPR scope only; no PRIVACY-1..4 |
| 14 | Resilience | yes | TOC p3-4, p11 | 8 | Good |
| 15 | Crypto keys | yes (p19-20) | **Foreword p5 ranked #1** | 2 | Listed Foreword topics |
| 16 | Storage + crypto | yes | p5 | 6 | Pasted both passages, little synthesis |
| 17 | Auth + session | partly (session not in doc) | - | 6 | Auth fine; didn't say session isn't covered |
| 18 | Network + cert validation | yes | p5, TOC p3-4 | 3 | Good start, then invented per-category text; hit the 400-token cap |
| 19 | Storage + privacy + crypto | 2 of 3 (no privacy page) | p5 | 4 | Crypto paragraph only |

**Total: 90/180 = 50%.** Retrieval reached the right pages in 17 of 18 questions.

## Findings

1. **Retrieval mostly finds the right pages, but they come mixed with junk.** The table of contents (p3-4) shows up in 7 questions, even though a TOC filter exists. The Foreword (p5) shows up in 6. When junk ranks first (Q15) or a neighbouring control's text is nearby (Q12, storage), the 0.5B copies the wrong passage. The fix is to choose and filter passages better (reranker plus junk removal), not to find more.
2. **Prompt processing dominates the time.** Prefill takes 18-49 s per question, at roughly 20-25 prompt tokens/s. Generation takes about 3-15 s, at 5-8 tokens/s. The mean is 41.7 s per question. Every evidence token the reranker removes saves time, so its own cost (~1 s, to be measured) is small by comparison.
3. **Not-in-document questions (10, 17) get confident wrong answers.** The word-overlap gate passes them.
4. **Q18 ran to the 400-token cap and invented content** after a correct first part.
