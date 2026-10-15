"""
build_paper_v4.py
Builds OfflineDocAI_paper_v4.docx using python-docx.

Changes from v3:
  - Abstract rewritten to match actual measured numbers
  - Leftover editor note (line 59 "If you need it even shorter...") removed
  - Fabricated RAGAS / Recall@1 / hallucination-rate / ablation table REMOVED
  - Scope section clarified: answer quality not measured
  - Limitations section expanded to note missing quality metrics
  - New Section IX: Proposed Future Experiments added
  - IEEE two-column style approximated with body text styles
"""

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

doc = Document()

# ── page margins (narrow, IEEE-like) ──────────────────────────────────────────
for section in doc.sections:
    section.top_margin    = Inches(0.9)
    section.bottom_margin = Inches(1.0)
    section.left_margin   = Inches(0.75)
    section.right_margin  = Inches(0.75)

# ── helper functions ───────────────────────────────────────────────────────────
def heading(text, level=1):
    p = doc.add_paragraph(text, style=f'Heading {level}')
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return p

def body(text, bold=False, italic=False):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold   = bold
    run.italic = italic
    p.paragraph_format.space_after  = Pt(4)
    p.paragraph_format.space_before = Pt(2)
    return p

def label(text):
    """Bold label paragraph (table title, figure caption)."""
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return p

def add_table(headers, rows):
    table = doc.add_table(rows=1+len(rows), cols=len(headers))
    table.style = 'Table Grid'
    hrow = table.rows[0]
    for i, h in enumerate(headers):
        cell = hrow.cells[i]
        cell.text = h
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for r_i, row in enumerate(rows):
        tr = table.rows[r_i+1]
        for c_i, val in enumerate(row):
            tr.cells[c_i].text = str(val)
    doc.add_paragraph()   # spacer

def hr():
    doc.add_paragraph('─' * 80)

# ══════════════════════════════════════════════════════════════════════════════
# TITLE & AUTHORS
# ══════════════════════════════════════════════════════════════════════════════
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run(
    'OfflineDocAI: An Efficient End-to-End On-Device\n'
    'Retrieval-Augmented Generation System for Edge Hardware'
)
run.bold      = True
run.font.size = Pt(16)

authors = doc.add_paragraph()
authors.alignment = WD_ALIGN_PARAGRAPH.CENTER
authors.add_run(
    'Ravinarayana U\n'
    'St. Aloysius Institute of Management and Information Technology (AIMIT)\n'
    'Mangalore, Karnataka\nRavinarayanau2@gmail.com\n\n'
    'Siddhartha Bhat S\n'
    'St. Aloysius Institute of Management and Information Technology (AIMIT)\n'
    'Mangalore, Karnataka\nSiddusbs2810@gmail.com'
)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# ABSTRACT  (v4 — matches actual measured numbers)
# ══════════════════════════════════════════════════════════════════════════════
heading('Abstract', 2)
body(
    'Retrieval-augmented generation (RAG) usually relies on cloud servers, making it '
    'unsuitable for confidential documents or environments with unreliable network access. '
    'This paper presents OfflineDocAI, an Android application that executes the complete '
    'RAG pipeline entirely on-device with zero network data egress: document parsing, '
    'structure-aware chunking, ONNX-based embedding, hybrid dense and BM25 retrieval fused '
    'by Reciprocal Rank Fusion (RRF), adaptive context assembly, and answer generation using '
    'quantized small language models via llama.cpp. We evaluate OfflineDocAI on three '
    'commodity smartphones (4 GB, 6 GB, and 8 GB RAM) using a 35-page PDF across 18 '
    'generation runs per model and device. Document ingestion completed in 5.9–14.1 s '
    '(2.5–6.0 pages/s), with a mean retrieval latency of 79–186 ms. Decoding achieved '
    '8.2–19.4 tokens/s for Qwen 2.5 0.5B and 3.7–10.7 tokens/s for Llama 3.2 1B, '
    'maintaining a mean peak memory footprint of 0.77–1.99 GB. Time-to-first-token '
    '(14.7–54.1 s), driven by prompt prefill, was the primary system bottleneck. Our '
    'results show that fully private, offline document question answering is practical on '
    'consumer-grade mobile devices, and that future gains depend principally on faster '
    'prompt processing rather than faster retrieval.'
)

body(
    'Keywords — Retrieval-Augmented Generation, On-Device AI, Edge Computing, Small '
    'Language Models, Hybrid Retrieval, Reciprocal Rank Fusion, Privacy-Preserving AI, '
    'Android, llama.cpp',
    italic=True
)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# I. INTRODUCTION
# ══════════════════════════════════════════════════════════════════════════════
heading('I. INTRODUCTION')

heading('A. Background of the Study', 2)
body(
    'Large language models (LLMs) answer questions fluently, but they cannot see a user\'s '
    'private documents and may produce statements that are not supported by any source. '
    'Retrieval-augmented generation (RAG) [1], [2] addresses both problems: relevant '
    'passages are retrieved from a document collection and the model is asked to answer '
    'using only those passages. In most deployments the embedding model, the vector '
    'database and the LLM all run on remote servers.'
)

heading('B. Research Problem', 2)
body(
    'Sending documents to a server is a problem for people who work with confidential '
    'material, such as legal, medical or internal company documents, and for people with '
    'limited or no connectivity. Running RAG on a phone avoids this, but the phone imposes '
    'hard limits: a few gigabytes of RAM shared with the operating system, CPU-only '
    'inference on most devices, thermal limits, and small models with short context windows. '
    'Existing on-device work usually optimises one stage, such as the vector index [3], [4], '
    'document selection [5] or processor scheduling [6], and is often evaluated on a single '
    'device or on a laptop [7].'
)

heading('C. Significance of the Study', 2)
body(
    'A complete on-device pipeline keeps documents on the device, works offline and has no '
    'per-query cost. Measurements on real budget and mid-range phones also show where time '
    'and memory are actually spent, which tells developers which stage is worth optimising.'
)

heading('D. Research Gap', 2)
body(
    'Few published systems cover the whole path from a raw PDF or DOCX file to a cited '
    'answer on a phone, and fewer report the measured cost of every stage (ingestion, '
    'retrieval, prompt processing, decoding, memory and temperature) across different '
    'hardware tiers.'
)

heading('E. Objectives and Research Questions', 2)
body('The objective of this study is to design, implement and measure a complete on-device '
     'RAG system for Android. The study addresses four research questions:')
body('RQ1: Can document ingestion (parsing, chunking, embedding and storage) run on '
     'commodity phones in acceptable time?')
body('RQ2: How much latency does hybrid dense and BM25 retrieval with Reciprocal Rank '
     'Fusion add on such phones?')
body('RQ3: What decoding speed, time-to-first-token (TTFT) and memory do quantized '
     '0.5B–1.5B models reach on 4 GB, 6 GB and 8 GB devices?')
body('RQ4: Which stage dominates the response time, and how does this change with the '
     'hardware tier?')

heading('F. Scope of the Study', 2)
body(
    'The study covers Android phones, English documents with a text layer (PDF, DOCX and '
    'TXT), CPU inference, three small language models (Qwen 2.5 0.5B and 1.5B [8], Llama '
    '3.2 1B [9]) and three phones, one per RAM tier. Answer quality (citation accuracy, '
    'faithfulness, hallucination rate) was not scored in this round of experiments and is '
    'identified as the primary direction for future work. The contributions of the paper are:'
)
for c in [
    'A working Android application that runs ingestion, embedding, hybrid retrieval, '
    'context assembly and generation fully on the device.',
    'A device-adaptive inference layer that calibrates CPU thread counts per phone and '
    'sizes the retrieved evidence from the measured prompt-processing speed.',
    'A measurement study on three phones (4, 6 and 8 GB RAM) reporting per-stage latency, '
    'throughput, memory and temperature.',
]:
    p = doc.add_paragraph(c, style='List Bullet')

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# II. REVIEW OF LITERATURE
# ══════════════════════════════════════════════════════════════════════════════
heading('II. REVIEW OF LITERATURE')

heading('A. Retrieval-Augmented Generation and Retrieval Methods', 2)
body(
    'Lewis et al. [1] introduced RAG, and Gao et al. [2] survey its variants. Dense '
    'retrieval represents queries and passages as vectors [10]; Sentence-BERT [11] made '
    'sentence embeddings practical, and MiniLM [12] showed that small distilled transformers '
    'keep most of the quality at a fraction of the cost. Lexical ranking with BM25 [13] '
    'remains strong for exact terms, numbers and rare words, which dense models often miss. '
    'Reciprocal Rank Fusion (RRF) [14] merges ranked lists using only rank positions, so '
    'scores from different retrievers do not need to be calibrated. Liu et al. [15] showed '
    'that models use information in the middle of long contexts poorly, which supports '
    'giving small models short, relevant context.'
)

heading('B. On-Device Language Model Inference', 2)
body(
    'MobileLLM [16] showed that carefully designed sub-billion-parameter models are useful '
    'for on-device tasks, and LLM in a flash [17] studied running models whose weights '
    'exceed DRAM by streaming them from flash storage. The llama.cpp project [18] provides '
    'CPU inference for quantized GGUF models and is widely used on phones. Open models such '
    'as Qwen 2.5 [8] and Llama 3.2 [9] are available at 0.5B–1.5B parameters.'
)

heading('C. On-Device and Edge RAG Systems', 2)
body(
    'EdgeRAG [3] and MobileRAG [4] focus on making the vector index fit in memory and on '
    'reducing the text passed to the model. Loo et al. [19] use RAG to guide mobile app '
    'automation rather than document question answering. SRAS [5] trains a compact '
    'reinforcement-learning policy to select documents. Pocket RAG [20] targets first-aid '
    'guidance within a 2 GB memory budget. RAGdb [7] packs ingestion and hybrid retrieval '
    'into one SQLite file, evaluated on a laptop. HeRo [6] schedules agentic RAG across '
    'CPU, GPU and NPU, and SmartRAG [21] builds an on-device knowledge graph for multi-hop '
    'questions.'
)

heading('D. Security of Document-Based Prompts', 2)
body(
    'Greshake et al. [22] showed that text retrieved from documents can carry hidden '
    'instructions that hijack an LLM (indirect prompt injection). Any RAG system that '
    'places document text in the prompt must handle this risk.'
)

heading('E. Need for the Present Study', 2)
body(
    'Prior work either optimises one stage in isolation or targets a fixed domain or a '
    'laptop. There is little measured evidence of how a complete document RAG pipeline, '
    'from raw file to cited answer, behaves on ordinary phones of different RAM tiers. '
    'This study fills that gap with an implemented system and per-stage measurements on '
    'three devices.'
)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# III. RESEARCH METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════
heading('III. RESEARCH METHODOLOGY')

heading('A. Research Design and Approach', 2)
body(
    'The study follows a design-and-evaluate approach: the system was built and then '
    'evaluated quantitatively in controlled experiments on physical phones. The analysis '
    'is descriptive and comparative across devices and models.'
)

heading('B. System Architecture', 2)
body(
    'Fig. 1 shows the five stages of OfflineDocAI. All stages run inside one Android '
    'application written in Kotlin, with a C++ native library for model inference. The '
    'only network use in the application is the optional one-time download of model files; '
    'documents, queries and answers never leave the device.'
)
body('[Fig. 1. Architecture of OfflineDocAI. All five stages execute on the phone.]', italic=True)

body('1) Document ingestion.', bold=True)
body(
    'PDF files are read page by page with PDFBox-Android. Text fragments are grouped into '
    'lines by their baseline, and pages with two columns are detected from the vertical gap '
    'between columns and re-ordered into reading order. DOCX files are read with Apache '
    'POI, with heading styles kept as headings; plain text is read directly. Each page is '
    'split into four block types: heading, list item, table row and paragraph. Paragraphs '
    'split by a page break are re-joined. Blocks are packed into chunks with a target of '
    '128 and a maximum of 180 WordPiece tokens, with a 32-token overlap; a new heading '
    'always starts a new chunk. Each chunk keeps its page span and section title for '
    'citation.'
)

body('2) Embedding and storage.', bold=True)
body(
    'Each chunk is embedded with all-MiniLM-L6-v2 (INT8, ONNX Runtime) using a maximum '
    'sequence length of 256, mean pooling and L2 normalisation, giving a 384-dimensional '
    'vector. Vectors are stored as 1,536-byte binary fields in a SQLite database (Android '
    'Room) together with the document, page, chunk index, section title and embedding model '
    'identifier. Vectors produced by different embedding models are never compared.'
)

body('3) Query routing.', bold=True)
body(
    'A rule-based router assigns each question to one of six intents: PAGE_LOOKUP, '
    'PAGE_RANGE, SUMMARY, COMPARISON, ENUMERATION or GENERAL. Page intents fetch the '
    'requested pages directly, SUMMARY samples 24 chunks evenly across the document, and '
    'the other intents use hybrid search. For COMPARISON, each side of the comparison is '
    'also searched as a sub-query, and short follow-up questions are combined with the '
    'previous question.'
)

body('4) Hybrid retrieval.', bold=True)
body(
    'Chunks are retrieved in two ways. Dense search ranks chunks by the dot product of '
    'query and chunk vectors (equal to cosine similarity, since vectors are unit length). '
    'Chunks scoring at least 0.20 are ranked and the top 50 kept. In parallel, Okapi '
    'BM25 [13] (k₁ = 1.2, b = 0.75) ranks chunks by keyword match. Query words missing '
    'from the document are first mapped to the closest indexed term to tolerate spelling '
    'mistakes. The two top-50 lists are merged with Reciprocal Rank Fusion (RRF) [14]:'
)
body('    RRF(c) = Σ  1 / (60 + r_L(q, c))  [Eq. 1]', italic=True)
body(
    'summed over both lists and over the question and any sub-queries, where r_L(q,c) is '
    'the 1-based rank of chunk c in list L. The top 20 chunks are kept (40 for '
    'ENUMERATION), and the neighbours of the top three are added with half their score to '
    'restore context. If no chunk passes the similarity floor and none shares at least half '
    'of the question\'s terms, the app answers "I couldn\'t find this in the selected '
    'document(s)." without running the language model.'
)

body('5) Context assembly and generation.', bold=True)
body(
    'The evidence placed in the prompt is limited by a token budget B:'
)
body('    B = min(B_fit, B_intent, max(600, S_p · T − P_other))  [Eq. 2]', italic=True)
body(
    'where B_fit is the space left in the context window, and B_intent is 2,000 tokens for '
    'GENERAL and 2,400 for other intents. S_p · T is the number of tokens the phone can '
    'process in the target time T (20, 35 or 75 s depending on speed mode, cut to 75% or '
    '50% under thermal pressure). S_p is a moving average of measured speed (α = 0.3), and '
    'P_other is the count of other new prompt tokens. Chunks are added in priority order '
    'until the budget is full. Adjacent chunks from the same document are merged, repeated '
    'overlap (up to three sentences) is removed, and each block is labelled like '
    '[Source 2, Page 7].'
)
body(
    'Generation uses llama.cpp [18] via JNI with Q4_K_M GGUF models. The context window '
    'is 4,096 tokens on devices with at least 3.4 GB of RAM, otherwise 2,048. Decoding is '
    'greedy (temperature 0) with a repetition penalty of 1.03 and a limit of 640 new '
    'tokens. The model stays loaded between questions, and the key–value cache of the '
    'shared prompt prefix is reused. Thread counts are calibrated once per phone and model '
    'by timing candidates, and fewer threads are used when the phone is hot or in battery '
    'saver. Because document text is untrusted [22], control markers in it '
    '(e.g. <|im_start|>, [INST], <think>) are replaced with look-alike characters. After '
    'generation, citation tags are checked against the sources actually provided and '
    'normalised.'
)

heading('C. Study Area and Context', 2)
body(
    'The experiments were run in a laboratory setting on three Android phones connected to '
    'a computer over USB, each representing a common RAM tier (Table II). All phones ran '
    'the same release build of the application with the Balanced speed mode, and all used '
    'a 4,096-token context window.'
)

label('Table II: Devices used in the evaluation')
add_table(
    ['RAM tier', 'Device (model code)', 'SoC platform', 'Models tested'],
    [
        ['4 GB', 'Xiaomi Redmi Note 9 (M2003J15SC)', 'MediaTek MT6768', 'Qwen 2.5 0.5B, Llama 3.2 1B'],
        ['6 GB', 'Realme RMX3943',                   'MediaTek MT6835', 'Qwen 2.5 0.5B, Llama 3.2 1B'],
        ['8 GB', 'Motorola moto g67 power 5G',        'Qualcomm (parrot)', 'Qwen 2.5 0.5B, Llama 3.2 1B, Qwen 2.5 1.5B'],
    ]
)

heading('D. Population, Sample and Sampling Technique', 2)
body(
    'The population of interest is Android smartphones with 4–8 GB of RAM, which covers '
    'most budget and mid-range devices. Purposive sampling was used to select one phone per '
    'RAM tier. The test document was the OWASP Mobile Application Security Verification '
    'Standard [23], a public 35-page PDF (992.6 KB) that produced 172 chunks. Eighteen '
    'questions about the document were written by the authors, covering definitions, '
    'recommendations and questions that combine several topics. The router classified '
    'retrieval events as GENERAL (120) and ENUMERATION (7). Each model answered the 18 '
    'questions on each phone, giving 126 generation records, 127 retrieval records and 3 '
    'ingestion records. Qwen 2.5 1.5B was tested only on the 8 GB phone.'
)

heading('E. Data Sources, Collection Methods and Research Instruments', 2)
body(
    'All data are primary measurements from the phones. The application\'s telemetry module '
    'writes one structured log record per ingestion, retrieval and generation event, '
    'containing stage timings and token counts. A Python harness reads these records over '
    'ADB (logcat), polls dumpsys meminfo for proportional set size (PSS) and native heap, '
    'and polls dumpsys battery for the battery temperature. Events are stored as raw JSON '
    'and aggregated into CSV tables. The dense, BM25 and fusion steps run in one search '
    'call that is timed as a whole, so they are reported together as hybrid search.'
)

heading('F. Data Analysis Techniques', 2)
body(
    'TTFT is the time from the start of the generation call to the first generated token, '
    'which includes prompt processing. Effective prefill rate is prompt tokens divided by '
    'TTFT. Decode speed is generated tokens divided by decoding time. Means and standard '
    'deviations are reported per device and model; charts were produced with Matplotlib.'
)

heading('G. Ethical Considerations', 2)
body(
    'The study involved no human participants and no personal data. The test document is '
    'publicly available, the models are used under their published licences, and all '
    'processing took place on the test devices.'
)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# IV. RESULTS
# ══════════════════════════════════════════════════════════════════════════════
heading('IV. RESULTS')

heading('A. Document Ingestion', 2)
body(
    'Table III shows the ingestion of the 35-page document. The 8 GB phone was 2.4 times '
    'faster than the 4 GB phone. Embedding was the largest cost on every device (55–62% '
    'of the total), followed by PDF text extraction; chunking and database insertion '
    'together took less than 0.6 s.'
)
label('Table III: Ingestion of the 35-page OWASP MASVS PDF (172 chunks)')
add_table(
    ['Tier', 'Extraction (s)', 'Chunking (s)', 'Embedding (s)', 'DB insert (s)', 'Total (s)', 'Pages/s (chunks/s)'],
    [
        ['4 GB', '5.76', '0.30', '7.79', '0.21', '14.11', '2.48 (22.1)'],
        ['6 GB', '2.81', '0.44', '5.36', '0.08',  '8.71', '4.02 (32.1)'],
        ['8 GB', '2.08', '0.37', '3.39', '0.03',  '5.88', '5.95 (50.8)'],
    ]
)

heading('B. Retrieval Latency', 2)
body(
    'Table IV breaks down retrieval time. Mean retrieval took 186 ms on the 4 GB phone and '
    '79 ms on the 8 GB phone; medians were 118, 117 and 70 ms. Query embedding and context '
    'assembly (which counts tokens with the language model\'s tokenizer) were the largest '
    'parts, while the hybrid search itself (dense cosine + BM25 + RRF fusion) took '
    '15–53 ms. Intent routing took under 3 ms on every device.'
)
label('Table IV: Mean retrieval latency per stage (ms) — 37/36/54 retrieval events per tier')
add_table(
    ['Tier', 'Routing', 'Query embed.', 'Hybrid search\n(dense+BM25+RRF)', 'Context assembly', 'Total (median)', 'Events'],
    [
        ['4 GB', '2.7', '47.7', '52.6', '81.1', '186.1 (118)', '37'],
        ['6 GB', '1.2', '32.2', '37.9', '62.7', '136.1 (117)', '36'],
        ['8 GB', '0.6', '18.4', '14.7', '43.5',  '78.7 (70)',  '54'],
    ]
)

heading('C. Generation Speed and Time-to-First-Token', 2)
body(
    'Table V and Figs. 2 and 3 summarise generation. For the same model, decoding on the '
    '8 GB phone was 2.3–2.9 times faster than on the lower tiers. TTFT was much longer '
    'than retrieval: 25–30 s for the 0.5B model and 50–54 s for the 1B model on the lower '
    'tiers, and 14.7–19.2 s on the 8 GB phone. The 8 GB phone processed prompts 3–6 times '
    'faster (72–105 tokens/s against 15–33 tokens/s), so it reached a lower TTFT even '
    'though its adaptive evidence budget allowed longer prompts (1,284–1,534 tokens on '
    'average).'
)
label('Table V: Generation results (mean ± standard deviation, 18 runs per row)')
add_table(
    ['Tier', 'Model', 'Prompt\ntokens', 'TTFT (s)', 'Prefill\n(tok/s)', 'Decode\n(tok/s)', 'Peak PSS (MB)'],
    [
        ['4 GB', 'Qwen 2.5 0.5B', '875',   '29.6 ± 8.2',   '31.1',  '8.57 ± 1.50',  '767 (max 785)'],
        ['4 GB', 'Llama 3.2 1B',  '739',   '54.1 ± 12.7',  '15.0',  '3.67 ± 0.77',  '1,199 (max 1,226)'],
        ['6 GB', 'Qwen 2.5 0.5B', '809',   '25.2 ± 5.2',   '33.1',  '8.18 ± 1.90',  '788 (max 795)'],
        ['6 GB', 'Llama 3.2 1B',  '705',   '50.2 ± 14.6',  '15.1',  '3.79 ± 0.74',  '1,259 (max 1,274)'],
        ['8 GB', 'Qwen 2.5 0.5B', '1,534', '14.7 ± 2.7',  '105.3', '19.42 ± 0.70',  '771 (max 777)'],
        ['8 GB', 'Llama 3.2 1B',  '1,284', '15.7 ± 5.4',   '86.5', '10.66 ± 0.88', '1,990 (max 1,994)'],
        ['8 GB', 'Qwen 2.5 1.5B', '1,317', '19.2 ± 6.4',   '71.8',  '9.07 ± 1.43', '1,886 (max 2,173)'],
    ]
)
body('[Fig. 2. Decoding throughput (tokens/s) per device and model (±1 SD).]', italic=True)
body('[Fig. 3. Time-to-first-token per device and model (±1 SD).]', italic=True)

heading('D. Memory Usage', 2)
body(
    'Fig. 4 shows memory use. Qwen 2.5 0.5B stayed below 800 MB on all phones, Llama 3.2 '
    '1B used 1.20–1.26 GB on the lower tiers and 1.99 GB on the 8 GB phone, and Qwen 2.5 '
    '1.5B peaked at 2.17 GB. All 126 generation runs completed and were logged.'
)
body('[Fig. 4. Peak PSS (A) and native heap (B) per device and model.]', italic=True)

heading('E. Thermal Behaviour', 2)
body(
    'Fig. 5 shows battery temperature over 36 consecutive generation runs on each phone. '
    'Temperature rose from 35–37 °C to 40–41.4 °C and levelled off in the second half of '
    'the runs. The highest reading was 41.4 °C on the 4 GB phone. No device triggered a '
    'thermal throttle event or OOM kill during the test.'
)
body('[Fig. 5. Battery temperature over 36 consecutive runs per phone.]', italic=True)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# V. DISCUSSION
# ══════════════════════════════════════════════════════════════════════════════
heading('V. DISCUSSION')

body(
    'RQ1 (ingestion). A 35-page document was ready for questions in 6–14 s on all three '
    'phones, which is acceptable for a one-time step. Since embedding takes more than half '
    'of this time, a faster embedding model or batching would bring the largest gain; '
    'chunking and storage are already cheap.',
    bold=True
)
body(
    'RQ2 (retrieval). Hybrid retrieval with RRF added 79–186 ms on average, which is less '
    'than 1% of the TTFT on every device. At the scale of a few hundred chunks, a full '
    'scan of all vectors with BM25 is fast enough on a phone. Index structures such as '
    'those in EdgeRAG [3] and MobileRAG [4] become important for much larger collections, '
    'which this study did not test.',
    bold=True
)
body(
    'RQ3 (generation and memory). The 0.5B model ran at an interactive 8–19 tokens/s and '
    'under 800 MB on every tier, so it is the practical choice for 4 GB and 6 GB phones. '
    'The 1B and 1.5B models fit within memory on the tested phones but decoded at under '
    '4 tokens/s on the lower tiers. The higher PSS of Llama 3.2 1B on the 8 GB phone is '
    'most likely because the model file is memory-mapped and Android keeps more of its '
    'pages resident when more RAM is free; this was not examined further.',
    bold=True
)
body(
    'RQ4 (bottleneck). The response time is dominated by prompt processing, not by '
    'retrieval or decoding. On the 4 GB and 6 GB phones the effective prefill rate was '
    '15–33 tokens/s, so a prompt of about 800 tokens takes 25–55 s before the first word '
    'appears. This is why OfflineDocAI sizes the evidence from the measured prefill speed '
    '(Eq. 2) rather than from a fixed number of chunks. It also supports the observation '
    'of Liu et al. [15] that short, relevant context is preferable for small models. '
    'Pocket RAG [20] reports a 3.7 s TTFT for a narrower first-aid task; the difference '
    'is consistent with our finding that prompt length is the main driver of TTFT, '
    'although the two setups are not directly comparable.',
    bold=True
)
body(
    'Compared with prior work that improves one stage [3]–[6], these results show that '
    'once the full pipeline runs on a phone, the cost is concentrated in prompt processing, '
    'so shorter prompts, faster prefill or hardware acceleration matter more than faster '
    'retrieval.'
)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# VI. CONCLUSION
# ══════════════════════════════════════════════════════════════════════════════
heading('VI. CONCLUSION')
body(
    'This paper presented OfflineDocAI, an Android application that runs the full RAG '
    'pipeline on the device, from parsing a PDF or DOCX file to generating an answer with '
    'page citations. On three phones with 4, 6 and 8 GB of RAM, ingestion of a 35-page '
    'document took 6–14 s (2.5–6.0 pages/s), retrieval took under 0.2 s on average, the '
    '0.5B model decoded at 8–19 tokens/s within 800 MB, and models up to 1.5B parameters '
    'fit in memory on the 8 GB phone. Prompt processing was the main bottleneck, with TTFT '
    'between 15 and 54 s. The study shows that private, offline document question answering '
    'is feasible on ordinary phones, and that future gains depend mostly on faster prompt '
    'processing. Answer quality measurement — citation accuracy, faithfulness and correct '
    'refusals — is the principal open task and is addressed in Section IX.'
)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# VII. RECOMMENDATIONS AND IMPLICATIONS
# ══════════════════════════════════════════════════════════════════════════════
heading('VII. RECOMMENDATIONS AND IMPLICATIONS')
for item in [
    ('Practical:', 'Use a 0.5B model on 4–6 GB phones and reserve 1B–1.5B models for '
     '8 GB phones; keep the evidence short and size it from the measured device speed.'),
    ('Policy:', 'On-device processing keeps documents under the user\'s control and fits '
     'data-minimisation principles in data protection laws; organisations handling sensitive '
     'records can consider it where cloud processing is restricted.'),
    ('Managerial:', 'Document assistants can be given to field staff without servers or '
     'per-query fees; when larger models are needed, devices with 8 GB of RAM should be '
     'budgeted for.'),
    ('Academic:', 'On-device RAG studies should report per-stage costs across several '
     'hardware tiers, with TTFT and prefill rate reported alongside decoding speed.'),
]:
    p = doc.add_paragraph()
    p.add_run(item[0] + ' ').bold = True
    p.add_run(item[1])

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# VIII. LIMITATIONS OF THE STUDY
# ══════════════════════════════════════════════════════════════════════════════
heading('VIII. LIMITATIONS OF THE STUDY')
body(
    'The evaluation used one phone per RAM tier, one document and 18 questions, most of '
    'which were routed to the GENERAL intent, so the results may not generalise to other '
    'devices or document types. Ingestion was measured once per device. Answer quality '
    '(faithfulness, citation accuracy, hallucination rate and closed-world refusal '
    'accuracy) was not scored in this round of experiments; establishing these metrics '
    'is the primary recommendation for future work (see Section IX). Temperature was read '
    'from the battery sensor rather than the processor, and energy use was not measured. '
    'Qwen 2.5 1.5B was tested only on the 8 GB phone. Scanned PDFs without a text layer '
    'are not supported. No ablation isolating the contribution of each pipeline component '
    'to retrieval accuracy was conducted; a controlled ablation is proposed in Section IX.'
)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# IX. PROPOSED FUTURE EXPERIMENTS  (new in v4)
# ══════════════════════════════════════════════════════════════════════════════
heading('IX. PROPOSED FUTURE EXPERIMENTS')
body(
    'The following experiments are proposed to fill the gaps identified in Section VIII. '
    'None of these results are claimed in the current paper; they represent the planned '
    'next phase of evaluation.'
)

heading('A. Experiment 1: Answer Quality Evaluation (RAGAS)', 2)
body('Goal: Measure faithfulness, answer relevance and context precision of generated answers.', bold=True)
body(
    'Procedure: Use the same 18 questions on the OWASP MASVS document plus 50 additional '
    'questions with ground-truth answers manually written by the authors. Run each '
    'model/device combination. Export all (question, context, answer) triples. Evaluate '
    'offline using the RAGAS framework (pip install ragas) with a local judge model '
    '(e.g. Qwen 2.5 7B via Ollama on a desktop) to score faithfulness and answer '
    'relevance. Compute hallucination rate as the fraction of claims in the answer not '
    'entailed by the retrieved context.'
)
body('Expected output: Faithfulness score (0–1), answer relevance (0–1), hallucination rate (%) per model/device.', italic=True)
body('Estimated effort: 1–2 days for question writing + 1 day running + 1 day analysis.', italic=True)

heading('B. Experiment 2: Retrieval Quality Ablation (Recall@k, MRR)', 2)
body('Goal: Quantify the contribution of each retrieval component (Dense-only → +BM25 → +RRF → +intent routing).', bold=True)
body(
    'Procedure: Create a ground-truth retrieval dataset: for each of the 68 questions '
    '(18 original + 50 new), manually label which of the 172 OWASP MASVS chunks contain '
    'the answer ("relevant chunks"). Run four retrieval configurations in code by toggling '
    'flags in VectorStore.kt: (A0) Dense-only, (A1) BM25-only, (A2) Linear hybrid '
    '(α·dense + (1−α)·bm25), (A3) RRF k=60 (current system). Compute Recall@1, Recall@5 '
    'and MRR@10 for each configuration. No new hardware required — this runs on any PC '
    'using the existing Kotlin unit tests with the OWASP chunk index.'
)
body('Expected output: Recall@k and MRR table showing incremental gain from each component.', italic=True)
body('Estimated effort: 2 days for ground-truth labelling + 1 day coding test harness + 1 day analysis.', italic=True)

heading('C. Experiment 3: Closed-World Refusal Accuracy', 2)
body('Goal: Measure how often the system correctly refuses to answer questions whose answers are not in the document.', bold=True)
body(
    'Procedure: Write 30 out-of-scope questions (questions whose answers are not in the '
    'OWASP MASVS document). Run the full pipeline for all models and devices. A refusal '
    'is correct if the answer contains "I couldn\'t find this in the selected document(s)" '
    'or equivalent. Compute accuracy = correct_refusals / 30.'
)
body('Expected output: Refusal accuracy (%) per model, showing whether larger models refuse more reliably.', italic=True)
body('Estimated effort: 1 day for question writing + 1 day running.', italic=True)

heading('D. Experiment 4: Multi-Document and Larger Collection Scaling', 2)
body('Goal: Test system behaviour with 5–10 documents (500–2,000 chunks) to characterise how retrieval latency scales.', bold=True)
body(
    'Procedure: Load 5 diverse documents (a technical manual, a legal contract, a research '
    'paper, a financial report and the OWASP MASVS). Run 20 cross-document queries. '
    'Measure retrieval latency and PSS at each scale point. This tests whether flat-scan '
    'BM25+Dense is still sufficient or whether indexing structures become necessary.'
)
body('Expected output: Retrieval latency vs. corpus size curve per device tier.', italic=True)
body('Estimated effort: 1 day document collection + 1 day running + 1 day analysis.', italic=True)

heading('E. Experiment 5: Energy Consumption Measurement', 2)
body('Goal: Measure milliampere-hours (mAh) consumed per query across device tiers and models.', bold=True)
body(
    'Procedure: Use Android dumpsys batterystats --reset before each run and read charge '
    'counters after. Average across 18 queries per configuration. Report mAh/query and '
    'mWh/query.'
)
body('Expected output: Energy table comparable to Pocket RAG [20] and HeRo [6].', italic=True)
body('Estimated effort: 1 day scripting + 1 day running.', italic=True)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# REFERENCES
# ══════════════════════════════════════════════════════════════════════════════
heading('REFERENCES')
refs = [
    '[1] P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," NeurIPS, vol. 33, 2020, pp. 9459–9474.',
    '[2] Y. Gao et al., "Retrieval-augmented generation for large language models: A survey," arXiv:2312.10997, 2023.',
    '[3] K. Seemakhupt, S. Liu, and S. Khan, "EdgeRAG: Online-indexed RAG for edge devices," arXiv:2412.21023, 2024.',
    '[4] T. Park, G. Lee, and M.-S. Kim, "MobileRAG: A fast, memory-efficient, and energy-efficient method for on-device RAG," arXiv:2507.01079, 2025.',
    '[5] R. C. Muttur, "SRAS: A lightweight reinforcement learning-based document selector for edge-native RAG pipelines," arXiv:2601.01785, 2026.',
    '[6] M. Li et al., "HeRo: Adaptive orchestration of agentic RAG on heterogeneous mobile SoC," DAC 2026, doi:10.1145/3770743.3804310.',
    '[7] A. B. Khalid, "RAGdb: A zero-dependency, embeddable architecture for multimodal RAG on the edge," arXiv:2602.22217, 2026.',
    '[8] Qwen Team, "Qwen2.5 technical report," arXiv:2412.15115, 2024.',
    '[9] A. Dubey et al., "The Llama 3 herd of models," arXiv:2407.21783, 2024.',
    '[10] V. Karpukhin et al., "Dense passage retrieval for open-domain QA," EMNLP 2020, pp. 6769–6781.',
    '[11] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," EMNLP-IJCNLP 2019, pp. 3982–3992.',
    '[12] W. Wang et al., "MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers," NeurIPS 2020, pp. 5776–5788.',
    '[13] S. Robertson and H. Zaragoza, "The probabilistic relevance framework: BM25 and beyond," Found. Trends Inf. Retr., vol. 3, no. 4, 2009.',
    '[14] G. V. Cormack, C. L. A. Clarke, and S. Büttcher, "Reciprocal rank fusion outperforms Condorcet and individual rank learning methods," SIGIR 2009, pp. 758–759.',
    '[15] N. F. Liu et al., "Lost in the middle: How language models use long contexts," TACL, vol. 12, pp. 157–173, 2024.',
    '[16] Z. Liu et al., "MobileLLM: Optimizing sub-billion parameter language models for on-device use cases," ICML 2024.',
    '[17] K. Alizadeh et al., "LLM in a flash: Efficient large language model inference with limited memory," ACL 2024, pp. 12562–12584.',
    '[18] G. Gerganov et al., "llama.cpp: LLM inference in C/C++," GitHub, 2023. https://github.com/ggml-org/llama.cpp',
    '[19] G. Loo et al., "MobileRAG: Enhancing mobile agent with retrieval-augmented generation," arXiv:2509.03891, 2025.',
    '[20] D. H. Kang et al., "Pocket RAG: On-device RAG for first aid guidance in offline mobile environments," arXiv:2602.13229, 2026.',
    '[21] Z. Jiang et al., "SmartRAG: Native graph-based RAG for mobile device," arXiv:2607.14661, 2026.',
    '[22] K. Greshake et al., "Not what you\'ve signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection," AISec 2023, pp. 79–90.',
    '[23] OWASP Foundation, "OWASP Mobile Application Security Verification Standard (MASVS)," 2024. https://mas.owasp.org/MASVS/',
]
for r in refs:
    p = doc.add_paragraph(r)
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.first_line_indent = Inches(-0.3)
    p.paragraph_format.space_after = Pt(2)

# ── save ──────────────────────────────────────────────────────────────────────
out_path = r'C:\Users\Ravinarayana U\projects\OfflineDocAIApp\paper_drafts\OfflineDocAI_paper_v4.docx'
doc.save(out_path)
print(f'Saved: {out_path}')
