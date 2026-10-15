const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  AlignmentType, WidthType, BorderStyle, ShadingType, LevelFormat, TabStopType,
  Footer, PageNumber, HeadingLevel, VerticalAlign,
  Math: OMath, MathRun, MathFraction, MathSubScript,
} = require("docx");

const ROOT = "C:/Users/Ravinarayana U/projects/OfflineDocAIApp";
const OUT = process.env.PAPER_OUT || path.join(ROOT, "paper_drafts", "paper_v1.docx");
const PLOTS = path.join(ROOT, "benchmarks", "plots", "combined");
const FIGS = path.join(ROOT, "paper_drafts", "figures");

const FONT = "Times New Roman";
const TEXT_W = 9026; // A4 width minus 1" margins (DXA)
const LINE = 360;    // 1.5 line spacing

// ---------------------------------------------------------------- references (IEEE)
const REFS = {
  lewis: 'P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, S. Riedel, and D. Kiela, "Retrieval-augmented generation for knowledge-intensive NLP tasks," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 33, 2020, pp. 9459–9474.',
  gao: 'Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, M. Wang, and H. Wang, "Retrieval-augmented generation for large language models: A survey," arXiv preprint arXiv:2312.10997, 2023.',
  karpukhin: 'V. Karpukhin, B. Oğuz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and W. Yih, "Dense passage retrieval for open-domain question answering," in Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP), 2020, pp. 6769–6781.',
  reimers: 'N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP-IJCNLP), 2019, pp. 3982–3992.',
  minilm: 'W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou, "MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 33, 2020, pp. 5776–5788.',
  robertson: 'S. Robertson and H. Zaragoza, "The probabilistic relevance framework: BM25 and beyond," Found. Trends Inf. Retr., vol. 3, no. 4, pp. 333–389, 2009.',
  cormack: 'G. V. Cormack, C. L. A. Clarke, and S. Büttcher, "Reciprocal rank fusion outperforms Condorcet and individual rank learning methods," in Proc. 32nd Int. ACM SIGIR Conf. Res. Develop. Inf. Retr., 2009, pp. 758–759.',
  lost: 'N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang, "Lost in the middle: How language models use long contexts," Trans. Assoc. Comput. Linguist., vol. 12, pp. 157–173, 2024.',
  mobilellm: 'Z. Liu, C. Zhao, F. Iandola, C. Lai, Y. Tian, I. Fedorov, Y. Xiong, E. Chang, Y. Shi, R. Krishnamoorthi, L. Lai, and V. Chandra, "MobileLLM: Optimizing sub-billion parameter language models for on-device use cases," in Proc. 41st Int. Conf. Mach. Learn. (ICML), PMLR vol. 235, 2024.',
  alizadeh: 'K. Alizadeh, I. Mirzadeh, D. Belenko, K. Khatamifard, M. Cho, C. C. Del Mundo, M. Rastegari, and M. Farajtabar, "LLM in a flash: Efficient large language model inference with limited memory," in Proc. 62nd Annu. Meeting Assoc. Comput. Linguist. (ACL), 2024, pp. 12562–12584.',
  llamacpp: 'G. Gerganov et al., "llama.cpp: LLM inference in C/C++," GitHub repository, 2023. [Online]. Available: https://github.com/ggml-org/llama.cpp',
  qwen: 'Qwen Team, "Qwen2.5 technical report," arXiv preprint arXiv:2412.15115, 2024.',
  llama3: 'A. Dubey et al., "The Llama 3 herd of models," arXiv preprint arXiv:2407.21783, 2024.',
  edgerag: 'K. Seemakhupt, S. Liu, and S. Khan, "EdgeRAG: Online-indexed RAG for edge devices," arXiv preprint arXiv:2412.21023, 2024.',
  mobilerag: 'T. Park, G. Lee, and M.-S. Kim, "MobileRAG: A fast, memory-efficient, and energy-efficient method for on-device RAG," arXiv preprint arXiv:2507.01079, 2025.',
  mobileragagent: 'G. Loo, C. Liu, Q. Yin, X. Chen, J. Chen, J. Zhang, and Y. Tian, "MobileRAG: Enhancing mobile agent with retrieval-augmented generation," arXiv preprint arXiv:2509.03891, 2025.',
  sras: 'R. C. Muttur, "SRAS: A lightweight reinforcement learning-based document selector for edge-native RAG pipelines," arXiv preprint arXiv:2601.01785, 2026.',
  pocketrag: 'D. H. Kang, H. Lee, H. Cha, M. Choi, and S. Lim, "Pocket RAG: On-device RAG for first aid guidance in offline mobile environments," arXiv preprint arXiv:2602.13229, 2026.',
  ragdb: 'A. B. Khalid, "RAGdb: A zero-dependency, embeddable architecture for multimodal retrieval-augmented generation on the edge," arXiv preprint arXiv:2602.22217, 2026.',
  hero: 'M. Li, J. Chen, Z. Zheng, Z. Li, X. Sun, G. Luo, C. Liu, and X. Chen, "HeRo: Adaptive orchestration of agentic RAG on heterogeneous mobile SoC," in Proc. 63rd ACM/IEEE Design Autom. Conf. (DAC), 2026, doi: 10.1145/3770743.3804310.',
  smartrag: 'Z. Jiang, M. Li, S. Liu, K. Li, R. Zhou, W. Wang, X. Deng, S. Wang, and H. Dai, "SmartRAG: Native graph-based RAG for mobile device," arXiv preprint arXiv:2607.14661, 2026.',
  greshake: 'K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, and M. Fritz, "Not what you\'ve signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection," in Proc. 16th ACM Workshop Artif. Intell. Secur. (AISec), 2023, pp. 79–90.',
  owasp: 'OWASP Foundation, "OWASP Mobile Application Security Verification Standard (MASVS)," 2024. [Online]. Available: https://mas.owasp.org/MASVS/',
};
const order = [];
function citeNums(keys) {
  return keys.map((k) => {
    if (!REFS[k]) throw new Error("Unknown reference: " + k);
    if (!order.includes(k)) order.push(k);
    return order.indexOf(k) + 1;
  });
}
function citeText(keys) {
  const n = citeNums(keys).sort((a, b) => a - b);
  // IEEE: consecutive runs of 3+ collapse to a range.
  const parts = [];
  for (let i = 0; i < n.length; ) {
    let j = i;
    while (j + 1 < n.length && n[j + 1] === n[j] + 1) j++;
    if (j - i >= 2) parts.push(`[${n[i]}]–[${n[j]}]`);
    else for (let t = i; t <= j; t++) parts.push(`[${n[t]}]`);
    i = j + 1;
  }
  return parts.join(", ");
}

// ---------------------------------------------------------------- text helpers
// Inline markup: {key1,key2} = citation, **bold**, _[x] = subscript.
function runs(text, base = {}) {
  const out = [];
  const re = /\{([a-z0-9,]+)\}|\*\*(.+?)\*\*|_\[(.+?)\]/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    if (m[1]) out.push(new TextRun({ text: citeText(m[1].split(",")), ...base }));
    else if (m[2]) out.push(new TextRun({ text: m[2], ...base, bold: true }));
    else out.push(new TextRun({ text: m[3], ...base, subScript: true }));
    last = re.lastIndex;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const words = [];
function P(text, opts = {}) {
  words.push(text);
  return new Paragraph({
    alignment: opts.align ?? AlignmentType.JUSTIFIED,
    spacing: { line: LINE, after: opts.after ?? 60, before: opts.before ?? 0 },
    indent: opts.indent,
    keepNext: opts.keepNext,
    children: runs(text, opts.run || {}),
  });
}
function H1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { line: LINE, before: 180, after: 60 },
    keepNext: true,
    children: [new TextRun({ text })],
  });
}
function H2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { line: LINE, before: 60, after: 0 },
    keepNext: true,
    children: [new TextRun({ text })],
  });
}
function bullet(text, ref = "bullets") {
  words.push(text);
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    alignment: AlignmentType.JUSTIFIED,
    spacing: { line: LINE, after: 0 },
    children: runs(text),
  });
}

// ---------------------------------------------------------------- figures and tables
let figNo = 0, tabNo = 0;
const ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"];
function figure(file, widthPx, caption) {
  const buf = fs.readFileSync(file);
  // PNG IHDR holds width/height at bytes 16..23.
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  figNo++;
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: widthPx, height: Math.round((widthPx * h) / w) } })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: `Fig. ${figNo}. `, bold: true, size: 20 }), ...runs(caption, { size: 20 })],
    }),
  ];
}
// Two figures next to each other, each with its own caption, in a borderless table.
function figurePair(a, b) {
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const nb = { top: none, bottom: none, left: none, right: none };
  const half = TEXT_W / 2;
  const cellFor = ({ file, caption }) => {
    const buf = fs.readFileSync(file);
    const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
    const px = 292;
    figNo++;
    return new TableCell({
      borders: nb, width: { size: half, type: WidthType.DXA }, margins: { left: 40, right: 40 },
      children: [
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
          children: [new ImageRun({ type: "png", data: buf, transformation: { width: px, height: Math.round((px * h) / w) } })] }),
        new Paragraph({ alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: `Fig. ${figNo}. `, bold: true, size: 18 }), ...runs(caption, { size: 18 })] }),
      ],
    });
  };
  return [
    new Table({
      width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: [half, half],
      borders: { top: none, bottom: none, left: none, right: none, insideHorizontal: none, insideVertical: none },
      rows: [new TableRow({ cantSplit: true, children: [cellFor(a), cellFor(b)] })],
    }),
    new Paragraph({ spacing: { after: 60 }, children: [] }),
  ];
}
const border = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
const borders = { top: border, bottom: border, left: border, right: border };
function cell(text, width, { header = false, align = AlignmentType.CENTER } = {}) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: header ? { fill: "E7E9EE", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: align, children: runs(String(text), { size: 19, bold: header || undefined }) })],
  });
}
function table(caption, widths, header, rows, { firstColLeft = true } = {}) {
  tabNo++;
  const sum = widths.reduce((a, b) => a + b, 0);
  if (sum !== TEXT_W) throw new Error(`Table ${tabNo} widths sum to ${sum}`);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 80 },
      children: [new TextRun({ text: `Table ${ROMAN[tabNo - 1]}: `, bold: true, size: 20 }), new TextRun({ text: caption, size: 20 })],
    }),
    new Table({
      width: { size: TEXT_W, type: WidthType.DXA },
      columnWidths: widths,
      rows: [
        new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, widths[i], { header: true })) }),
        ...rows.map((r) => new TableRow({
          cantSplit: true,
          children: r.map((v, i) => cell(v, widths[i], { align: i === 0 && firstColLeft ? AlignmentType.LEFT : AlignmentType.CENTER })),
        })),
      ],
    }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}

// ---------------------------------------------------------------- equations
const mr = (t) => new MathRun(t);
const sub = (base, s) => new MathSubScript({ children: [mr(base)], subScript: [mr(s)] });
const frac = (num, den) => new MathFraction({ numerator: num, denominator: den });
let eqNo = 0;
function equation(children) {
  eqNo++;
  return new Paragraph({
    tabStops: [{ type: TabStopType.CENTER, position: TEXT_W / 2 }, { type: TabStopType.RIGHT, position: TEXT_W }],
    spacing: { before: 60, after: 120, line: LINE },
    children: [new TextRun("\t"), new OMath({ children }), new TextRun(`\t(${eqNo})`)],
  });
}

// ================================================================ CONTENT
const body = [];
const add = (...xs) => xs.flat().forEach((x) => body.push(x));

// ---- Title and authors
add(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 240, line: LINE },
  children: [new TextRun({ text: "OfflineDocAI: An Efficient End-to-End On-Device Retrieval-Augmented Generation System for Edge Hardware", bold: true, size: 32 })],
}));
for (let i = 1; i <= 3; i++) {
  add(new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 0 },
    children: [new TextRun({ text: `[Author ${i} Full Name]`, bold: true })],
  }));
  add(new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: i === 3 ? 240 : 120 },
    children: [new TextRun({ text: "[Designation, Department], [Institution], [City, State, Country]; [Email ID]; ORCID: [if available]", italics: true, size: 20 })],
  }));
}

// ---- Abstract and keywords
const ABSTRACT =
  "Retrieval-augmented generation (RAG) usually depends on cloud servers, which is unsuitable when documents are confidential or network access is unreliable. " +
  "This paper presents OfflineDocAI, an Android application that runs the complete RAG pipeline on the device: document parsing, structure-aware chunking, ONNX-based embedding, hybrid dense and BM25 retrieval fused by Reciprocal Rank Fusion, adaptive context assembly, and answer generation with quantized small language models through llama.cpp. " +
  "The system was evaluated on three commodity smartphones with 4 GB, 6 GB and 8 GB of RAM using a 35-page PDF and 18 generation runs per model and device. " +
  "Ingestion took 5.9–14.1 s and mean retrieval latency was 79–186 ms. Decoding reached 8.2–19.4 tokens/s with Qwen 2.5 0.5B and 3.7–10.7 tokens/s with Llama 3.2 1B, with mean peak memory of 0.77–1.99 GB. " +
  "Time-to-first-token (14.7–54.1 s) was the main bottleneck. The results show that private document question answering is practical on mid-range phones.";
add(new Paragraph({
  alignment: AlignmentType.JUSTIFIED, spacing: { line: LINE, after: 120 },
  children: [new TextRun({ text: "Abstract—", bold: true, italics: true }), new TextRun({ text: ABSTRACT, bold: true })],
}));
add(new Paragraph({
  alignment: AlignmentType.JUSTIFIED, spacing: { line: LINE, after: 200 },
  children: [new TextRun({ text: "Keywords—", bold: true, italics: true }), new TextRun({
    text: "Retrieval-Augmented Generation, On-Device AI, Edge Computing, Small Language Models, Hybrid Retrieval, Reciprocal Rank Fusion, Privacy-Preserving AI, Android, llama.cpp",
    italics: true,
  })],
}));

// ---- I. Introduction
add(H1("I. Introduction"));
add(H2("A. Background of the Study"));
add(P("Large language models (LLMs) answer questions fluently, but they cannot see a user's private documents and may produce statements that are not supported by any source. Retrieval-augmented generation (RAG) {lewis,gao} addresses both problems: relevant passages are retrieved from a document collection and the model is asked to answer using only those passages. In most deployments the embedding model, the vector database and the LLM all run on remote servers."));
add(H2("B. Research Problem"));
add(P("Sending documents to a server is a problem for people who work with confidential material, such as legal, medical or company documents, and for people with limited connectivity. Running RAG on a phone avoids this, but the phone imposes hard limits: a few gigabytes of RAM shared with the operating system, CPU-only inference on most devices, thermal limits, and small models with short context windows."));
add(H2("C. Significance of the Study"));
add(P("A complete on-device pipeline keeps documents on the device, works offline and has no per-query cost. Measurements on real phones also show which stage is worth optimising."));
add(H2("D. Research Gap"));
add(P("Few published systems cover the whole path from a raw PDF or DOCX file to a cited answer on a phone, and fewer report the measured cost of every stage (ingestion, retrieval, prompt processing, decoding, memory and temperature) across different hardware tiers."));
add(H2("E. Objectives and Research Questions"));
add(P("The objective of this study is to design, implement and measure a complete on-device RAG system for Android. The study addresses four research questions:", { keepNext: true }));
add(bullet("**RQ1:** Can document ingestion (parsing, chunking, embedding and storage) run on commodity phones in acceptable time?"));
add(bullet("**RQ2:** How much latency does hybrid dense and BM25 retrieval with Reciprocal Rank Fusion add on such phones?"));
add(bullet("**RQ3:** What decoding speed, time-to-first-token (TTFT) and memory do quantized 0.5B–1.5B models reach on 4 GB, 6 GB and 8 GB devices?"));
add(bullet("**RQ4:** Which stage dominates the response time, and how does this change with the hardware tier?"));
add(H2("F. Scope of the Study"));
add(P("The study covers Android phones, English documents with a text layer (PDF, DOCX and TXT), CPU inference, three small language models (Qwen 2.5 0.5B and 1.5B {qwen}, Llama 3.2 1B {llama3}) and three phones, one per RAM tier. Answer quality (citation accuracy, faithfulness) was not scored in this round of experiments. The contributions of the paper are:"));
add(bullet("A working Android application that runs ingestion, embedding, hybrid retrieval, context assembly and generation fully on the device."));
add(bullet("A device-adaptive inference layer that calibrates CPU thread counts per phone and sizes the retrieved evidence from the measured prompt-processing speed."));
add(bullet("A measurement study on three phones (4, 6 and 8 GB RAM) reporting per-stage latency, throughput, memory and temperature."));

// ---- II. Literature
add(H1("II. Review of Literature"));
add(H2("A. Retrieval-Augmented Generation and Retrieval Methods"));
add(P("Lewis et al. {lewis} introduced RAG, and Gao et al. {gao} survey its variants. Dense retrieval represents queries and passages as vectors {karpukhin}; Sentence-BERT {reimers} made sentence embeddings practical, and MiniLM {minilm} showed that small distilled transformers keep most of the quality at a fraction of the cost. Lexical ranking with BM25 {robertson} remains strong for exact terms, numbers and rare words, which dense models often miss. Reciprocal Rank Fusion (RRF) {cormack} merges ranked lists using only rank positions, so scores from different retrievers do not need to be calibrated. Liu et al. {lost} showed that models use information in the middle of long contexts poorly, which supports giving small models short, relevant context."));
add(H2("B. On-Device Language Model Inference"));
add(P("MobileLLM {mobilellm} showed that carefully designed sub-billion-parameter models are useful for on-device tasks, and LLM in a flash {alizadeh} studied running models whose weights exceed DRAM by streaming them from flash storage. The llama.cpp project {llamacpp} provides CPU inference for quantized GGUF models and is widely used on phones. Open models such as Qwen 2.5 {qwen} and Llama 3.2 {llama3} are available at 0.5B–1.5B parameters."));
add(H2("C. On-Device and Edge RAG Systems"));
add(P("Table I summarises recent edge RAG work. Most of these systems optimise one component, such as the vector index {edgerag,mobilerag}, document selection {sras} or processor scheduling {hero}, or target a specific domain {pocketrag}, task {mobileragagent,smartrag} or platform {ragdb}."));
add(table("Summary of related on-device and edge RAG work", [2400, 2300, 4326],
  ["Work", "Evaluation platform", "Main technique"],
  [
    ["EdgeRAG {edgerag}", "Edge device", "Pruned IVF index with on-demand embedding generation and caching"],
    ["MobileRAG {mobilerag}", "Smartphone", "EcoVector partitioned index and selective content reduction"],
    ["MobileRAG agent {mobileragagent}", "Smartphone (agent)", "RAG for mobile app automation tasks"],
    ["SRAS {sras}", "CPU", "0.76 MB PPO policy for document selection"],
    ["Pocket RAG {pocketrag}", "Android phone", "Domain-specific first-aid RAG under a 2 GB budget"],
    ["RAGdb {ragdb}", "Laptop (Intel i7)", "Single SQLite container with TF-IDF and substring boosting"],
    ["HeRo {hero}", "Mobile SoC (CPU/GPU/NPU)", "Heterogeneity-aware scheduling of agentic RAG"],
    ["SmartRAG {smartrag}", "Smartphone", "On-device knowledge graph with continual entity recognition"],
  ]));
add(H2("D. Security of Document-Based Prompts"));
add(P("Greshake et al. {greshake} showed that text retrieved from documents can carry hidden instructions that hijack an LLM (indirect prompt injection). Any RAG system that places document text in the prompt must handle this risk."));
add(H2("E. Need for the Present Study"));
add(P("There is little measured evidence of how a complete document RAG pipeline, from raw file to cited answer, behaves on ordinary phones of different RAM tiers. This study addresses that gap with an implemented system and per-stage measurements on three devices."));

// ---- III. Methodology
add(H1("III. Research Methodology"));
add(H2("A. Research Design and Approach"));
add(P("The study follows a design-and-evaluate approach: the system was built and then evaluated quantitatively in controlled experiments on physical phones. The analysis is descriptive and comparative across devices and models."));
add(H2("B. System Architecture"));
add(P("Fig. 1 shows the five stages of OfflineDocAI. All stages run inside one Android application written in Kotlin, with a C++ native library for model inference. The only network use in the application is the optional one-time download of model files; documents, queries and answers never leave the device."));
add(figure(path.join(FIGS, "fig_architecture.png"), 600, "Architecture of OfflineDocAI. All five stages execute on the phone."));
add(P("**1) Document ingestion.** PDF files are read page by page with PDFBox-Android. Text fragments are grouped into lines by their baseline, and pages with two columns are detected from the vertical gap between columns and re-ordered into reading order. DOCX files are read with Apache POI, with heading styles kept as headings; plain text is read directly. Each page is split into four block types: heading, list item, table row and paragraph. Paragraphs split by a page break are re-joined. Blocks are packed into chunks with a target of 128 and a maximum of 180 WordPiece tokens, with a 32-token overlap; a new heading always starts a new chunk. Each chunk keeps its page span and section title for citation."));
add(P("**2) Embedding and storage.** Each chunk is embedded with all-MiniLM-L6-v2 (INT8, ONNX Runtime) using a maximum sequence length of 256, mean pooling and L2 normalisation, giving a 384-dimensional vector. Vectors are stored as 1,536-byte binary fields in a SQLite database (Android Room) together with the document, page, chunk index, section title and embedding model identifier. Vectors produced by different embedding models are never compared."));
add(P("**3) Query routing.** A rule-based router assigns each question to one of six intents: PAGE_LOOKUP, PAGE_RANGE, SUMMARY, COMPARISON, ENUMERATION or GENERAL. Page intents fetch the requested pages directly, SUMMARY samples 24 chunks evenly across the document, and the other intents use hybrid search. For COMPARISON, each side of the comparison is also searched as a sub-query, and short follow-up questions are combined with the previous question."));
add(P("**4) Hybrid retrieval.** Because vectors are unit length, dense similarity is the dot product of the query and chunk vectors. Chunks with similarity of at least 0.20 are ranked and the top 50 kept. In parallel, chunks are scored with Okapi BM25 {robertson}, using k₁ = 1.2 and b = 0.75; query words that do not occur in the document are first mapped to the closest indexed term to tolerate spelling mistakes:", { keepNext: true }));
add(equation([
  mr("BM25(q,c) = "), sub("∑", "t∈q"), mr(" IDF(t) · "),
  frac([mr("tf(t,c) · ("), sub("k", "1"), mr(" + 1)")], [mr("tf(t,c) + "), sub("k", "1"), mr(" · (1 − b + b · |c| / avgdl)")]),
]));
add(equation([
  mr("IDF(t) = ln"), mr("( 1 + "),
  frac([mr("N − df(t) + 0.5")], [mr("df(t) + 0.5")]),
  mr(" )"),
]));
add(P("where tf(t,c) is the frequency of term t in chunk c, |c| is the chunk length, avgdl is the average chunk length, N is the number of chunks and df(t) is the number of chunks containing t. The top 50 BM25 results and the dense list are fused with RRF {cormack}, using k = 60:", { keepNext: true }));
add(equation([
  mr("RRF(c) = "), sub("∑", "q∈Q"), mr(" "), sub("∑", "L∈{dense, BM25}"), mr(" "),
  frac([mr("1")], [mr("k + "), sub("r", "L"), mr("(q,c)")]),
]));
add(P("where Q contains the question and any sub-queries, and r_[L](q,c) is the 1-based rank of chunk c in list L. The 20 highest-scoring chunks are kept (40 for ENUMERATION), and the neighbouring chunks of the top three are added with half their score to restore surrounding context. If no retrieved chunk passes the embedding model's similarity floor and none shares at least half of the question's terms, the application answers \"I couldn't find this in the selected document(s).\" without running the language model."));
add(P("**5) Context assembly and generation.** The amount of evidence placed in the prompt is limited by an evidence budget B (in tokens):", { keepNext: true }));
add(equation([
  mr("B = min( "), sub("B", "fit"), mr(", "), sub("B", "intent"), mr(", max( 600, "),
  sub("S", "p"), mr(" · T − "), sub("P", "other"), mr(" ) )"),
]));
add(P("where B_[fit] is the space left in the context window after the instructions, question and answer allowance; B_[intent] is 2,000 tokens for GENERAL and 2,400 tokens for the other intents; S_[p] is the prompt-processing speed measured on this device (an exponential moving average with α = 0.3 over previous answers); T is the target prompt-processing time of the selected speed mode (20, 35 or 75 s), reduced to 75% or 50% when Android reports thermal pressure; and P_[other] is the number of other new prompt tokens. Retrieved chunks are added in priority order until the budget is full. Adjacent chunks from the same document are merged, and up to three leading sentences of the next chunk that already appear at the end of the previous chunk are removed so the model does not read the overlap twice. Each evidence block is labelled with a tag such as [Source 2, Page 7]."));
add(P("Generation uses llama.cpp {llamacpp} compiled into the application and called through JNI, with GGUF models in Q4_K_M quantization. The context window is 4,096 tokens on devices with at least 3.4 GB of RAM and 2,048 tokens otherwise. Grounded answers use greedy decoding (temperature 0) with a repetition penalty of 1.03 and a default limit of 640 new tokens. The model stays loaded between questions and the key–value cache of the shared prompt prefix is reused. Thread counts are calibrated once per phone and model: candidate counts derived from the CPU core clusters are timed for prompt processing and decoding, and the fastest are kept; fewer threads are used when the phone is hot or in battery saver mode. Because document text is untrusted {greshake}, chat-template control markers found in it (for example <|im_start|>, [INST] or <think>) are replaced with look-alike characters before the prompt is built. After generation, citation tags are checked against the source numbers actually provided and normalised."));
add(H2("C. Study Area and Context"));
add(P("The experiments were run in a laboratory setting on three Android phones connected to a computer over USB, each representing a common RAM tier (Table II). All phones ran the same release build of the application with the Balanced speed mode, and all used a 4,096-token context window."));
add(table("Devices used in the evaluation", [1300, 2900, 2300, 2526],
  ["RAM tier", "Device (model code)", "SoC platform", "Models tested"],
  [
    ["4 GB", "Xiaomi Redmi Note 9 (M2003J15SC)", "MediaTek MT6768", "Qwen 2.5 0.5B, Llama 3.2 1B"],
    ["6 GB", "Realme RMX3943", "MediaTek MT6835", "Qwen 2.5 0.5B, Llama 3.2 1B"],
    ["8 GB", "Motorola moto g67 power 5G", "Qualcomm (parrot)", "Qwen 2.5 0.5B, Llama 3.2 1B, Qwen 2.5 1.5B"],
  ]));
add(H2("D. Population, Sample and Sampling Technique"));
add(P("The population of interest is Android smartphones with 4–8 GB of RAM, which covers most budget and mid-range devices. Purposive sampling was used to select one phone per RAM tier. The test document was the OWASP Mobile Application Security Verification Standard {owasp}, a public 35-page PDF (992.6 KB) that produced 172 chunks. Eighteen questions about the document were written by the authors, covering definitions, recommendations and questions that combine several topics. The router classified retrieval events as GENERAL (120) and ENUMERATION (7). Each model answered the 18 questions on each phone, giving 126 generation records, 127 retrieval records and 3 ingestion records. Qwen 2.5 1.5B was tested only on the 8 GB phone."));
add(H2("E. Data Sources, Collection Methods and Research Instruments"));
add(P("All data are primary measurements from the phones. The application's telemetry module writes one structured log record per ingestion, retrieval and generation event, containing stage timings and token counts. A Python harness reads these records over ADB (logcat), polls dumpsys meminfo for proportional set size (PSS) and native heap, and polls dumpsys battery for the battery temperature. Events are stored as raw JSON and aggregated into CSV tables. The dense, BM25 and fusion steps run in one search call that is timed as a whole, so they are reported together as hybrid search."));
add(H2("F. Data Analysis Techniques"));
add(P("TTFT is the time from the start of the generation call to the first generated token, which includes prompt processing. Effective prefill rate is prompt tokens divided by TTFT. Decode speed is generated tokens divided by decoding time. Means and standard deviations are reported per device and model; charts were produced with Matplotlib."));
add(H2("G. Ethical Considerations"));
add(P("The study involved no human participants and no personal data. The test document is publicly available, the models are used under their published licences, and all processing took place on the test devices."));

// ---- IV. Results
add(H1("IV. Results"));
add(H2("A. Document Ingestion"));
add(P("Table III shows the ingestion of the 35-page document. The 8 GB phone was 2.4 times faster than the 4 GB phone. Embedding was the largest cost on every device (55–62% of the total), followed by PDF text extraction; chunking and database insertion together took less than 0.6 s."));
add(table("Ingestion of the 35-page OWASP MASVS PDF (172 chunks)", [1300, 1300, 1200, 1300, 1100, 1300, 1526],
  ["Tier", "Extraction (s)", "Chunking (s)", "Embedding (s)", "DB insert (s)", "Total (s)", "Pages/s (chunks/s)"],
  [
    ["4 GB", "5.76", "0.30", "7.79", "0.21", "14.11", "2.48 (22.1)"],
    ["6 GB", "2.81", "0.44", "5.36", "0.08", "8.71", "4.02 (32.1)"],
    ["8 GB", "2.08", "0.37", "3.39", "0.03", "5.88", "5.95 (50.8)"],
  ]));
add(H2("B. Retrieval Latency"));
add(P("Table IV breaks down retrieval time. Mean retrieval took 186 ms on the 4 GB phone and 79 ms on the 8 GB phone; medians were 118, 117 and 70 ms. Query embedding and context assembly (which counts tokens with the language model's tokenizer) were the largest parts, while the hybrid search itself took 15–53 ms. Intent routing took under 3 ms."));
add(table("Mean retrieval latency per stage (ms)", [1100, 1100, 1300, 1400, 1300, 1300, 1526],
  ["Tier", "Routing", "Query embedding", "Hybrid search (dense + BM25 + RRF)", "Context assembly", "Total (median)", "Retrieval events"],
  [
    ["4 GB", "2.7", "47.7", "52.6", "81.1", "186.1 (118)", "37"],
    ["6 GB", "1.2", "32.2", "37.9", "62.7", "136.1 (117)", "36"],
    ["8 GB", "0.6", "18.4", "14.7", "43.5", "78.7 (70)", "54"],
  ]));
add(H2("C. Generation Speed and Time-to-First-Token"));
add(P("Table V and Figs. 2 and 3 summarise generation. For the same model, decoding on the 8 GB phone was 2.3–2.9 times faster than on the lower tiers. TTFT was 25–54 s on the lower tiers and 14.7–19.2 s on the 8 GB phone. The 8 GB phone processed prompts 3–6 times faster (72–105 tokens/s against 15–33 tokens/s), so it reached a lower TTFT even though its adaptive evidence budget allowed longer prompts (1,284–1,534 tokens on average)."));
add(table("Generation results (mean ± standard deviation, 18 runs per row)", [900, 1500, 1050, 1500, 1250, 1400, 1426],
  ["Tier", "Model", "Prompt tokens", "TTFT (s)", "Prefill (tok/s)", "Decode (tok/s)", "Peak PSS (MB)"],
  [
    ["4 GB", "Qwen 2.5 0.5B", "875", "29.6 ± 8.2", "31.1", "8.57 ± 1.50", "767 (max 785)"],
    ["4 GB", "Llama 3.2 1B", "739", "54.1 ± 12.7", "15.0", "3.67 ± 0.77", "1,199 (max 1,226)"],
    ["6 GB", "Qwen 2.5 0.5B", "809", "25.2 ± 5.2", "33.1", "8.18 ± 1.90", "788 (max 795)"],
    ["6 GB", "Llama 3.2 1B", "705", "50.2 ± 14.6", "15.1", "3.79 ± 0.74", "1,259 (max 1,274)"],
    ["8 GB", "Qwen 2.5 0.5B", "1,534", "14.7 ± 2.7", "105.3", "19.42 ± 0.70", "771 (max 777)"],
    ["8 GB", "Llama 3.2 1B", "1,284", "15.7 ± 5.4", "86.5", "10.66 ± 0.88", "1,990 (max 1,994)"],
    ["8 GB", "Qwen 2.5 1.5B", "1,317", "19.2 ± 6.4", "71.8", "9.07 ± 1.43", "1,886 (max 2,173)"],
  ]));
add(figurePair(
  { file: path.join(PLOTS, "fig1_multitier_token_generation_throughput.png"), caption: "Decoding throughput (tokens/s) per device and model (±1 SD)." },
  { file: path.join(PLOTS, "fig2_ttft_prefill_latency_across_hardware.png"), caption: "Time-to-first-token per device and model (±1 SD)." }));
add(H2("D. Memory Usage"));
add(P("Fig. 4 shows memory use. Qwen 2.5 0.5B stayed below 800 MB on all phones, Llama 3.2 1B used 1.20–1.26 GB on the lower tiers and 1.99 GB on the 8 GB phone, and Qwen 2.5 1.5B peaked at 2.17 GB. All 126 generation runs completed and were logged."));
add(H2("E. Thermal Behaviour"));
add(P("Fig. 5 shows battery temperature over 36 consecutive generation runs on each phone. Temperature rose from 35–37 °C to 40–41.4 °C and levelled off in the second half of the runs. The highest reading was 41.4 °C on the 4 GB phone."));
add(figurePair(
  { file: path.join(PLOTS, "fig3_memory_pss_native_allocation_scaling.png"), caption: "Peak PSS (A) and native heap (B) per device and model." },
  { file: path.join(PLOTS, "fig6_thermal_stability_profile_multi_device.png"), caption: "Battery temperature over 36 consecutive runs per phone." }));

// ---- V. Discussion
add(H1("V. Discussion"));
add(P("**RQ1 (ingestion).** A 35-page document was ready for questions in 6–14 s on all three phones, which is acceptable for a one-time step. Embedding takes more than half of this time, so it is the stage to optimise."));
add(P("**RQ2 (retrieval).** Hybrid retrieval with RRF added 79–186 ms on average, which is less than 1% of the TTFT on every device. At the scale of a few hundred chunks, a full scan of all vectors with BM25 is fast enough on a phone. Index structures such as those in EdgeRAG {edgerag} and MobileRAG {mobilerag} become important for much larger collections, which this study did not test."));
add(P("**RQ3 (generation and memory).** The 0.5B model ran at an interactive 8–19 tokens/s and under 800 MB on every tier, so it is the practical choice for 4 GB and 6 GB phones. The 1B and 1.5B models fit within memory on the tested phones but decoded at under 4 tokens/s on the lower tiers. The higher PSS of Llama 3.2 1B on the 8 GB phone is most likely because the model file is memory-mapped and Android keeps more of its pages resident when more RAM is free; this was not examined further."));
add(P("**RQ4 (bottleneck).** The response time is dominated by prompt processing, not by retrieval or decoding. On the 4 GB and 6 GB phones the effective prefill rate was 15–33 tokens/s, so a prompt of about 800 tokens takes 25–55 s before the first word appears. This is why OfflineDocAI sizes the evidence from the measured prefill speed (Eq. 4) rather than from a fixed number of chunks. Short, relevant context is also preferable for answer quality {lost}. Pocket RAG {pocketrag} reports a 3.7 s TTFT for a narrower first-aid task; the difference is consistent with our finding that prompt length is the main driver of TTFT, although the two setups are not directly comparable."));
add(P("Compared with prior work that improves one stage {edgerag,mobilerag,sras,hero}, these results show that once the full pipeline runs on a phone, the cost is concentrated in prompt processing, so shorter prompts, faster prefill or hardware acceleration matter more than faster retrieval."));

// ---- VI. Conclusion
add(H1("VI. Conclusion"));
add(P("This paper presented OfflineDocAI, an Android application that runs the full RAG pipeline on the device, from parsing a PDF or DOCX file to generating an answer with page citations. On three phones with 4, 6 and 8 GB of RAM, ingestion of a 35-page document took 6–14 s, retrieval took under 0.2 s on average, the 0.5B model decoded at 8–19 tokens/s within 800 MB, and models up to 1.5B parameters fit in memory on the 8 GB phone. Prompt processing was the main bottleneck, with TTFT between 15 and 54 s. The study shows that private, offline document question answering is feasible on ordinary phones, and that future gains depend mostly on faster prompt processing."));

// ---- VII. Recommendations
add(H1("VII. Recommendations and Implications"));
add(bullet("**Practical:** use a 0.5B model on 4–6 GB phones and reserve 1B–1.5B models for 8 GB phones; keep the evidence short and size it from the measured device speed."));
add(bullet("**Policy:** on-device processing keeps documents under the user's control and fits the data-minimisation principle of data protection laws."));
add(bullet("**Managerial:** document assistants can be deployed to staff without servers or per-query fees; 8 GB devices should be budgeted for if larger models are needed."));
add(bullet("**Academic:** on-device RAG studies should report per-stage costs across several hardware tiers, with TTFT and prefill rate reported alongside decoding speed."));
add(bullet("**Future research:** measure answer quality (citation accuracy, faithfulness and correct refusals), use mobile GPUs or NPUs for prompt processing, test larger collections with many documents, measure energy use, and add OCR for scanned PDFs."));

// ---- VIII. Limitations
add(H1("VIII. Limitations of the Study"));
add(P("The evaluation used one phone per RAM tier, one document and 18 questions, most of which were routed to the GENERAL intent, so the results may not generalise to other devices or document types. Ingestion was measured once per device. Answer quality was not scored. Temperature was read from the battery sensor rather than the processor, and energy use was not measured. Qwen 2.5 1.5B was tested only on the 8 GB phone. Scanned PDFs without a text layer are not supported."));

// ---- References
add(H1("References"));
order.forEach((k, i) => {
  add(new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { line: 276, after: 80 },
    indent: { left: 540, hanging: 540 },
    tabStops: [{ type: TabStopType.LEFT, position: 540 }],
    children: [new TextRun({ text: `[${i + 1}]\t`, size: 20 }), new TextRun({ text: REFS[k], size: 20 })],
  }));
});
const unused = Object.keys(REFS).filter((k) => !order.includes(k));
if (unused.length) throw new Error("Uncited references: " + unused.join(", "));

// ================================================================ DOCUMENT
const doc = new Document({
  creator: "OfflineDocAI authors",
  title: "OfflineDocAI: An Efficient End-to-End On-Device Retrieval-Augmented Generation System for Edge Hardware",
  styles: {
    default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, allCaps: true }, paragraph: { outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, italics: true }, paragraph: { outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 270 } } } }],
    }],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 20 })] })] }),
    },
    children: body,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, buf);
  const count = (s) => s.trim().split(/\s+/).length;
  console.log("Wrote", OUT);
  console.log("Abstract words:", count(ABSTRACT));
  console.log("Body words (approx):", count(words.join(" ")));
  console.log("References:", order.length);
});
