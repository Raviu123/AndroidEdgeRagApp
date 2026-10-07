import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

os.makedirs('figures', exist_ok=True)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# Color palette (academic blue/teal/slate)
c_card = '#FFFFFF'
c_border = '#CBD5E1'
c_primary = '#1E3A8A'  # Deep Navy
c_accent1 = '#0284C7'  # Ocean Blue
c_accent2 = '#0D9488'  # Teal
c_accent3 = '#D97706'  # Amber
c_accent4 = '#4F46E5'  # Indigo
c_text_dark = '#0F172A'
c_text_muted = '#475569'

# -------------------------------------------------------------
# FIGURE 1: Complete System Architecture Pipeline
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Header banner
ax.text(50, 96, "OfflineDocAI: End-to-End On-Device RAG Architecture", 
        fontsize=14, fontweight='bold', ha='center', va='center', color=c_primary)
ax.text(50, 92.5, "100% Air-Gapped Zero-Egress Mobile Edge Processing Pipeline", 
        fontsize=10, fontstyle='italic', ha='center', va='center', color=c_text_muted)

# Subsystem Boxes (5 distinct stages)
stages = [
    {"x": 2, "w": 17, "title": "1. Document Ingestion", "color": c_accent1, 
     "items": ["• Multi-Format Parser\n  (PDFBox, POI, UTF-8)", "• Hierarchical Chunker\n  (Headings, Tables, Lists)", "• Page-Span Mapping\n  (Attribution Preserved)"]},
    {"x": 21.5, "w": 17, "title": "2. Embedding & Storage", "color": c_accent2, 
     "items": ["• ONNX Runtime Edge\n  (all-MiniLM-L6-v2)", "• 384D Dense Vector\n  (Mean Pool + L2 Norm)", "• Room SQLite DB\n  (Zero-Copy BLOB Store)"]},
    {"x": 41, "w": 17, "title": "3. Intent & Retrieval", "color": c_accent3, 
     "items": ["• QueryIntentRouter\n  (10 Intent Classifiers)", "• Multi-Signal Scorer\n  (Dense + BM25 + Exact)", "• Reciprocal Rank Fusion\n  (Tier A/B/C Filtering)"]},
    {"x": 60.5, "w": 17, "title": "4. Local Inference", "color": c_accent4, 
     "items": ["• Context Assembler\n  (Dynamic Token Budget)", "• llama.cpp JNI Bridge\n  (ARM NEON / AVX2)", "• SLM Model Fleet\n  (Qwen 1.5B/0.5B, Llama 1B)"]},
    {"x": 80, "w": 18, "title": "5. Citation & UI", "color": c_primary, 
     "items": ["• CitationValidator\n  (Phase 6 Grounding)", "• 4-Tier Fallback Ladder\n  (L1 Support to L4 Refusal)", "• Material 3 UI\n  (Jetpack Compose View)"]}
]

for s in stages:
    # Card Background
    rect = patches.FancyBboxPatch((s["x"], 18), s["w"], 70, boxstyle="round,pad=1,rounding_size=2",
                                  facecolor=c_card, edgecolor=s["color"], linewidth=1.8)
    ax.add_patch(rect)
    
    # Header box
    h_rect = patches.FancyBboxPatch((s["x"], 78), s["w"], 9.5, boxstyle="round,pad=0.5,rounding_size=1.5",
                                    facecolor=s["color"], edgecolor=s["color"])
    ax.add_patch(h_rect)
    ax.text(s["x"] + s["w"]/2, 82.8, s["title"], fontsize=9.5, fontweight='bold', ha='center', va='center', color='#FFFFFF')
    
    # Content boxes
    y_pos = 66
    for item in s["items"]:
        i_rect = patches.FancyBboxPatch((s["x"] + 1, y_pos - 6), s["w"] - 2, 14, boxstyle="round,pad=0.5,rounding_size=1",
                                        facecolor='#F1F5F9', edgecolor=c_border, linewidth=1)
        ax.add_patch(i_rect)
        ax.text(s["x"] + 2, y_pos + 1, item, fontsize=7.8, ha='left', va='center', color=c_text_dark, linespacing=1.2)
        y_pos -= 18

# Connection Arrows between stages
for i in range(len(stages) - 1):
    x_start = stages[i]["x"] + stages[i]["w"] + 0.3
    x_end = stages[i+1]["x"] - 0.3
    ax.annotate('', xy=(x_end, 53), xytext=(x_start, 53),
                arrowprops=dict(arrowstyle="-|>", color=c_primary, lw=2.2, mutation_scale=14))

# Bottom Air-Gap / Security Banner
b_rect = patches.FancyBboxPatch((5, 3), 90, 10, boxstyle="round,pad=0.8,rounding_size=1.5",
                                facecolor='#EFF6FF', edgecolor='#3B82F6', linewidth=1.5)
ax.add_patch(b_rect)
ax.text(50, 8, "[AIR-GAPPED EDGE ENVIRONMENT] Zero Telemetry | Zero Cloud Ingestion | Zero Data Egress (0.0 KB Egress)", 
        fontsize=8.8, fontweight='bold', ha='center', va='center', color='#1E40AF')

plt.tight_layout()
plt.savefig('figures/fig1_system_architecture.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# FIGURE 2: Chunking Strategy Comparison
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

ax.text(50, 95, "Ingestion Chunking Paradigms Comparison", fontsize=13, fontweight='bold', ha='center', color=c_primary)

# Left: Naive Chunking
rect_left = patches.FancyBboxPatch((3, 8), 45, 80, boxstyle="round,pad=1,rounding_size=2",
                                   facecolor='#FEF2F2', edgecolor='#EF4444', linewidth=1.5)
ax.add_patch(rect_left)
ax.text(25.5, 83, "Naive Sliding-Window Chunking (State 001)", fontsize=10.5, fontweight='bold', ha='center', color='#B91C1C')

n_items = [
    ("Arbitrary Word Slicing", "Splits blindly at fixed word limits (e.g. 400 words).\nFractures sentences, lists, and tables across chunk borders."),
    ("Loss of Section Context", "Discards heading hierarchy. Chunks lose topical context,\nleading to severe semantic drift in embedding space."),
    ("Table & List Destruction", "Multi-column tables flattened into unstructured text strings.\nNumeric data associations completely lost."),
    ("Context Clutter & Noise", "High retrieval noise rate; irrelevant paragraphs pollute\nsmall SLM context window, triggering hallucinations.")
]
y_n = 70
for title, desc in n_items:
    r = patches.FancyBboxPatch((5.5, y_n - 6.5), 40, 12, boxstyle="round,pad=0.5,rounding_size=1",
                               facecolor='#FFFFFF', edgecolor='#FECACA', linewidth=1)
    ax.add_patch(r)
    ax.text(7, y_n + 2, f"[FAIL] {title}", fontsize=8.5, fontweight='bold', color='#991B1B')
    ax.text(7, y_n - 2.5, desc, fontsize=7.2, color='#450A0A')
    y_n -= 15

# Right: Structure-Preserving Chunking
rect_right = patches.FancyBboxPatch((52, 8), 45, 80, boxstyle="round,pad=1,rounding_size=2",
                                    facecolor='#F0FDF4', edgecolor='#22C55E', linewidth=1.5)
ax.add_patch(rect_right)
ax.text(74.5, 83, "Structure-Preserving Hierarchical Chunking (v1.0)", fontsize=10.5, fontweight='bold', ha='center', color='#15803D')

p_items = [
    ("Heading & Section Inheritance", "Detects titles, sections, and clauses. Propagates\nsectionTitle metadata to all child chunks for grounding."),
    ("Table Row Integrity", "Detects ASCII/Markdown table rows (|) and tab columns.\nPreserves complete rows and relational data."),
    ("List & Bullet Protection", "Preserves unordered (bullets) and ordered (numbered) lists\nas atomic blocks without splitting."),
    ("Page-Span Attribution", "Tracks continuous page boundaries (e.g., '14-15').\nEnables exact physical source verification.")
]
y_p = 70
for title, desc in p_items:
    r = patches.FancyBboxPatch((54.5, y_p - 6.5), 40, 12, boxstyle="round,pad=0.5,rounding_size=1",
                               facecolor='#FFFFFF', edgecolor='#BBF7D0', linewidth=1)
    ax.add_patch(r)
    ax.text(56, y_p + 2, f"[PASS] {title}", fontsize=8.5, fontweight='bold', color='#166534')
    ax.text(56, y_p - 2.5, desc, fontsize=7.2, color='#052E16')
    y_p -= 15

plt.tight_layout()
plt.savefig('figures/fig2_chunking_comparison.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# FIGURE 3: Hybrid Retrieval & Multi-Signal Reranking Architecture
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

ax.text(50, 96, "Hybrid Multi-Signal Retrieval & Dynamic Reranking Architecture", 
        fontsize=13, fontweight='bold', ha='center', color=c_primary)

# User Query Node
q_box = patches.FancyBboxPatch((38, 82), 24, 9, boxstyle="round,pad=0.8,rounding_size=1.5",
                               facecolor='#E0E7FF', edgecolor=c_accent4, linewidth=1.5)
ax.add_patch(q_box)
ax.text(50, 86.5, "User Query (Prompt)", fontsize=9.5, fontweight='bold', ha='center', va='center', color='#3730A3')

# QueryIntentRouter Box
ir_box = patches.FancyBboxPatch((30, 65), 40, 11, boxstyle="round,pad=0.8,rounding_size=1.5",
                                facecolor='#FEF3C7', edgecolor=c_accent3, linewidth=1.5)
ax.add_patch(ir_box)
ax.text(50, 72, "Deterministic QueryIntentRouter", fontsize=9.5, fontweight='bold', ha='center', color='#92400E')
ax.text(50, 68, "10 Intents  |  10 Typed Factual Entity Extractors", fontsize=8, ha='center', color='#78350F')

# Arrow from Query to Router
ax.annotate('', xy=(50, 76), xytext=(50, 82),
            arrowprops=dict(arrowstyle="-|>", color=c_primary, lw=1.8, mutation_scale=12))

# 3 Parallel Retrieval Channels
channels = [
    {"x": 6, "w": 26, "title": "Dense Vector Search", "sub": "ONNX all-MiniLM-L6-v2\n384D Cosine Similarity", "color": c_accent2},
    {"x": 37, "w": 26, "title": "Lexical BM25 / TF-IDF", "sub": "Inverted term frequency\nStop-word suppression", "color": c_accent1},
    {"x": 68, "w": 26, "title": "Exact Phrase & Entity Boost", "sub": "Verbatim (+50.0) / Term (+20.0)\nRegex & Number Matching", "color": '#8B5CF6'}
]

for ch in channels:
    b = patches.FancyBboxPatch((ch["x"], 43), ch["w"], 14, boxstyle="round,pad=0.8,rounding_size=1.5",
                               facecolor='#FFFFFF', edgecolor=ch["color"], linewidth=1.6)
    ax.add_patch(b)
    ax.text(ch["x"] + ch["w"]/2, 53, ch["title"], fontsize=8.8, fontweight='bold', ha='center', color=ch["color"])
    ax.text(ch["x"] + ch["w"]/2, 47.5, ch["sub"], fontsize=7.2, ha='center', color=c_text_muted, linespacing=1.2)

# Arrows from Router to 3 Channels
ax.annotate('', xy=(19, 57), xytext=(40, 65), arrowprops=dict(arrowstyle="-|>", color=c_accent2, lw=1.5))
ax.annotate('', xy=(50, 57), xytext=(50, 65), arrowprops=dict(arrowstyle="-|>", color=c_accent1, lw=1.5))
ax.annotate('', xy=(81, 57), xytext=(60, 65), arrowprops=dict(arrowstyle="-|>", color='#8B5CF6', lw=1.5))

# RRF Box
rrf_box = patches.FancyBboxPatch((26, 24), 48, 12, boxstyle="round,pad=0.8,rounding_size=1.5",
                                 facecolor='#F1F5F9', edgecolor=c_primary, linewidth=1.8)
ax.add_patch(rrf_box)
ax.text(50, 32, "Reciprocal Rank Fusion (RRF) & Multi-Signal Blending", fontsize=9.2, fontweight='bold', ha='center', color=c_primary)
ax.text(50, 27.5, "RRF(d) = 1/(60 + rank_dense) + 1/(60 + rank_bm25) + Score_exact", 
        fontsize=8.5, ha='center', color='#1E293B')

# Arrows from 3 channels to RRF
for ch in channels:
    ax.annotate('', xy=(50, 36), xytext=(ch["x"] + ch["w"]/2, 43),
                arrowprops=dict(arrowstyle="-|>", color=c_primary, lw=1.4))

# Output Relevance Tiers
tiers_box = patches.FancyBboxPatch((10, 5), 80, 12, boxstyle="round,pad=0.8,rounding_size=1.5",
                                   facecolor='#FFFFFF', edgecolor='#059669', linewidth=1.5)
ax.add_patch(tiers_box)
ax.text(50, 13.5, "Relevance Tier Partitioning & Dynamic Context Budgeting", fontsize=9, fontweight='bold', ha='center', color='#059669')
ax.text(23, 8.5, "[Tier A: Direct Answer]\n(Top-1 Fact / Target Page)", fontsize=7.5, ha='center', color='#065F46')
ax.text(50, 8.5, "[Tier B: Supporting Context]\n(Top-3 Semantic Neighbors)", fontsize=7.5, ha='center', color='#854D0E')
ax.text(77, 8.5, "[Tier C: Discarded Noise]\n(Filtered out before Prompt)", fontsize=7.5, ha='center', color='#991B1B')

ax.annotate('', xy=(50, 17), xytext=(50, 24),
            arrowprops=dict(arrowstyle="-|>", color='#059669', lw=1.6))

plt.tight_layout()
plt.savefig('figures/fig3_retrieval_flow.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# FIGURE 4: Phase 6 Citation Grounding & Fallback Ladder
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

ax.text(50, 96, "Phase 6: Citation Grounding & Fallback Ladder Workflow", 
        fontsize=13, fontweight='bold', ha='center', color=c_primary)

ladders = [
    {"level": "Level 1: FULLY SUPPORTED", "tag": "L1", "color": "#059669", "bg": "#ECFDF5", "y": 74,
     "condition": "All generated claims, citations, numbers, metrics, and dates verified in retrieved evidence.",
     "action": "Output raw neural generation with verified bracketed page citations [Source X, Page Y]."},
    {"level": "Level 2: FILTERED SUPPORTED", "tag": "L2", "color": "#0284C7", "bg": "#F0F9FF", "y": 53,
     "condition": "Mixed generation containing supported facts alongside unsupported hallucinated clauses.",
     "action": "Prune unsupported clauses/sentences; preserve fully grounded factual sub-statements."},
    {"level": "Level 3: GROUNDED EXTRACTIVE FALLBACK", "tag": "L3", "color": "#D97706", "bg": "#FFFBEB", "y": 32,
     "condition": "Zero claims verified, generation severe hallucination, but relevant chunks exist.",
     "action": "Bypass generation; extract top-scoring grounded sentence directly from Top-1 chunk with citation."},
    {"level": "Level 4: DETERMINISTIC NOT FOUND", "tag": "L4", "color": "#DC2626", "bg": "#FEF2F2", "y": 11,
     "condition": "Question outside document domain or [NO_EVIDENCE_FOUND] / [PAGE_NOT_FOUND] sentinel triggered.",
     "action": "State: 'I searched the provided document(s), but could not find information answering this question.'"}
]

for l in ladders:
    box = patches.FancyBboxPatch((8, l["y"]), 84, 16, boxstyle="round,pad=0.8,rounding_size=1.5",
                                 facecolor=l["bg"], edgecolor=l["color"], linewidth=1.8)
    ax.add_patch(box)
    
    # Badge
    b_badge = patches.FancyBboxPatch((10, l["y"] + 4), 10, 8, boxstyle="round,pad=0.4,rounding_size=1",
                                    facecolor=l["color"], edgecolor=l["color"])
    ax.add_patch(b_badge)
    ax.text(15, l["y"] + 8, l["tag"], fontsize=10, fontweight='bold', ha='center', va='center', color='#FFFFFF')
    
    # Level title & details
    ax.text(22, l["y"] + 11.5, l["level"], fontsize=9.5, fontweight='bold', color=l["color"], va='center')
    ax.text(22, l["y"] + 6.8, f"Condition: {l['condition']}", fontsize=7.8, color=c_text_dark, va='center')
    ax.text(22, l["y"] + 2.5, f"Execution: {l['action']}", fontsize=7.8, fontstyle='italic', color=c_text_muted, va='center')

# Arrows connecting ladder levels downwards
for y_arr in [74, 53, 32]:
    ax.annotate('', xy=(5, y_arr - 2), xytext=(5, y_arr + 4),
                arrowprops=dict(arrowstyle="-|>", color=c_text_muted, lw=1.5))
ax.text(4, 50, "Verification Fallback Progression", fontsize=7.5, rotation=90, ha='center', va='center', color=c_text_muted)

plt.tight_layout()
plt.savefig('figures/fig4_validation_ladder.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# FIGURE 5: Empirical Benchmarking Charts (Memory, Latency, Throughput)
# -------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)

# Subplot 1: Memory Footprint (RAM in MB) across pipeline phases
models = ['Qwen 2.5 0.5B', 'Llama 3.2 1B', 'Qwen 2.5 1.5B']
idle_ram = [110, 110, 110]
ingest_ram = [280, 280, 280]
infer_ram = [780, 1200, 1450]

x = np.arange(len(models))
width = 0.25

rects1 = ax1.bar(x - width, idle_ram, width, label='Baseline / Idle', color='#94A3B8')
rects2 = ax1.bar(x, ingest_ram, width, label='Active Ingestion', color='#0284C7')
rects3 = ax1.bar(x + width, infer_ram, width, label='Active Inference', color='#1E3A8A')

ax1.set_ylabel('RAM Allocation (MB)', fontsize=9, fontweight='bold')
ax1.set_title('(a) Memory Footprint Profile', fontsize=10, fontweight='bold', color=c_primary)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=8)
ax1.legend(fontsize=7.5, loc='upper left')
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# Subplot 2: Inference Token Generation Speed (tokens/sec)
speeds = [24.5, 17.2, 14.8]
colors = ['#10B981', '#3B82F6', '#6366F1']
bars = ax2.bar(x, speeds, color=colors, width=0.5, edgecolor='#1E293B', linewidth=1)
ax2.set_ylabel('Generation Speed (tokens/sec)', fontsize=9, fontweight='bold')
ax2.set_title('(b) Inference Throughput (ARM64)', fontsize=10, fontweight='bold', color=c_primary)
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=8)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
for bar in bars:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.6, f"{yval:.1f} t/s", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

# Subplot 3: End-to-End Latency by Query Intent Category (seconds)
intent_labels = ['Exact Fact\n(384 tok)', 'Comparison\n(640 tok)', 'Summary\n(1152 tok)']
ttft = [0.42, 0.78, 1.35]      # Time to First Token (Prompt eval)
gen_time = [0.85, 1.95, 3.80]  # Generation time
x3 = np.arange(len(intent_labels))

ax3.bar(x3, ttft, label='Prompt Eval (TTFT)', color='#F59E0B', width=0.45)
ax3.bar(x3, gen_time, bottom=ttft, label='Token Generation', color='#3B82F6', width=0.45)
ax3.set_ylabel('Latency (Seconds)', fontsize=9, fontweight='bold')
ax3.set_title('(c) End-to-End Latency vs. Intent', fontsize=10, fontweight='bold', color=c_primary)
ax3.set_xticks(x3)
ax3.set_xticklabels(intent_labels, fontsize=8)
ax3.legend(fontsize=7.5, loc='upper left')
ax3.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('figures/fig5_performance_benchmarks.png', dpi=300)
plt.close()
print("All 5 figures successfully generated with high-DPI quality.")
