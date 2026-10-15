#!/usr/bin/env python3
"""
OfflineDocAI - Benchmark Visualization Engine
Generates publication-quality charts for Tier-1 (4GB RAM) hardware using the 0.5B model.
Target Folder: benchmarks/plots/tier_4gb_redmi_half_billion_model/
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# Ensure utf-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
PLOTS_DIR = BASE_DIR / "plots" / "tier_4gb_redmi_half_billion_model"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Publication styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#444444'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.color'] = '#E2E8F0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

COLOR_PRIMARY = '#10A37F'   # ChatGPT Emerald
COLOR_ACCENT = '#2563EB'    # Deep Blue
COLOR_WARN = '#F59E0B'      # Amber / Orange
COLOR_PURPLE = '#7C3AED'    # Violet
COLOR_DARK = '#1E293B'      # Slate 800

def load_data():
    gen_path = DATA_DIR / "generation_benchmarks.csv"
    ret_path = DATA_DIR / "retrieval_benchmarks.csv"
    ing_path = DATA_DIR / "ingestion_benchmarks.csv"

    df_gen = pd.read_csv(gen_path) if gen_path.exists() else pd.DataFrame()
    df_ret = pd.read_csv(ret_path) if ret_path.exists() else pd.DataFrame()
    df_ing = pd.read_csv(ing_path) if ing_path.exists() else pd.DataFrame()

    return df_gen, df_ret, df_ing

def plot_token_generation_speed(df_gen):
    if df_gen.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    
    runs = np.arange(1, len(df_gen) + 1)
    tps = df_gen['tokens_per_second'].values
    mean_tps = np.mean(tps)
    std_tps = np.std(tps)

    bars = ax.bar(runs, tps, color=COLOR_PRIMARY, edgecolor='#0B6B53', width=0.65, label='Measured Decode Speed')
    ax.axhline(mean_tps, color='#DC2626', linestyle='--', linewidth=1.8, label=f'Mean Throughput: {mean_tps:.2f} tok/s')
    ax.fill_between([0, len(df_gen) + 1], mean_tps - std_tps, mean_tps + std_tps, color='#FCA5A5', alpha=0.25, label=f'±1 Std Dev ({std_tps:.2f})')

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.2, f'{h:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color=COLOR_DARK)

    ax.set_title("Decode Throughput (Tokens / Sec) across 18 Consecutive Queries\n[Device: Redmi Note 9 (4GB RAM, MediaTek MT6768) | Model: Qwen 2.5 0.5B Q4_K_M]",
                 fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel("Benchmark Query Run Index", fontsize=10, fontweight='bold')
    ax.set_ylabel("Generation Speed (Tokens / Second)", fontsize=10, fontweight='bold')
    ax.set_xticks(runs)
    ax.set_ylim(0, max(tps) + 2.5)
    ax.grid(True, axis='y')
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95)

    plt.tight_layout()
    out = PLOTS_DIR / "fig1_token_generation_throughput.png"
    plt.savefig(out)
    plt.close()
    print(f"[+] Generated: {out.name}")

def plot_ttft_and_prefill(df_gen):
    if df_gen.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    prompt_tokens = df_gen['prompt_tokens'].values
    ttft_s = df_gen['ttft_ms'].values / 1000.0  # seconds

    scatter = ax.scatter(prompt_tokens, ttft_s, color=COLOR_ACCENT, s=80, edgecolors='#1D4ED8', zorder=5, label='Measured TTFT')

    # Fit linear trendline
    if len(prompt_tokens) > 1:
        z = np.polyfit(prompt_tokens, ttft_s, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(prompt_tokens) - 50, max(prompt_tokens) + 50, 100)
        ax.plot(x_trend, p(x_trend), color='#DC2626', linestyle='--', linewidth=1.8, label=f'Prefill Scaling Trend (Slope: {z[0]*1000:.1f} ms/token)')

    ax.set_title("Time-To-First-Token (TTFT) vs Prompt Context Length\n[Device: Redmi 4GB RAM | Model: Qwen 2.5 0.5B Q4_K_M | n_ctx: 4096]",
                 fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel("Total Prompt Context (Tokens)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Time To First Token (Seconds)", fontsize=10, fontweight='bold')
    ax.grid(True)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95)

    plt.tight_layout()
    out = PLOTS_DIR / "fig2_ttft_prefill_scaling.png"
    plt.savefig(out)
    plt.close()
    print(f"[+] Generated: {out.name}")

def plot_retrieval_latency(df_ret):
    if df_ret.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    runs = np.arange(1, len(df_ret) + 1)
    embed_ms = df_ret['query_embedding_ms'].values
    dense_ms = df_ret['dense_search_ms'].values
    bm25_ms = df_ret['bm25_search_ms'].values
    rrf_ms = df_ret['rrf_fusion_ms'].values
    assembly_ms = df_ret['context_assembly_ms'].values

    b1 = ax.bar(runs, embed_ms, label='Query Embedding (ONNX MiniLM)', color='#3B82F6', width=0.65)
    b2 = ax.bar(runs, dense_ms, bottom=embed_ms, label='Dense Vector Search', color='#10B981', width=0.65)
    b3 = ax.bar(runs, bm25_ms, bottom=embed_ms + dense_ms, label='BM25 Sparse Retrieval', color='#F59E0B', width=0.65)
    b4 = ax.bar(runs, rrf_ms, bottom=embed_ms + dense_ms + bm25_ms, label='RRF Fusion Ranker', color='#EC4899', width=0.65)
    b5 = ax.bar(runs, assembly_ms, bottom=embed_ms + dense_ms + bm25_ms + rrf_ms, label='Context Assembler', color='#8B5CF6', width=0.65)

    total_ms = df_ret['total_retrieval_ms'].values
    mean_ret = np.mean(total_ms)
    ax.axhline(mean_ret, color='#DC2626', linestyle='--', linewidth=1.5, label=f'Mean Total Retrieval: {mean_ret:.1f} ms')

    ax.set_title("On-Device RAG Retrieval Latency Breakdown across Queries\n[Corpus: OWASP MASVS (35 Pages, 172 Chunks) | Hardware: Redmi 4GB RAM]",
                 fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel("Query Run Index", fontsize=10, fontweight='bold')
    ax.set_ylabel("Latency (Milliseconds)", fontsize=10, fontweight='bold')
    ax.set_xticks(runs)
    ax.grid(True, axis='y')
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=8)

    plt.tight_layout()
    out = PLOTS_DIR / "fig3_retrieval_latency_breakdown.png"
    plt.savefig(out)
    plt.close()
    print(f"[+] Generated: {out.name}")

def plot_memory_stability(df_gen):
    if df_gen.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)

    runs = np.arange(1, len(df_gen) + 1)
    pss = df_gen['peak_pss_mb'].values
    native = df_gen['native_heap_mb'].values

    ax.plot(runs, pss, marker='o', linewidth=2.2, color='#2563EB', markersize=6, label=f'Total PSS Memory (Mean: {np.mean(pss):.1f} MB)')
    ax.plot(runs, native, marker='s', linewidth=2.0, color='#10B981', markersize=6, label=f'Native C++ Heap (Mean: {np.mean(native):.1f} MB)')

    # Add Device RAM reference lines
    ax.axhline(3840, color='#EF4444', linestyle=':', linewidth=1.5, label='Physical Device RAM Limit (3,840 MB)')
    ax.axhline(1500, color='#F59E0B', linestyle='--', linewidth=1.2, label='Conservative 4GB Safe Budget (1,500 MB)')

    ax.set_title("Hardware Memory Footprint Stability during Continuous Inference\n[Redmi Note 9 4GB RAM | Qwen 2.5 0.5B + ONNX Runtime]",
                 fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel("Sequential Query Run Index", fontsize=10, fontweight='bold')
    ax.set_ylabel("Memory Consumed (MB)", fontsize=10, fontweight='bold')
    ax.set_xticks(runs)
    ax.set_ylim(0, 4200)
    ax.grid(True)
    ax.legend(loc='center right', frameon=True, facecolor='white', framealpha=0.95, fontsize=8.5)

    plt.tight_layout()
    out = PLOTS_DIR / "fig4_memory_footprint_stability.png"
    plt.savefig(out)
    plt.close()
    print(f"[+] Generated: {out.name}")

def plot_thermal_progression(df_gen):
    if df_gen.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)

    runs = np.arange(1, len(df_gen) + 1)
    temp = df_gen['temp_c'].values

    ax.plot(runs, temp, marker='^', linewidth=2.5, color='#DC2626', markersize=7, label='Battery Surface Temperature (°C)')
    ax.fill_between(runs, min(temp) - 1, temp, color='#FEE2E2', alpha=0.5)

    ax.axhline(45.0, color='#B91C1C', linestyle='--', linewidth=1.5, label='MIUI Thermal Throttling Threshold (45.0°C)')

    for r, t in zip(runs, temp):
        ax.text(r, t + 0.25, f'{t:.1f}°', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#7F1D1D')

    ax.set_title("Thermal Load Curve under 18 Continuous Queries\n[Device: Redmi Note 9 | No Throttling Observed]",
                 fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel("Sequential Query Run Index", fontsize=10, fontweight='bold')
    ax.set_ylabel("Temperature (°C)", fontsize=10, fontweight='bold')
    ax.set_xticks(runs)
    ax.set_ylim(32, 47)
    ax.grid(True)
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95)

    plt.tight_layout()
    out = PLOTS_DIR / "fig5_thermal_temperature_progression.png"
    plt.savefig(out)
    plt.close()
    print(f"[+] Generated: {out.name}")

def plot_ingestion_summary(df_ing):
    if df_ing.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    row = df_ing.iloc[0]
    stages = ['PDFBox Extraction', 'WordPiece Chunking', 'ONNX Embeddings', 'Room DB Index']
    times_ms = [row['extraction_ms'], row['chunking_ms'], row['embedding_ms'], row['db_insert_ms']]
    colors = ['#3B82F6', '#F59E0B', '#10B981', '#8B5CF6']

    ax1.pie(times_ms, labels=stages, autopct='%1.1f%%', startangle=140, colors=colors, wedgeprops=dict(edgecolor='white', linewidth=2))
    ax1.set_title(f"Ingestion Time Breakdown\n({row['doc_name']}, {row['page_count']} Pages, {row['chunk_count']} Chunks)", fontsize=10, fontweight='bold')

    # Throughput metrics
    metrics = ['Pages / Sec', 'Chunks / Sec']
    vals = [row['pages_per_sec'], row['chunks_per_sec']]
    ax2.bar(metrics, vals, color=['#2563EB', '#10B981'], width=0.45, edgecolor='#333333')
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax2.set_title("Ingestion Throughput Rate", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Throughput (Units / Second)", fontsize=9, fontweight='bold')
    ax2.set_ylim(0, max(vals) + 4)
    ax2.grid(True, axis='y')

    plt.suptitle("On-Device Document Ingestion Performance (Redmi 4GB RAM)", fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = PLOTS_DIR / "fig6_document_ingestion_summary.png"
    plt.savefig(out)
    plt.close()
    print(f"[+] Generated: {out.name}")

def plot_executive_dashboard(df_gen, df_ret, df_ing):
    if df_gen.empty or df_ret.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)

    # Panel 1: Decode Speed
    ax1 = axes[0, 0]
    runs = np.arange(1, len(df_gen) + 1)
    tps = df_gen['tokens_per_second'].values
    ax1.plot(runs, tps, marker='o', color=COLOR_PRIMARY, linewidth=2, label='Tokens/Sec')
    ax1.axhline(np.mean(tps), color='#DC2626', linestyle='--', label=f'Mean: {np.mean(tps):.2f} tok/s')
    ax1.set_title("A. Generation Speed (Decode Tokens/s)", fontweight='bold', fontsize=10)
    ax1.set_xlabel("Query Index", fontsize=9)
    ax1.set_ylabel("Tokens / Sec", fontsize=9)
    ax1.grid(True)
    ax1.legend(loc='lower left', fontsize=8)

    # Panel 2: Total Retrieval Latency
    ax2 = axes[0, 1]
    ret_ms = df_ret['total_retrieval_ms'].values
    ax2.bar(runs, ret_ms, color=COLOR_ACCENT, width=0.6)
    ax2.axhline(np.mean(ret_ms), color='#DC2626', linestyle='--', label=f'Mean: {np.mean(ret_ms):.1f} ms')
    ax2.set_title("B. Hybrid Retrieval Latency (ms)", fontweight='bold', fontsize=10)
    ax2.set_xlabel("Query Index", fontsize=9)
    ax2.set_ylabel("Latency (ms)", fontsize=9)
    ax2.grid(True, axis='y')
    ax2.legend(loc='upper right', fontsize=8)

    # Panel 3: Memory Footprint
    ax3 = axes[1, 0]
    pss = df_gen['peak_pss_mb'].values
    native = df_gen['native_heap_mb'].values
    ax3.plot(runs, pss, marker='s', color='#7C3AED', label='Peak Total PSS (MB)')
    ax3.plot(runs, native, marker='^', color='#059669', label='Native Heap (MB)')
    ax3.set_title("C. RAM Stability Across Query Runs", fontweight='bold', fontsize=10)
    ax3.set_xlabel("Query Index", fontsize=9)
    ax3.set_ylabel("RAM (MB)", fontsize=9)
    ax3.set_ylim(0, 1000)
    ax3.grid(True)
    ax3.legend(loc='center right', fontsize=8)

    # Panel 4: Thermal Progression
    ax4 = axes[1, 1]
    temp = df_gen['temp_c'].values
    ax4.plot(runs, temp, marker='o', color='#DC2626', linewidth=2)
    ax4.set_title("D. Battery Temperature Profile (°C)", fontweight='bold', fontsize=10)
    ax4.set_xlabel("Query Index", fontsize=9)
    ax4.set_ylabel("Temp (°C)", fontsize=9)
    ax4.set_ylim(32, 45)
    ax4.grid(True)

    plt.suptitle("OfflineDocAI Empirical Evaluation Dashboard\nDevice: Redmi M2003J15SC (4GB RAM) | Model: Qwen 2.5 0.5B Q4_K_M",
                 fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    out = PLOTS_DIR / "fig7_overall_executive_dashboard.png"
    plt.savefig(out)
    plt.close()
    print(f"[+] Generated: {out.name}")

def main():
    print(f"[*] Reading recorded benchmark datasets from: {DATA_DIR}")
    df_gen, df_ret, df_ing = load_data()
    print(f"    • Loaded {len(df_gen)} LLM generation events.")
    print(f"    • Loaded {len(df_ret)} RAG retrieval events.")
    print(f"    • Loaded {len(df_ing)} Document ingestion events.")

    plot_token_generation_speed(df_gen)
    plot_ttft_and_prefill(df_gen)
    plot_retrieval_latency(df_ret)
    plot_memory_stability(df_gen)
    plot_thermal_progression(df_gen)
    plot_ingestion_summary(df_ing)
    plot_executive_dashboard(df_gen, df_ret, df_ing)
    print(f"\n[+] ALL PLOTS GENERATED IN: {PLOTS_DIR}")

if __name__ == "__main__":
    main()
