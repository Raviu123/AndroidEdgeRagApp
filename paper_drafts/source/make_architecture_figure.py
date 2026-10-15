import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams['font.family']='Times New Roman'
fig,ax=plt.subplots(figsize=(12,5.2),dpi=300); ax.set_xlim(0,100); ax.set_ylim(0,46); ax.axis('off')
stages=[("1. Ingestion",["PDF: PDFBox-Android","  (column-aware order)","DOCX: Apache POI; TXT","Blocks: heading, list,","  table row, paragraph","Chunks: 128 / 180 tokens,","  32-token overlap"]),
("2. Embedding & Storage",["all-MiniLM-L6-v2","  (INT8, ONNX Runtime)","Mean pooling + L2 norm","384-d vector","Room / SQLite","  (1536-byte BLOB)"]),
("3. Routing & Retrieval",["Rule-based intent router","  (6 intents)","Dense top-50 (cos≥0.20)","BM25 top-50","Reciprocal Rank Fusion","  (k = 60)","± 1 neighbour expansion"]),
("4. Context Assembly",["Relevance gate","Adaptive evidence budget","  (context fit, intent cap,","   measured prefill speed)","Overlap removal","Prompt sanitisation"]),
("5. Local Generation",["llama.cpp via JNI","GGUF Q4_K_M SLMs","Calibrated thread counts","KV-cache prefix reuse","Greedy decoding (T = 0)","Citation check","  [Source N, Page P]"])]
w=18.2; gap=1.2; x0=1
for i,(t,items) in enumerate(stages):
    x=x0+i*(w+gap)
    ax.add_patch(FancyBboxPatch((x,8),w,34,boxstyle="round,pad=0.3,rounding_size=1",fc='#f4f6f9',ec='#2b3a55',lw=1.3))
    ax.add_patch(FancyBboxPatch((x,37),w,5,boxstyle="round,pad=0.3,rounding_size=1",fc='#2b3a55',ec='#2b3a55'))
    ax.text(x+w/2,39.5,t,ha='center',va='center',color='white',fontsize=10.5,weight='bold')
    for j,s in enumerate(items):
        ax.text(x+0.8,34.5-j*3.4,("• " if not s.startswith("  ") else "   ")+s.strip(),ha='left',va='top',fontsize=9)
    if i<4: ax.annotate('',xy=(x+w+gap+0.1,25),xytext=(x+w-0.1,25),arrowprops=dict(arrowstyle='-|>',color='#2b3a55',lw=1.8))
ax.add_patch(FancyBboxPatch((1,1),96.5,4.5,boxstyle="round,pad=0.3,rounding_size=1",fc='white',ec='#2b3a55',ls='--',lw=1))
ax.text(49.5,3.25,"All stages run on the Android device. Documents, embeddings, queries and answers are never sent off the device.",ha='center',va='center',fontsize=11.5,style='italic')
plt.savefig(r"C:\Users\Ravinarayana U\projects\OfflineDocAIApp\paper_drafts\figures\fig_architecture.png",bbox_inches='tight',facecolor='white')
