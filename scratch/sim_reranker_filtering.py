import sys
sys.stdout.reconfigure(encoding='utf-8')
import onnxruntime as ort
import numpy as np
import sqlite3

# Load vocab
with open('vocab.txt', 'r', encoding='utf-8') as f:
    vocab = {line.strip(): i for i, line in enumerate(f)}

def tokenize_pair(q, text, max_len=256):
    tokens = ['[CLS]'] + q.lower().split() + ['[SEP]'] + text.lower().split() + ['[SEP]']
    input_ids = [vocab.get(t, vocab.get('[UNK]', 100)) for t in tokens][:max_len]
    token_type_ids = [0 if i <= tokens.index('[SEP]') else 1 for i in range(len(input_ids))]
    attention_mask = [1] * len(input_ids)
    
    pad_len = max_len - len(input_ids)
    input_ids += [0] * pad_len
    token_type_ids += [0] * pad_len
    attention_mask += [0] * pad_len
    
    return np.array([input_ids], dtype=np.int64), np.array([attention_mask], dtype=np.int64), np.array([token_type_ids], dtype=np.int64)

session_rerank = ort.InferenceSession('ms-marco-MiniLM-L-6-v2.onnx')

conn = sqlite3.connect('offline_docai_device.db')
c = conn.cursor()

# The 28 candidates from hybrid search
top_28_cids = [553, 510, 803, 657, 1003, 1538, 577, 1539, 1598, 996, 690, 497, 509, 1789, 1309, 1312, 1433, 1187, 496, 1271, 1313, 1841, 1599, 1600, 554, 1790, 1281, 1837]

question = "Did Adam Smith send a message to Li Hua about the upcoming building maintenance schedule before the administrators announced a temporary change in the construction schedule due to weather conditions?"

rerank_results = []
for cid in top_28_cids:
    c.execute('SELECT text, pageNumber FROM chunks WHERE id = ?;', (cid,))
    text, page = c.fetchone()
    ids, mask, types = tokenize_pair(question, text)
    out = session_rerank.run(None, {'input_ids': ids, 'attention_mask': mask, 'token_type_ids': types})
    logit = float(out[0][0][0])
    rerank_results.append((cid, page, logit, text[:60].replace('\n', ' ')))

rerank_results.sort(key=lambda x: x[2], reverse=True)

print("=== ALL 28 RERANKED BY CROSS-ENCODER ===")
for rank, (cid, page, logit, t) in enumerate(rerank_results):
    print(f"#{rank+1}: Chunk {cid} (p.{page}) -> logit: {logit:.3f} | {t}")

# Now apply focusOnRelevant logic from ChatRepository.kt!
# const val RERANK_KEEP_FLOOR = -6f
# const val RERANK_KEEP_MARGIN = 6f
# val margin = if (isComparison) 10f else RERANK_KEEP_MARGIN
# val kept = results.filter { r ->
#     val score = r.rerankScore ?: return@filter true
#     score >= RERANK_KEEP_FLOOR && score >= best - margin
# }

best = max(x[2] for x in rerank_results)
RERANK_KEEP_FLOOR = -6.0
margin = 10.0 # isComparison = true

print(f"\nBest logit = {best:.3f}")
print(f"RERANK_KEEP_FLOOR = {RERANK_KEEP_FLOOR}")
print(f"best - margin = {best - margin:.3f}")

kept = [x for x in rerank_results if x[2] >= RERANK_KEEP_FLOOR and x[2] >= (best - margin)]

print(f"\nKEPT CHUNKS AFTER focusOnRelevant (count: {len(kept)}):")
for rank, (cid, page, logit, t) in enumerate(kept):
    print(f"  #{rank+1}: Chunk {cid} (p.{page}) -> logit: {logit:.3f}")

print("\nDid Chunk 1538 survive?", 1538 in [x[0] for x in kept])
print("Did Chunk 1539 survive?", 1539 in [x[0] for x in kept])
