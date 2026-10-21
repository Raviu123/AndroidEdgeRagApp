package com.siddu.offlinedocai.ai.retrieval

import android.util.Log
import com.siddu.offlinedocai.data.FloatArrayConverter
import com.siddu.offlinedocai.data.dao.ChunkDao
import com.siddu.offlinedocai.data.dao.ChunkSignature
import com.siddu.offlinedocai.data.entities.ChunkEntity
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlin.math.ln
import kotlin.math.max

/** How a chunk entered the result list. */
enum class RetrievalSource {
    HYBRID,
    PAGE,
    SUMMARY_SAMPLE,
    NEIGHBOR
}

data class RetrievalResult(
    val chunk: ChunkEntity,
    /** Fused rank score (RRF) for hybrid results; larger is better. Only comparable within one query. */
    val relevanceScore: Float,
    val source: RetrievalSource,
    val intent: QueryIntent,
    val denseScore: Float = 0f,
    val bm25Score: Float = 0f,
    /** Cross-encoder logit (larger = answers the question better); null when the result was not reranked. */
    val rerankScore: Float? = null
)

/** One search string and its embedding (the user question, or a sub-query such as one side of a comparison). */
data class SearchQuery(val text: String, val embedding: FloatArray)

/**
 * Document-agnostic hybrid retriever.
 *
 *  - Dense: cosine similarity between the query embedding and every chunk embedding -> top [DENSE_K].
 *  - Lexical: Okapi BM25 over stemmed terms ([TextAnalyzer]) -> top [BM25_K]. Query terms absent
 *    from the index are typo-corrected to the closest indexed term first.
 *  - Fusion: Reciprocal Rank Fusion (k = 60) across every ranked list (one dense + one BM25 list per query).
 *  - A candidate must show *some* evidence (dense >= [MIN_DENSE] or a BM25 hit) to survive.
 *  - Rerank: with a [PassageReranker], the top [RERANK_POOL] fused candidates are re-scored against
 *    the question by a cross-encoder and re-ordered; fusion only decides which candidates it sees.
 *  - Neighbouring chunks of the best hits are appended (lower priority) to restore surrounding context.
 *
 * It contains no domain vocabulary, boilerplate lists or per-document rules.
 */
class VectorStore(private val chunkDao: ChunkDao, var reranker: PassageReranker? = null) {

    private val TAG = "VectorStore"

    companion object {
        const val DEFAULT_TOP_K = 20
        const val ENUMERATION_TOP_K = 40
        private const val DENSE_K = 50
        private const val BM25_K = 50
        private const val RRF_K = 60.0
        /** MiniLM cosine below this is noise-level similarity for English text. */
        const val MIN_DENSE = 0.20f
        private const val BM25_K1 = 1.2
        private const val BM25_B = 0.75
        private const val NEIGHBOR_SEEDS = 12
        private const val SUMMARY_SAMPLES = 24
        /** Fused candidates the cross-encoder re-scores (cost grows linearly: ~one passage per 10–40 ms on phone CPUs). */
        const val RERANK_POOL = 24
    }

    /** Pre-analysed chunk: decoded embedding + term frequencies, cached per document. */
    private class IndexedChunk(
        val entity: ChunkEntity,
        val embedding: FloatArray?,
        val termFreq: Map<String, Int>,
        val length: Int
    )

    private class IndexedDocument(
        val signature: ChunkSignature,
        val chunks: List<IndexedChunk>,
        val docFreq: Map<String, Int>
    )

    private val cache = HashMap<Long, IndexedDocument>()
    private val cacheMutex = Mutex()

    suspend fun retrieve(
        query: SearchQuery,
        documentIds: List<Long>,
        classification: QueryClassification,
        subQueries: List<SearchQuery> = emptyList(),
        topK: Int = when (classification.intent) {
            QueryIntent.ENUMERATION -> ENUMERATION_TOP_K
            QueryIntent.COMPARISON -> 28
            else -> DEFAULT_TOP_K
        },
        /** Only chunks embedded by this model are searched (null = no filter). Vectors of different models are never compared. */
        embeddingModelId: String? = null
    ): List<RetrievalResult> {
        if (documentIds.isEmpty()) return emptyList()
        val index = loadIndex(documentIds)
        val chunks = if (embeddingModelId == null) index.chunks else index.chunks.filter { it.entity.embeddingModelId == embeddingModelId }
        if (chunks.isEmpty()) return emptyList()

        return when (classification.intent) {
            QueryIntent.PAGE_LOOKUP, QueryIntent.PAGE_RANGE -> pageResults(chunks, classification)
            QueryIntent.SUMMARY -> summarySample(chunks, classification)
            else -> {
                val reranker = reranker
                val pool = if (reranker != null) maxOf(topK, RERANK_POOL) else topK
                val fused = hybrid(chunks, index.docFreq, listOf(query) + subQueries, classification, pool, expandNeighbors = false)
                val primary = if (reranker != null) rerank(reranker, query.text, fused, topK) else fused
                primary + neighbors(primary, chunks, classification)
            }
        }
    }

    /**
     * Re-orders [fused] by cross-encoder score. The relevance score becomes the sigmoid of the logit so
     * block order and neighbour priority follow the reranker. Without scores the fused order is kept.
     */
    private suspend fun rerank(reranker: PassageReranker, question: String, fused: List<RetrievalResult>, topK: Int): List<RetrievalResult> {
        if (fused.isEmpty()) return fused
        val scores = reranker.score(question, fused.map { listOfNotNull(it.chunk.sectionTitle, it.chunk.text).joinToString("\n") })
            ?: return fused.take(topK)
        val reranked = fused.mapIndexed { i, r ->
            r.copy(rerankScore = scores[i], relevanceScore = (1.0 / (1.0 + kotlin.math.exp(-scores[i].toDouble()))).toFloat())
        }.sortedByDescending { it.rerankScore }.take(topK)
        runCatching {
            Log.d(TAG, "Reranked for \"$question\":")
            reranked.take(5).forEachIndexed { rank, r ->
                Log.d(TAG, "  #${rank + 1} chunk=${r.chunk.id} page=${r.chunk.pageSpan} rerank=%.2f dense=%.3f bm25=%.2f".format(r.rerankScore, r.denseScore, r.bm25Score))
            }
        }
        return reranked
    }

    // ---------------------------------------------------------------------------------------------
    // Hybrid search
    // ---------------------------------------------------------------------------------------------

    private fun hybrid(
        chunks: List<IndexedChunk>,
        docFreq: Map<String, Int>,
        queries: List<SearchQuery>,
        classification: QueryClassification,
        topK: Int,
        expandNeighbors: Boolean
    ): List<RetrievalResult> {
        val avgLength = chunks.sumOf { it.length }.toDouble() / chunks.size
        val n = chunks.size

        val fused = HashMap<Int, Double>()
        val bestDense = FloatArray(n) { -1f }
        val bestBm25 = FloatArray(n)

        for (query in queries) {
            val dense = FloatArray(n) { i -> chunks[i].embedding?.let { cosine(query.embedding, it) } ?: -1f }
            // Misspelled words that never occur in the documents are mapped to their closest indexed term.
            val queryTerms = TextAnalyzer.correctTerms(TextAnalyzer.terms(query.text).distinct(), docFreq)
            val bm25 = FloatArray(n) { i -> bm25(chunks[i], queryTerms, docFreq, n, avgLength).toFloat() }

            for (i in 0 until n) {
                bestDense[i] = max(bestDense[i], dense[i])
                bestBm25[i] = max(bestBm25[i], bm25[i])
            }

            val denseRanking = (0 until n).filter { dense[it] >= MIN_DENSE }.sortedByDescending { dense[it] }.take(DENSE_K)
            val bm25Ranking = (0 until n).filter { bm25[it] > 0f }.sortedByDescending { bm25[it] }.take(BM25_K)
            denseRanking.forEachIndexed { rank, i -> fused.merge(i, 1.0 / (RRF_K + rank + 1), Double::plus) }
            bm25Ranking.forEachIndexed { rank, i -> fused.merge(i, 1.0 / (RRF_K + rank + 1), Double::plus) }
        }

        val ranked = fused.entries.sortedByDescending { it.value }.take(topK)
        if (ranked.isEmpty()) {
            Log.i(TAG, "No chunk passed the relevance floor for: \"${classification.rawQuery}\"")
            return emptyList()
        }

        val primary = ranked.map { (i, score) ->
            RetrievalResult(
                chunk = chunks[i].entity,
                relevanceScore = score.toFloat(),
                source = RetrievalSource.HYBRID,
                intent = classification.intent,
                denseScore = bestDense[i],
                bm25Score = bestBm25[i]
            )
        }
        logTop(primary, classification.rawQuery)
        return if (expandNeighbors) primary + neighbors(primary, chunks, classification) else primary
    }

    private fun bm25(chunk: IndexedChunk, queryTerms: List<String>, docFreq: Map<String, Int>, n: Int, avgLength: Double): Double {
        if (chunk.length == 0) return 0.0
        var score = 0.0
        for (term in queryTerms) {
            val tf = chunk.termFreq[term] ?: continue
            val df = docFreq[term] ?: continue
            val idf = max(0.0, ln(1.0 + (n - df + 0.5) / (df + 0.5)))
            // Quadratic IDF scaling emphasizes rare, discriminative query terms over vocabulary that
            // appears throughout the document.
            val queryWeight = idf * idf
            val tfWeight = (tf * (BM25_K1 + 1)) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * chunk.length / avgLength))
            score += queryWeight * tfWeight
        }
        return score
    }

    /** Adjacent chunks of the strongest hits, appended after all primary results (filled only if budget remains). */
    private fun neighbors(primary: List<RetrievalResult>, chunks: List<IndexedChunk>, classification: QueryClassification): List<RetrievalResult> {
        val byPosition = chunks.associateBy { it.entity.documentId to it.entity.chunkIndex }
        val seen = primary.map { it.chunk.id }.toMutableSet()
        val out = mutableListOf<RetrievalResult>()
        for (seed in primary.take(NEIGHBOR_SEEDS)) {
            for (offset in intArrayOf(-1, 1)) {
                val neighbor = byPosition[seed.chunk.documentId to seed.chunk.chunkIndex + offset] ?: continue
                if (seen.add(neighbor.entity.id)) {
                    out.add(RetrievalResult(neighbor.entity, seed.relevanceScore * 0.5f, RetrievalSource.NEIGHBOR, classification.intent))
                }
            }
        }
        return out
    }

    // ---------------------------------------------------------------------------------------------
    // Structural intents
    // ---------------------------------------------------------------------------------------------

    private fun pageResults(chunks: List<IndexedChunk>, classification: QueryClassification): List<RetrievalResult> {
        val start = classification.pageStart ?: return emptyList()
        val end = classification.pageEnd ?: start
        return chunks
            .filter { c -> pageRange(c.entity).let { (a, b) -> a <= end && b >= start } }
            .sortedWith(compareBy({ it.entity.documentId }, { it.entity.chunkIndex }))
            .map { RetrievalResult(it.entity, 1f, RetrievalSource.PAGE, classification.intent) }
    }

    /**
     * Evenly spread sample for whole-document summaries. Results are ordered so that *any prefix*
     * covers the document evenly (beginning, end, middle, quarters, ...), which keeps coverage
     * balanced when the context budget truncates the list.
     */
    private fun summarySample(chunks: List<IndexedChunk>, classification: QueryClassification): List<RetrievalResult> {
        val byDoc = chunks.groupBy { it.entity.documentId }.values
        val perDoc = max(4, SUMMARY_SAMPLES / byDoc.size)
        val orderedPerDoc = byDoc.map { docChunks -> spreadOrder(docChunks.size).take(perDoc).map { docChunks[it] } }
        // Interleave documents so each one is represented before any is covered in depth.
        val result = mutableListOf<RetrievalResult>()
        var round = 0
        while (result.size < orderedPerDoc.sumOf { it.size }) {
            for (list in orderedPerDoc) list.getOrNull(round)?.let {
                result.add(RetrievalResult(it.entity, 1f / (round + 1), RetrievalSource.SUMMARY_SAMPLE, classification.intent))
            }
            round++
        }
        return result
    }

    /** 0, n-1, n/2, n/4, 3n/4, n/8, ... (deduplicated). */
    internal fun spreadOrder(n: Int): List<Int> {
        if (n <= 0) return emptyList()
        val order = LinkedHashSet<Int>()
        order.add(0)
        order.add(n - 1)
        var denominator = 2
        while (order.size < n && denominator <= 2 * n) {
            for (numerator in 1 until denominator step 2) order.add((numerator.toLong() * (n - 1) / denominator).toInt())
            denominator *= 2
        }
        for (i in 0 until n) order.add(i)
        return order.toList()
    }

    // ---------------------------------------------------------------------------------------------
    // Index cache
    // ---------------------------------------------------------------------------------------------

    private class LoadedIndex(val chunks: List<IndexedChunk>, val docFreq: Map<String, Int>)

    private suspend fun loadIndex(documentIds: List<Long>): LoadedIndex = cacheMutex.withLock {
        val signatures = chunkDao.getSignatures(documentIds).associateBy { it.documentId }
        val chunks = mutableListOf<IndexedChunk>()
        val docFreq = HashMap<String, Int>()
        for (docId in documentIds.distinct().sorted()) {
            val signature = signatures[docId] ?: run { cache.remove(docId); null } ?: continue
            val cached = cache[docId]
            val indexed = if (cached != null && cached.signature == signature) cached else buildIndex(docId, signature).also { cache[docId] = it }
            chunks.addAll(indexed.chunks)
            indexed.docFreq.forEach { (term, df) -> docFreq.merge(term, df, Int::plus) }
        }
        LoadedIndex(chunks, docFreq)
    }

    private suspend fun buildIndex(docId: Long, signature: ChunkSignature): IndexedDocument {
        val rawEntities = chunkDao.getAllChunksForDocument(docId)
        val entities = rawEntities.filterNot {
            com.siddu.offlinedocai.ai.chunking.TextChunker.isTableOfContents(it.text, it.sectionTitle)
        }
        val docFreq = HashMap<String, Int>()
        val indexed = entities.map { entity ->
            val terms = TextAnalyzer.terms(listOfNotNull(entity.sectionTitle, entity.text).joinToString("\n"))
            val tf = HashMap<String, Int>()
            for (t in terms) tf.merge(t, 1, Int::plus)
            for (t in tf.keys) docFreq.merge(t, 1, Int::plus)
            IndexedChunk(
                // The BLOB is decoded once here; drop it from the cached entity to halve memory use.
                entity = entity.copy(embedding = null),
                embedding = FloatArrayConverter.toFloatArray(entity.embedding),
                termFreq = tf,
                length = terms.size
            )
        }
        return IndexedDocument(signature, indexed, docFreq)
    }


    // ---------------------------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------------------------

    private fun pageRange(chunk: ChunkEntity): Pair<Int, Int> {
        val parts = chunk.pageSpan.split("-").mapNotNull { it.trim().toIntOrNull() }
        return when (parts.size) {
            0 -> chunk.pageNumber to chunk.pageNumber
            1 -> parts[0] to parts[0]
            else -> parts[0] to parts[1]
        }
    }

    private fun cosine(a: FloatArray, b: FloatArray): Float {
        if (a.size != b.size) return -1f // vectors from a different embedding model are never comparable
        var dot = 0.0
        var na = 0.0
        var nb = 0.0
        for (i in a.indices) {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        val denom = kotlin.math.sqrt(na) * kotlin.math.sqrt(nb)
        return if (denom == 0.0) 0f else (dot / denom).toFloat()
    }

    private fun logTop(results: List<RetrievalResult>, query: String) {
        runCatching {
            Log.d(TAG, "Top results for \"$query\":")
            results.take(5).forEachIndexed { rank, r ->
                Log.d(TAG, "  #${rank + 1} chunk=${r.chunk.id} page=${r.chunk.pageSpan} rrf=%.4f dense=%.3f bm25=%.2f".format(r.relevanceScore, r.denseScore, r.bm25Score))
            }
        }
    }
}
