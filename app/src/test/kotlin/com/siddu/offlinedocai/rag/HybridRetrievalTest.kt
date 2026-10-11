package com.siddu.offlinedocai.rag

import com.siddu.offlinedocai.ai.retrieval.QueryClassification
import com.siddu.offlinedocai.ai.retrieval.QueryIntent
import com.siddu.offlinedocai.ai.retrieval.QueryIntentRouter
import com.siddu.offlinedocai.ai.retrieval.RetrievalSource
import com.siddu.offlinedocai.ai.retrieval.SearchQuery
import com.siddu.offlinedocai.ai.retrieval.TextAnalyzer
import com.siddu.offlinedocai.ai.retrieval.VectorStore
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Retrieval behaviour across unrelated domains. Embeddings are hand-built "topic" vectors so the
 * tests can express semantic similarity that has no word overlap.
 */
class HybridRetrievalTest {

    // Topic dimensions for the synthetic embedding space.
    private val REFUNDS = 0
    private val SHIPPING = 1
    private val API_AUTH = 2
    private val COOKING = 3
    private val INTEREST = 4

    private fun store(): Pair<VectorStore, FakeChunkDao> {
        val dao = FakeChunkDao()
        runBlocking {
            dao.insertAll(
                listOf(
                    // Document 1: a retail policy
                    chunk(1, 1, 0, "Customers may return unused items within 30 days for a full refund to the original payment method.",
                        TestVectors.topic(REFUNDS to 1f)),
                    chunk(2, 1, 1, "Orders ship within two business days. Express delivery arrives the next day for an extra fee.",
                        TestVectors.topic(SHIPPING to 1f)),
                    chunk(3, 1, 2, "Our estimates are accurate and we keep separate generator logs for every warehouse.",
                        TestVectors.topic(SHIPPING to 0.3f, COOKING to 0.2f)),
                    // Document 2: a software manual
                    chunk(10, 2, 0, "Every request must include a bearer token in the Authorization header.",
                        TestVectors.topic(API_AUTH to 1f)),
                    chunk(11, 2, 1, "Tokens expire after one hour; call the refresh endpoint to obtain a new one.",
                        TestVectors.topic(API_AUTH to 0.9f, SHIPPING to 0.1f)),
                    chunk(12, 2, 2, "The database stores request logs for ninety days.",
                        TestVectors.topic(API_AUTH to 0.3f)),
                    // Document 3: a recipe book and a savings brochure
                    chunk(20, 3, 0, "Simmer the tomatoes with garlic for twenty minutes, then season to taste.",
                        TestVectors.topic(COOKING to 1f)),
                    chunk(21, 3, 1, "The savings account pays an annual interest rate of 4.1 percent.",
                        TestVectors.topic(INTEREST to 1f))
                )
            )
        }
        return VectorStore(dao) to dao
    }

    private fun classify(q: String) = QueryIntentRouter.route(q)

    @Test
    fun paraphrasedQuestionWithNoSharedWordsIsAnswered() = runBlocking {
        val (store, _) = store()
        // "How do I get my money back?" shares no content word with chunk 1, but is semantically close.
        val q = "How do I get my money back?"
        val results = store.retrieve(SearchQuery(q, TestVectors.topic(REFUNDS to 1f, SHIPPING to 0.1f)), listOf(1L, 2L, 3L), classify(q))
        assertEquals(1L, results.first().chunk.id)
    }

    @Test
    fun exactTermsRankHighlyThroughBm25() = runBlocking {
        val (store, _) = store()
        val q = "When does the refresh endpoint token expire?"
        // A deliberately uninformative embedding: only lexical evidence can find the answer.
        val results = store.retrieve(SearchQuery(q, TestVectors.topic(COOKING to 0.05f, INTEREST to 1f)), listOf(2L), classify(q))
        assertEquals(11L, results.first().chunk.id)
    }

    @Test
    fun misspelledTermsStillMatchThroughBm25() = runBlocking {
        val (store, _) = store()
        val q = "endpiont"
        // Uninformative embedding: only the typo-corrected keyword ("endpoint") can find chunk 11.
        val results = store.retrieve(SearchQuery(q, TestVectors.topic(250 to 1f)), listOf(2L), QueryClassification(QueryIntent.GENERAL, q))
        assertEquals(11L, results.first().chunk.id)
    }

    @Test
    fun typoMatchingIsConservative() {
        assertTrue(TextAnalyzer.isTypoOf("reciev", "receiv"))   // transposition
        assertTrue(TextAnalyzer.isTypoOf("summeris", "summaris"))
        assertFalse(TextAnalyzer.isTypoOf("rate", "late"))      // too short to guess
        assertFalse(TextAnalyzer.isTypoOf("model", "hotel"))    // different first letter
        assertFalse(TextAnalyzer.isTypoOf("budget", "bucket")) // two edits on a short word
        assertFalse(TextAnalyzer.isTypoOf("2023", "2024"))      // numbers are exact
    }

    @Test
    fun lexicalMatchingIsNotSubstringMatching() = runBlocking {
        val (store, _) = store()
        val q = "rate"
        val results = store.retrieve(SearchQuery(q, TestVectors.topic(250 to 1f)), listOf(1L, 3L), QueryClassification(QueryIntent.GENERAL, q))
        // "accurate" and "separate" (chunk 3) must not match "rate"; the interest-rate chunk must.
        assertEquals(listOf(21L), results.filter { it.source == RetrievalSource.HYBRID }.map { it.chunk.id })
    }

    @Test
    fun unrelatedQuestionReturnsNothing() = runBlocking {
        val (store, _) = store()
        val q = "Who painted the ceiling of the chapel?"
        val results = store.retrieve(SearchQuery(q, TestVectors.topic(300 to 1f)), listOf(1L, 2L, 3L), classify(q))
        assertTrue(results.isEmpty())
    }

    @Test
    fun retrievalIsScopedToTheSelectedDocuments() = runBlocking {
        val (store, _) = store()
        val q = "token expiry"
        val results = store.retrieve(SearchQuery(q, TestVectors.topic(API_AUTH to 1f)), listOf(1L), classify(q))
        assertTrue(results.all { it.chunk.documentId == 1L })
    }

    @Test
    fun neighboursOfTopHitsAreAppendedAfterPrimaryResults() = runBlocking {
        val (store, _) = store()
        val q = "How long are request logs kept in the database?"
        val results = store.retrieve(SearchQuery(q, TestVectors.topic(API_AUTH to 0.3f)), listOf(2L), classify(q))
        val firstNeighbor = results.indexOfFirst { it.source == RetrievalSource.NEIGHBOR }
        val lastPrimary = results.indexOfLast { it.source == RetrievalSource.HYBRID }
        assertTrue(firstNeighbor == -1 || firstNeighbor > lastPrimary)
    }

    @Test
    fun comparisonFusesBothSides() = runBlocking {
        val (store, _) = store()
        val q = "Compare standard shipping and express delivery"
        val cls = classify(q)
        assertEquals(QueryIntent.COMPARISON, cls.intent)
        val subs = cls.subQueries.map { SearchQuery(it, TestVectors.topic(SHIPPING to 1f)) }
        val results = store.retrieve(SearchQuery(q, TestVectors.topic(SHIPPING to 1f)), listOf(1L), cls, subs)
        assertEquals(2L, results.first().chunk.id)
    }

    @Test
    fun pageLookupUsesPageSpans() = runBlocking {
        val dao = FakeChunkDao()
        dao.insertAll(
            listOf(
                chunk(1, 7, 0, "Intro text.", TestVectors.topic(0 to 1f), page = 1),
                chunk(2, 7, 1, "Text that starts on page 2 and continues onto page 3.", TestVectors.topic(0 to 1f), page = 2, pageSpan = "2-3"),
                chunk(3, 7, 2, "Page four text.", TestVectors.topic(0 to 1f), page = 4)
            )
        )
        val q = "What does page 3 say?"
        val cls = classify(q)
        assertEquals(QueryIntent.PAGE_LOOKUP, cls.intent)
        val results = VectorStore(dao).retrieve(SearchQuery(q, TestVectors.topic(0 to 1f)), listOf(7L), cls)
        assertEquals(listOf(2L), results.map { it.chunk.id })
    }

    @Test
    fun summarySamplingCoversBeginningEndAndMiddleFirst() = runBlocking {
        val dao = FakeChunkDao()
        dao.insertAll((0 until 40).map { chunk(it + 1L, 9, it, "Part $it of the report.", TestVectors.topic(0 to 1f)) })
        val q = "Summarize this document"
        val cls = classify(q)
        assertEquals(QueryIntent.SUMMARY, cls.intent)
        val indices = VectorStore(dao).retrieve(SearchQuery(q, TestVectors.topic(0 to 1f)), listOf(9L), cls).map { it.chunk.chunkIndex }
        assertEquals(listOf(0, 39, 19), indices.take(3))
        assertEquals(indices.size, indices.distinct().size)
    }

    @Test
    fun cacheIsRefreshedWhenADocumentIsReindexed() = runBlocking {
        val (store, dao) = store()
        val q = "refund"
        store.retrieve(SearchQuery(q, TestVectors.topic(REFUNDS to 1f)), listOf(1L), classify(q))
        dao.deleteChunksForDocument(1L)
        dao.insertAll(listOf(chunk(0, 1, 0, "Refunds are no longer offered; store credit only.", TestVectors.topic(REFUNDS to 1f))))
        val results = store.retrieve(SearchQuery(q, TestVectors.topic(REFUNDS to 1f)), listOf(1L), classify(q))
        assertEquals("Refunds are no longer offered; store credit only.", results.first().chunk.text)
        assertFalse(results.any { it.chunk.id == 1L })
    }

    @Test
    fun vectorsFromADifferentModelAreNeverCompared() = runBlocking {
        val dao = FakeChunkDao()
        dao.insertAll(listOf(chunk(1, 5, 0, "Completely unrelated words here.", FloatArray(768) { 1f })))
        val q = "anything"
        val results = VectorStore(dao).retrieve(SearchQuery(q, TestVectors.topic(0 to 1f)), listOf(5L), QueryClassification(QueryIntent.GENERAL, q))
        assertTrue(results.isEmpty())
    }
}
