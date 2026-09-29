package com.siddu.offlinedocai.ai.retrieval

import android.util.Log
import com.siddu.offlinedocai.data.FloatArrayConverter
import com.siddu.offlinedocai.data.dao.ChunkDao
import com.siddu.offlinedocai.data.entities.ChunkEntity
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Provenance tag indicating the retrieval strategy that selected a chunk.
 */
enum class RetrievalSource {
    PAGE_DETERMINISTIC,
    PAGE_RANGE_DETERMINISTIC,
    EXACT_FACT_LEXICAL,
    ENTITY_LOOKUP,
    SECTION_METADATA,
    EXHAUSTIVE_SCAN,
    FIND_ALL_SCAN,
    COMPARISON_BALANCED,
    GLOBAL_SUMMARY_STRUCTURAL,
    HYBRID_SEMANTIC,
    PARENT_EXPANSION
}

/**
 * Rich retrieval result model containing chunk entity, score, provenance, and intent.
 */
data class RetrievalResult(
    val chunk: ChunkEntity,
    val relevanceScore: Float,
    val source: RetrievalSource,
    val intent: QueryIntent
)

/**
 * Intent-aware hybrid retriever with relevance validation, multi-signal cross-reranking,
 * exhaustive scanning, and neighbor context expansion.
 */
class VectorStore(private val chunkDao: ChunkDao) {

    private val TAG = "VectorStore"

    companion object {
        private val GENERIC_ENTITY_REGEX = Regex("""\b[A-Z][a-z0-9]+(?:\s+[A-Z0-9][a-z0-9\-]+)+\b|\b[A-Z]{2,10}(?:[\-_]\d+)?\b|\b[A-Z][a-zA-Z0-9]*\s+\d+(?:\.\d+)?\b""")
        private val COMMA_LIST_REGEX = Regex("""\b[A-Z][a-zA-Z0-9_\-\.\+]{1,25}\b(?:\s*,\s*\b[A-Z][a-zA-Z0-9_\-\.\+]{1,25}\b){2,}""")
        private val LIST_EVIDENCE_REGEX = Regex("""(?:\b[A-Z][a-zA-Z0-9_\-]{2,20}\b\s*[:\-]|•|\*|\b\d+[\.\)])""")
        private val ANY_METRIC_REGEX = Regex("""(?i)\b\d+(?:\.\d+)?\s*(?:mah|ah|wh|kwh|w|kw|mw|v|kv|mv|gb|mb|tb|kb|hz|mhz|ghz|c|°c|f|°f|k|kg|g|mg|lbs|km|m|cm|mm|nm|mph|km/h|ms|s|min|hrs|v|a|ma|rpm|db|dpi|px|%)\b""")
        private val WORD_SPLIT_REGEX = Regex("[^a-zA-Z0-9]+")
        private val NON_ALPHANUMERIC_REGEX = Regex("""[^a-zA-Z0-9\s]""")
        private val MULTI_SPACE_REGEX = Regex("""\s+""")
        private val COMPOUND_IDENTIFIER_PATTERN = Regex("""(?i)\b(pre[- ]phase|phase|stage|step|chapter|section|appendix|type|level|tier)\s+([a-zA-Z0-9]+)\b""")
        private val COMPARISON_REGEX = Regex("""(?i)\b(?:compare|difference\s+between)\s+(.+?)\s+(?:and|with|to|vs|versus)\s+(.+?)(?:\s+in\s+the\s+document|\?|$|\.)|\bhow\s+does\s+(.+?)\s+differ\s+from\s+(.+?)(?:\?|$|\.)|^\s*(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\?|$|\.)""")
    }

    private val stopWords = setOf(
        "what", "when", "where", "how", "this", "that", "there", "their", "who", "why",
        "whom", "whose", "which", "them", "with", "from", "the", "is", "are", "was", "were",
        "and", "or", "in", "on", "at", "to", "a", "an", "of", "for", "by", "about", "give",
        "tell", "explain", "show", "me", "some", "can", "you", "does", "do", "did", "define",
        "list", "name", "mention", "describe", "find", "any", "all", "please", "these", "those",
        "be", "been", "being", "have", "has", "had", "would", "should", "could"
    )

    private val dimensionUnitMap = mapOf(
        setOf("capacity", "charge", "charging", "power", "wattage", "energy", "voltage", "current", "battery") to
            Regex("""(?i)\b\d+(?:\.\d+)?\s*(?:mah|ah|wh|kwh|w|kw|mw|v|kv|mv|a|ma)\b"""),
        setOf("temperature", "temp", "thermal", "heat", "hot", "cold") to
            Regex("""(?i)\b\d+(?:\.\d+)?\s*(?:°c|c|°f|f|k|deg|degrees)\b"""),
        setOf("storage", "memory", "ram", "rom", "disk", "capacity", "space") to
            Regex("""(?i)\b\d+(?:\.\d+)?\s*(?:gb|mb|tb|kb|pb|bytes)\b"""),
        setOf("speed", "frequency", "clock", "rate", "latency") to
            Regex("""(?i)\b\d+(?:\.\d+)?\s*(?:ghz|mhz|khz|hz|fps|rpm|mph|km/h|ms|ns|s)\b"""),
        setOf("weight", "mass", "heavy", "light") to
            Regex("""(?i)\b\d+(?:\.\d+)?\s*(?:kg|g|mg|lbs|oz|ton|tons)\b"""),
        setOf("dimension", "dimensions", "size", "width", "height", "length", "distance", "resolution") to
            Regex("""(?i)\b\d+(?:\.\d+)?\s*(?:mm|cm|m|km|nm|inches|px|dpi)\b""")
    )

    enum class RelevanceTier {
        TIER_A_DIRECT_ANSWER,      // High confidence direct answer evidence
        TIER_B_SUPPORTING_CONTEXT, // Strongly relevant supporting context
        TIER_C_BOILERPLATE_WEAK    // Weakly related or generic administrative boilerplate
    }

    data class CandidateScore(
        val chunk: ChunkEntity,
        val initialScore: Float,
        val denseScore: Float = 0f,
        var bm25Score: Float = 0f,
        var exactDefScore: Float = 0f,
        var rrfScore: Float = 0f,
        var rerankScore: Float = 0f,
        var tier: RelevanceTier = RelevanceTier.TIER_B_SUPPORTING_CONTEXT,
        var rejectionReason: String? = null,
        var survived: Boolean = true
    )

    /**
     * Primary retrieval entry point returning [ChunkEntity] list.
     * Retains 100% backward compatibility with existing callers and tests.
     */
    suspend fun retrieve(
        queryEmbedding: FloatArray,
        relevantDocumentIds: List<Long>,
        topK: Int,
        query: String = ""
    ): List<ChunkEntity> {
        return retrieveDetailed(queryEmbedding, relevantDocumentIds, topK, query).map { it.chunk }
    }

    /**
     * Rich retrieval entry point returning [RetrievalResult] list with scores, provenance, and intent metadata.
     */
    suspend fun retrieveDetailed(
        queryEmbedding: FloatArray,
        relevantDocumentIds: List<Long>,
        topK: Int,
        query: String = ""
    ): List<RetrievalResult> {
        if (relevantDocumentIds.isEmpty()) return emptyList()

        val chunks = chunkDao.getAllChunksForDocumentIds(relevantDocumentIds)
        if (chunks.isEmpty()) return emptyList()

        val classification = QueryIntentRouter.route(query)
        logInfo(TAG, "[RETRIEVAL] Query: \"$query\" -> Intent: ${classification.intent}")

        val results: List<RetrievalResult> = when (classification.intent) {
            QueryIntent.PAGE_LOOKUP -> retrievePageLookup(chunks, classification)
            QueryIntent.PAGE_RANGE -> retrievePageRange(chunks, classification)
            QueryIntent.EXACT_FACT -> retrieveExactFact(chunks, queryEmbedding, query, classification, topK)
            QueryIntent.ENTITY_LOOKUP -> retrieveEntityLookup(chunks, queryEmbedding, query, classification, topK)
            QueryIntent.SECTION_QUERY -> retrieveSectionQuery(chunks, queryEmbedding, query, classification, topK)
            QueryIntent.EXHAUSTIVE_LIST -> retrieveExhaustiveList(chunks, queryEmbedding, query, classification)
            QueryIntent.FIND_ALL -> retrieveFindAll(chunks, queryEmbedding, query, classification)
            QueryIntent.COMPARISON -> retrieveComparison(chunks, queryEmbedding, query, classification, topK)
            QueryIntent.GLOBAL_SUMMARY -> retrieveGlobalSummary(chunks, topK)
            QueryIntent.CONCEPTUAL_QUERY -> retrieveConceptual(chunks, queryEmbedding, query, topK, classification)
        }

        return results
    }

    // -------------------------------------------------------------------------
    // 1. PAGE_LOOKUP (Deterministic Database / Chunk Retrieval)
    // -------------------------------------------------------------------------
    private fun retrievePageLookup(chunks: List<ChunkEntity>, classification: QueryClassification): List<RetrievalResult> {
        val targetPage = classification.pageStart ?: 1
        val pageChunks = chunks.filter { it.pageNumber == targetPage }
            .sortedWith(compareBy({ it.chunkIndex }, { it.id }))

        logInfo(TAG, "[PAGE_LOOKUP] Page $targetPage: retrieved ${pageChunks.size} chunks")
        return pageChunks.map {
            RetrievalResult(chunk = it, relevanceScore = 100.0f, source = RetrievalSource.PAGE_DETERMINISTIC, intent = QueryIntent.PAGE_LOOKUP)
        }
    }

    // -------------------------------------------------------------------------
    // 2. PAGE_RANGE (Deterministic Page Range Retrieval)
    // -------------------------------------------------------------------------
    private fun retrievePageRange(chunks: List<ChunkEntity>, classification: QueryClassification): List<RetrievalResult> {
        val start = classification.pageStart ?: 1
        val end = classification.pageEnd ?: start
        val rangeChunks = chunks.filter { it.pageNumber in start..end }
            .sortedWith(compareBy({ it.pageNumber }, { it.chunkIndex }, { it.id }))

        logInfo(TAG, "[PAGE_RANGE] Pages $start..$end: retrieved ${rangeChunks.size} chunks")
        return rangeChunks.map {
            RetrievalResult(chunk = it, relevanceScore = 100.0f, source = RetrievalSource.PAGE_RANGE_DETERMINISTIC, intent = QueryIntent.PAGE_RANGE)
        }
    }

    // -------------------------------------------------------------------------
    // 3. EXACT_FACT (Fact Pattern Scanning & Lexical Prioritization)
    // -------------------------------------------------------------------------
    private fun retrieveExactFact(
        chunks: List<ChunkEntity>,
        queryEmbedding: FloatArray,
        query: String,
        classification: QueryClassification,
        topK: Int
    ): List<RetrievalResult> {
        val factType = classification.factType ?: FactType.EXACT_VALUE
        val candidates = scoreCandidates(chunks, queryEmbedding, query)
        val factRegex = getFactTypeRegex(factType)

        val factMatchingCandidates = mutableListOf<CandidateScore>()

        for (cand in candidates) {
            val text = cand.chunk.text
            val matches = if (factRegex != null) factRegex.findAll(text).toList() else emptyList()
            val hasValidFactMatch = when (factType) {
                FactType.PHONE -> matches.any { m -> m.value.count { it.isDigit() } in 7..15 }
                FactType.EMAIL -> matches.isNotEmpty()
                FactType.URL -> matches.isNotEmpty()
                FactType.DATE -> matches.isNotEmpty()
                FactType.ID -> matches.isNotEmpty()
                FactType.PERCENTAGE -> matches.isNotEmpty()
                FactType.AMOUNT -> matches.isNotEmpty()
                FactType.NUMBER -> matches.isNotEmpty()
                FactType.TEMPERATURE -> matches.isNotEmpty()
                FactType.EXACT_VALUE -> {
                    val kw = classification.targetTerm ?: cleanQueryText(query)
                    kw.isNotEmpty() && text.contains(kw, ignoreCase = true)
                }
            }

            if (hasValidFactMatch && !isBoilerplateText(text)) {
                cand.rerankScore += 100.0f
                cand.tier = RelevanceTier.TIER_A_DIRECT_ANSWER
                factMatchingCandidates.add(cand)
            }
        }

        // If fact-matching candidates exist, return them ranked by rerank score
        if (factMatchingCandidates.isNotEmpty()) {
            val sorted = factMatchingCandidates.sortedByDescending { it.rerankScore }.take(topK)
            logInfo(TAG, "[EXACT_FACT] Found ${factMatchingCandidates.size} matches for $factType, returning ${sorted.size}")
            return sorted.map {
                RetrievalResult(chunk = it.chunk, relevanceScore = it.rerankScore, source = RetrievalSource.EXACT_FACT_LEXICAL, intent = QueryIntent.EXACT_FACT)
            }
        }

        // Negative check: If NO candidate matches the sought exact fact, return emptyList()
        // so that non-existent facts do not falsely return unrelated chunks.
        logWarn(TAG, "[EXACT_FACT] No valid evidence found for $factType in document chunks")
        return emptyList()
    }

    private fun getFactTypeRegex(factType: FactType): Regex? {
        return when (factType) {
            FactType.PHONE -> Regex("""(?:\+\d{1,3}[\s-]*)?(?:\(?\d{2,4}\)?[\s-]*)?\d{3,4}[\s-]*\d{3,4}(?:[\s-]*\d{1,4})?\b|\b\d{10}\b""")
            FactType.EMAIL -> Regex("""\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b""")
            FactType.URL -> Regex("""(?:https?://|www\.)[^\s]+|github\.com/[^\s]+|linkedin\.com/[^\s]+""")
            FactType.DATE -> Regex("""\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b""")
            FactType.ID -> Regex("""(?i)\b(?:id|employee\s*id|roll\s*(?:no|number)|serial\s*(?:no|number)|registration\s*(?:no|number)|account\s*no)[:\s#]*([A-Z0-9\-_]{3,})\b|\b[A-Z]{2,}\d{3,}\b|\bID[:\s]+[A-Z0-9\-_]+\b""")
            FactType.PERCENTAGE -> Regex("""\b\d+(?:\.\d+)?%|\b\d+(?:\.\d+)?\s+percent\b""", RegexOption.IGNORE_CASE)
            FactType.AMOUNT -> Regex("""(?i)(?:[\$€£₹]\s*\d+(?:,\d+)*(?:\.\d+)?|\b\d+(?:,\d+)*(?:\.\d+)?\s*(?:usd|eur|inr|dollars|k|lakh|crore|million|billion)\b)""")
            FactType.NUMBER -> Regex("""\b\d+(?:\.\d+)?\b""")
            FactType.TEMPERATURE -> Regex("""(?i)\b\d+(?:\.\d+)?\s*(?:°c|c|°f|f|k)\b""")
            FactType.EXACT_VALUE -> null
        }
    }

    // -------------------------------------------------------------------------
    // 4. ENTITY_LOOKUP (Multi-Location Entity Retrieval)
    // -------------------------------------------------------------------------
    private fun retrieveEntityLookup(
        chunks: List<ChunkEntity>,
        queryEmbedding: FloatArray,
        query: String,
        classification: QueryClassification,
        topK: Int
    ): List<RetrievalResult> {
        val targetEntity = classification.targetTerm ?: cleanQueryText(query)
        val candidates = scoreCandidates(chunks, queryEmbedding, query)
        val defRegex = Regex("""(?i)\b${Regex.escape(targetEntity)}\b\s+(?:is|was|refers\s+to|serves\s+as|developed\s+by|features)\b""")

        for (cand in candidates) {
            val text = cand.chunk.text
            val hasExactCase = text.contains(targetEntity)
            val hasIgnoreCase = text.contains(targetEntity, ignoreCase = true)
            if (hasIgnoreCase && !isBoilerplateText(text)) {
                cand.rerankScore += if (hasExactCase) 50.0f else 35.0f
                if (defRegex.containsMatchIn(text)) {
                    cand.rerankScore += 30.0f
                }
                cand.tier = RelevanceTier.TIER_A_DIRECT_ANSWER
            }
        }

        // Multi-page distribution: Ensure chunks from distinct pages are included
        val validMatches = candidates.filter {
            it.chunk.text.contains(targetEntity, ignoreCase = true) && !isBoilerplateText(it.chunk.text)
        }

        val selected = mutableListOf<CandidateScore>()
        val seenPages = mutableSetOf<Int>()

        // Take top candidate per distinct page
        val byPage = validMatches.groupBy { it.chunk.pageNumber }
        for ((page, pageCandidates) in byPage) {
            val bestOnPage = pageCandidates.maxByOrNull { it.rerankScore }
            if (bestOnPage != null && seenPages.add(page)) {
                selected.add(bestOnPage)
            }
        }

        // If slots remain, fill with highest remaining candidates
        val remaining = candidates.filter { it !in selected && it.tier != RelevanceTier.TIER_C_BOILERPLATE_WEAK && it.rerankScore >= 5.0f }
            .sortedByDescending { it.rerankScore }
        for (cand in remaining) {
            if (selected.size >= topK) break
            selected.add(cand)
        }

        val sorted = selected.sortedByDescending { it.rerankScore }
        val mapped = sorted.map {
            RetrievalResult(chunk = it.chunk, relevanceScore = it.rerankScore, source = RetrievalSource.ENTITY_LOOKUP, intent = QueryIntent.ENTITY_LOOKUP)
        }

        return expandNeighbors(mapped, chunks)
    }

    // -------------------------------------------------------------------------
    // 5. SECTION_QUERY (SectionTitle Metadata & Heading Anchors)
    // -------------------------------------------------------------------------
    private fun retrieveSectionQuery(
        chunks: List<ChunkEntity>,
        queryEmbedding: FloatArray,
        query: String,
        classification: QueryClassification,
        topK: Int
    ): List<RetrievalResult> {
        val targetSection = classification.targetTerm ?: cleanQueryText(query)

        // 1. Check sectionTitle metadata
        val metadataMatches = chunks.filter {
            it.sectionTitle != null && it.sectionTitle.contains(targetSection, ignoreCase = true)
        }.sortedWith(compareBy({ it.chunkIndex }, { it.id }))

        if (metadataMatches.isNotEmpty()) {
            logInfo(TAG, "[SECTION_QUERY] Found ${metadataMatches.size} chunks matching sectionTitle '$targetSection'")
            val mapped = metadataMatches.map {
                RetrievalResult(chunk = it, relevanceScore = 80.0f, source = RetrievalSource.SECTION_METADATA, intent = QueryIntent.SECTION_QUERY)
            }
            return expandNeighbors(mapped, chunks)
        }

        // 2. Heading detection in chunk text
        val headingRegex = Regex("""(?im)^(?:#+\s*|section\s+[\d\.]+:?\s*|chapter\s+[\d\.]+:?\s*)?""" + Regex.escape(targetSection) + """[:\s\n]""")
        val headingMatches = chunks.filter { headingRegex.containsMatchIn(it.text) && !isBoilerplateText(it.text) }
            .sortedWith(compareBy({ it.chunkIndex }, { it.id }))

        if (headingMatches.isNotEmpty()) {
            logInfo(TAG, "[SECTION_QUERY] Found ${headingMatches.size} chunks matching heading pattern for '$targetSection'")
            val mapped = headingMatches.map {
                RetrievalResult(chunk = it, relevanceScore = 75.0f, source = RetrievalSource.SECTION_METADATA, intent = QueryIntent.SECTION_QUERY)
            }
            return expandNeighbors(mapped, chunks)
        }

        // 3. Fallback to hybrid search
        logInfo(TAG, "[SECTION_QUERY] No explicit section heading found; falling back to hybrid retrieval")
        return retrieveConceptual(chunks, queryEmbedding, query, topK)
    }

    // -------------------------------------------------------------------------
    // 6. EXHAUSTIVE_LIST (Document-Wide Complete Candidate Collection)
    // -------------------------------------------------------------------------
    private fun retrieveExhaustiveList(
        chunks: List<ChunkEntity>,
        queryEmbedding: FloatArray,
        query: String,
        classification: QueryClassification
    ): List<RetrievalResult> {
        val category = classification.targetTerm ?: cleanQueryText(query)
        val singular = when {
            category.endsWith("ies") && category.length > 3 -> category.removeSuffix("ies") + "y"
            category.endsWith("s") && category.length > 2 -> category.removeSuffix("s")
            else -> category
        }
        val stems = getRootStems(category) + getRootStems(singular)

        val headingPattern = Regex("""(?i)\b(?:key|major|all|core|technical|list\s+of)?\s*(?:${Regex.escape(category)}|${Regex.escape(singular)})\s*[:\-]""")
        val listLeadPattern = Regex("""(?i)\b(?:${Regex.escape(category)}|${Regex.escape(singular)})\b[\s\w]*[:\-]\s*(?:•|\*|\d+[\.\)]|[A-Z])""")

        val collectedChunks = mutableListOf<RetrievalResult>()
        val seenIds = mutableSetOf<Long>()

        for (chunk in chunks) {
            if (isBoilerplateText(chunk.text)) continue

            val chunkWords = chunk.text.split(WORD_SPLIT_REGEX).filter { it.isNotEmpty() }

            val hasHeading = headingPattern.containsMatchIn(chunk.text)
            val hasListLead = listLeadPattern.containsMatchIn(chunk.text)
            val hasStemMatch = stems.any { stem -> chunkWords.any { w -> wordMatchesKeyword(w, stem, setOf(stem)) } }
            val hasBulletList = LIST_EVIDENCE_REGEX.containsMatchIn(chunk.text)

            // Include chunk if it contains category heading, or category mention with list structure, or direct category items
            val isMatch = hasHeading || (hasStemMatch && hasBulletList) || (hasStemMatch && chunkWords.size < 60) || hasListLead

            if (isMatch && seenIds.add(chunk.id)) {
                val score = when {
                    hasHeading -> 90.0f
                    hasListLead -> 85.0f
                    hasStemMatch && hasBulletList -> 80.0f
                    else -> 70.0f
                }
                collectedChunks.add(
                    RetrievalResult(chunk = chunk, relevanceScore = score, source = RetrievalSource.EXHAUSTIVE_SCAN, intent = QueryIntent.EXHAUSTIVE_LIST)
                )
            }
        }

        // If no explicit list chunks found via scan, fall back to hybrid scoring to not return empty
        if (collectedChunks.isEmpty()) {
            logInfo(TAG, "[EXHAUSTIVE_LIST] Scan found 0 explicit list chunks; running hybrid scoring fallback")
            return retrieveConceptual(chunks, queryEmbedding, query, topK = 15)
        }

        // Sort by document sequence: pageNumber ASC, chunkIndex ASC, id ASC
        val sorted = collectedChunks.sortedWith(compareBy({ it.chunk.pageNumber }, { it.chunk.chunkIndex }, { it.chunk.id }))
        logInfo(TAG, "[EXHAUSTIVE_LIST] Collected ${sorted.size} exhaustive candidates across pages: ${sorted.map { it.chunk.pageNumber }.distinct()}")
        return sorted
    }

    // -------------------------------------------------------------------------
    // 7. FIND_ALL (Exhaustive Mention Scanning Across Document)
    // -------------------------------------------------------------------------
    private fun retrieveFindAll(
        chunks: List<ChunkEntity>,
        queryEmbedding: FloatArray,
        query: String,
        classification: QueryClassification
    ): List<RetrievalResult> {
        val target = classification.targetTerm ?: cleanQueryText(query)
        val targetLower = target.lowercase()
        val targetTokens = targetLower.split(Regex("[^a-zA-Z0-9]+")).filter { it.isNotEmpty() }
        val targetStems = targetTokens.flatMap { getRootStems(it) }.toSet()

        val collected = mutableListOf<RetrievalResult>()
        val seenIds = mutableSetOf<Long>()

        for (chunk in chunks) {
            if (isBoilerplateText(chunk.text)) continue

            val textLower = chunk.text.lowercase()
            val chunkWords = chunk.text.split(WORD_SPLIT_REGEX).filter { it.isNotEmpty() }

            val hasExactPhrase = textLower.contains(targetLower)
            val hasAllTokens = targetTokens.isNotEmpty() && targetTokens.all { tok ->
                chunkWords.any { w -> wordMatchesKeyword(w, tok, targetStems) }
            }

            // Supplemental dense similarity
            var denseScore = 0f
            if (chunk.embedding != null && queryEmbedding.isNotEmpty()) {
                val emb = FloatArrayConverter.toFloatArray(chunk.embedding)
                if (emb != null) {
                    denseScore = cosineSimilarity(queryEmbedding, emb)
                }
            }

            val isMatch = hasExactPhrase || hasAllTokens || denseScore >= 0.45f

            if (isMatch && seenIds.add(chunk.id)) {
                val score = when {
                    hasExactPhrase -> 95.0f
                    hasAllTokens -> 80.0f
                    else -> denseScore * 100f
                }
                collected.add(
                    RetrievalResult(chunk = chunk, relevanceScore = score, source = RetrievalSource.FIND_ALL_SCAN, intent = QueryIntent.FIND_ALL)
                )
            }
        }

        // Sort by document sequence: pageNumber ASC, chunkIndex ASC, id ASC
        val sorted = collected.sortedWith(compareBy({ it.chunk.pageNumber }, { it.chunk.chunkIndex }, { it.chunk.id }))
        logInfo(TAG, "[FIND_ALL] Found ${sorted.size} occurrences of '$target' across pages: ${sorted.map { it.chunk.pageNumber }}")
        return sorted
    }

    // -------------------------------------------------------------
    // 8. COMPARISON (Balanced Dual-Entity Retrieval)
    // -------------------------------------------------------------
    private fun retrieveComparison(
        chunks: List<ChunkEntity>,
        queryEmbedding: FloatArray,
        query: String,
        classification: QueryClassification,
        topK: Int
    ): List<RetrievalResult> {
        val entityA = classification.entityA ?: ""
        val entityB = classification.entityB ?: ""
        val phraseA = entityA.trim()
        val phraseB = entityB.trim()

        val candidates = scoreCandidates(chunks, queryEmbedding, query)
            .filter { it.tier != RelevanceTier.TIER_C_BOILERPLATE_WEAK && it.rerankScore >= 5.0f }

        val regexA = if (phraseA.isNotEmpty()) Regex("""(?i)\b${Regex.escape(phraseA)}\b""") else null
        val regexB = if (phraseB.isNotEmpty()) Regex("""(?i)\b${Regex.escape(phraseB)}\b""") else null

        val candJoint = mutableListOf<CandidateScore>()
        val candA = mutableListOf<CandidateScore>()
        val candB = mutableListOf<CandidateScore>()

        for (c in candidates) {
            val matchA = regexA != null && regexA.containsMatchIn(c.chunk.text)
            val matchB = regexB != null && regexB.containsMatchIn(c.chunk.text)

            if (matchA && matchB) {
                candJoint.add(c)
            } else if (matchA) {
                candA.add(c)
            } else if (matchB) {
                candB.add(c)
            }
        }

        val merged = mutableListOf<CandidateScore>()
        val halfK = maxOf(1, topK / 2)

        for (c in candJoint) {
            if (c !in merged && merged.size < topK) merged.add(c)
        }
        for (c in candA.take(halfK)) {
            if (c !in merged && merged.size < topK) merged.add(c)
        }
        for (c in candB.take(halfK)) {
            if (c !in merged && merged.size < topK) merged.add(c)
        }
        for (c in candidates) {
            if (c !in merged && merged.size < topK) merged.add(c)
        }

        logInfo(TAG, "[COMPARISON] Retrieved ${merged.size} candidates (Joint=${candJoint.size}, A=${candA.size}, B=${candB.size})")
        return merged.map {
            RetrievalResult(chunk = it.chunk, relevanceScore = it.rerankScore, source = RetrievalSource.COMPARISON_BALANCED, intent = QueryIntent.COMPARISON)
        }
    }

    // -------------------------------------------------------------------------
    // 9. GLOBAL_SUMMARY (Representative Structural Sampling)
    // -------------------------------------------------------------------------
    private fun retrieveGlobalSummary(chunks: List<ChunkEntity>, topK: Int): List<RetrievalResult> {
        val nonBoilerplate = chunks.filter { !isBoilerplateText(it.text) }
        val sourceChunks = if (nonBoilerplate.isNotEmpty()) nonBoilerplate else chunks

        if (sourceChunks.size <= topK) {
            return sourceChunks.map {
                RetrievalResult(chunk = it, relevanceScore = 50.0f, source = RetrievalSource.GLOBAL_SUMMARY_STRUCTURAL, intent = QueryIntent.GLOBAL_SUMMARY)
            }
        }

        val selected = mutableListOf<ChunkEntity>()
        val seenIds = mutableSetOf<Long>()

        // 1. Introduction / Preamble (first chunk)
        sourceChunks.firstOrNull()?.let {
            if (seenIds.add(it.id)) selected.add(it)
        }

        // 2. Chunks with section titles / headings
        val headingChunks = sourceChunks.filter {
            it.sectionTitle != null || it.text.startsWith("#") || LIST_EVIDENCE_REGEX.containsMatchIn(it.text)
        }
        for (hc in headingChunks) {
            if (selected.size >= topK - 1) break
            if (seenIds.add(hc.id)) selected.add(hc)
        }

        // 3. Stratified sampling across document length
        val step = sourceChunks.size.toDouble() / topK
        for (i in 0 until topK) {
            if (selected.size >= topK - 1) break
            val idx = (i * step).toInt().coerceIn(0, sourceChunks.size - 1)
            val chunk = sourceChunks[idx]
            if (seenIds.add(chunk.id)) selected.add(chunk)
        }

        // 4. Conclusion / Ending chunk
        sourceChunks.lastOrNull()?.let {
            if (seenIds.add(it.id)) selected.add(it)
        }

        val sorted = selected.sortedWith(compareBy({ it.pageNumber }, { it.chunkIndex }, { it.id })).take(topK)
        logInfo(TAG, "[GLOBAL_SUMMARY] Selected ${sorted.size} structural summary chunks")
        return sorted.map {
            RetrievalResult(chunk = it, relevanceScore = 50.0f, source = RetrievalSource.GLOBAL_SUMMARY_STRUCTURAL, intent = QueryIntent.GLOBAL_SUMMARY)
        }
    }

    // -------------------------------------------------------------------------
    // 10. CONCEPTUAL_QUERY (Two-Stage Hybrid Search + Cross-Reranker)
    // -------------------------------------------------------------------------
    private fun retrieveConceptual(
        chunks: List<ChunkEntity>,
        queryEmbedding: FloatArray,
        query: String,
        topK: Int,
        classification: QueryClassification? = null
    ): List<RetrievalResult> {
        val candidates = scoreCandidates(chunks, queryEmbedding, query, classification)
        val relevant = candidates.filter { it.survived }.sortedByDescending { it.rerankScore }

        if (relevant.isEmpty()) {
            logWarn(TAG, "No relevant candidate passed evidence threshold for conceptual query: '$query'")
            return emptyList()
        }

        val top = relevant.first()
        // Retrieval sanity check: if the top candidate has weak relevance and zero concept match, reject
        if (top.rerankScore < 15.0f && top.denseScore < 0.28f && top.exactDefScore <= 0f) {
            logWarn(TAG, "[SANITY_CHECK] Top candidate failed relevance validation: score=${top.rerankScore}, dense=${top.denseScore}, def=${top.exactDefScore}. Query: \"$query\"")
            return emptyList()
        }

        val isDef = classification?.isDefinition == true || (classification == null && isDefinitionQueryText(query))
        val selected = if (isDef) {
            val defMatches = relevant.filter { it.exactDefScore > 20.0f }
            if (defMatches.isNotEmpty()) {
                val nonDef = relevant.filter { it !in defMatches }
                (defMatches + nonDef).take(topK)
            } else {
                relevant.take(topK)
            }
        } else {
            // Strategy A: Comparison fallback (if comparison syntax present in conceptual fallback)
            val compMatch = COMPARISON_REGEX.find(query)
            if (compMatch != null) {
                val nonEmpties = compMatch.groupValues.drop(1).filter { it.isNotBlank() }
                if (nonEmpties.size >= 2) {
                    val wordsA = nonEmpties[0].split(Regex("[^a-zA-Z0-9]+")).filter { it.isNotEmpty() }
                    val wordsB = nonEmpties[1].split(Regex("[^a-zA-Z0-9]+")).filter { it.isNotEmpty() }
                    val candA = relevant.filter { c -> wordsA.any { c.chunk.text.contains(it, ignoreCase = true) } }
                    val candB = relevant.filter { c -> wordsB.any { c.chunk.text.contains(it, ignoreCase = true) } }
                    val merged = mutableListOf<ChunkEntity>()
                    val halfK = maxOf(1, topK / 2)
                    for (c in candA.take(halfK)) if (c.chunk !in merged) merged.add(c.chunk)
                    for (c in candB.take(halfK)) if (c.chunk !in merged) merged.add(c.chunk)
                    for (c in relevant) if (c.chunk !in merged && merged.size < topK) merged.add(c.chunk)
                    if (merged.isNotEmpty()) {
                        val mapped = merged.map {
                            RetrievalResult(chunk = it, relevanceScore = 50.0f, source = RetrievalSource.HYBRID_SEMANTIC, intent = QueryIntent.CONCEPTUAL_QUERY)
                        }
                        return expandNeighbors(mapped, chunks, isDefinition = false)
                    }
                }
            }

            // Strategy B: Location / Page Diversity fallback
            val locationRegex = Regex("""(?i)\bwhere\s+(?:does|can|in|is|are|do)?\s*(?:the\s+book|the\s+document|we|i)?\s*(?:discuss|mention|talk\s+about|cover|find|address|describe)\b|\bwhere\s+in\s+the\s+book\b|\bwhich\s+(?:pages|sections|chapters)\s+(?:discuss|mention|cover)\b""")
            if (locationRegex.containsMatchIn(query)) {
                val pageSeen = mutableSetOf<Int>()
                val diverse = mutableListOf<CandidateScore>()
                for (c in relevant) {
                    if (pageSeen.add(c.chunk.pageNumber)) {
                        diverse.add(c)
                    }
                    if (diverse.size >= topK) break
                }
                if (diverse.isNotEmpty()) {
                    val mapped = diverse.map {
                        RetrievalResult(chunk = it.chunk, relevanceScore = it.rerankScore, source = RetrievalSource.HYBRID_SEMANTIC, intent = QueryIntent.CONCEPTUAL_QUERY)
                    }
                    return expandNeighbors(mapped, chunks, isDefinition = false)
                }
            }

            relevant.take(topK)
        }

        val mapped = selected.map {
            RetrievalResult(chunk = it.chunk, relevanceScore = it.rerankScore, source = RetrievalSource.HYBRID_SEMANTIC, intent = QueryIntent.CONCEPTUAL_QUERY)
        }

        return expandNeighbors(mapped, chunks, isDefinition = isDef)
    }

    // -------------------------------------------------------------------------
    // 11. PARENT / NEIGHBOR EXPANSION
    // -------------------------------------------------------------------------
    private fun expandNeighbors(
        primaryResults: List<RetrievalResult>,
        allChunks: List<ChunkEntity>,
        isDefinition: Boolean = false
    ): List<RetrievalResult> {
        if (primaryResults.isEmpty()) return emptyList()

        val chunkByIndex = allChunks.groupBy { it.documentId to it.chunkIndex }
        val expanded = mutableListOf<RetrievalResult>()
        val seenIds = primaryResults.map { it.chunk.id }.toMutableSet()

        for (res in primaryResults) {
            expanded.add(res)
        }

        // Expand context around the top 1-2 high-confidence primary matches (>= 20f score)
        val candidatesToExpand = primaryResults.take(2).filter { it.relevanceScore >= 20.0f }

        for (target in candidatesToExpand) {
            val docId = target.chunk.documentId
            val currIndex = target.chunk.chunkIndex

            for (offset in listOf(-1, 1)) {
                val neighborIndex = currIndex + offset
                val neighbors = chunkByIndex[docId to neighborIndex] ?: emptyList()
                for (neighbor in neighbors) {
                    if (neighbor.id in seenIds) continue

                    // Constraint 1: Same document
                    if (neighbor.documentId != target.chunk.documentId) continue

                    // Constraint 2: Adjacent page (within 1 page)
                    if (abs(neighbor.pageNumber - target.chunk.pageNumber) > 1) continue

                    // Constraint 3: Section consistency (if both have section titles, must match)
                    if (target.chunk.sectionTitle != null && neighbor.sectionTitle != null &&
                        !target.chunk.sectionTitle.equals(neighbor.sectionTitle, ignoreCase = true)
                    ) {
                        continue
                    }

                    // Constraint 4: Do not expand administrative boilerplate
                    if (isBoilerplateText(neighbor.text)) continue

                    // Constraint 5: For definition queries, do NOT expand into troubleshooting/failure tables
                    if (isDefinition) {
                        val isTroubleshooting = neighbor.text.contains("Troubleshooting", ignoreCase = true) ||
                            neighbor.text.contains("Causes of Failure", ignoreCase = true) ||
                            neighbor.text.contains("What to check", ignoreCase = true) ||
                            neighbor.text.contains("Failure", ignoreCase = true)
                        if (isTroubleshooting) continue
                    }

                    seenIds.add(neighbor.id)
                    expanded.add(
                        RetrievalResult(
                            chunk = neighbor,
                            relevanceScore = target.relevanceScore * 0.7f,
                            source = RetrievalSource.PARENT_EXPANSION,
                            intent = target.intent
                        )
                    )
                    logDebug(TAG, "[EXPANSION] Added neighbor ${neighbor.id} (chunkIndex $neighborIndex, page ${neighbor.pageNumber}) for chunk ${target.chunk.id}")
                }
            }
        }

        return expanded
    }

    // -------------------------------------------------------------------------
    // HYBRID SCORING ENGINE (Dense + BM25 + Domain-Agnostic Cross-Reranker + RRF)
    // -------------------------------------------------------------------------
    private fun scoreCandidates(
        chunks: List<ChunkEntity>,
        queryEmbedding: FloatArray,
        query: String,
        classification: QueryClassification? = null
    ): List<CandidateScore> {
        val cleanQuery = query.lowercase().trim()

        val compoundMatches = COMPOUND_IDENTIFIER_PATTERN.findAll(cleanQuery).toList()
        val compoundTokens = compoundMatches.map { it.value.lowercase() }
        val compoundSubTokens = compoundMatches.flatMap { listOf(it.groupValues[1].lowercase(), it.groupValues[2].lowercase()) }

        val rawTokens = cleanQuery.split(Regex("[^a-zA-Z0-9]+")).filter { it.length > 1 || it in compoundSubTokens }
        val queryKeywords = rawTokens.filter { it !in stopWords || it in compoundSubTokens }.distinct()
        val queryStemsMap = queryKeywords.associateWith { getRootStems(it) }
        val queryPhrase = queryKeywords.joinToString(" ")
        val isDefinition = classification?.isDefinition == true || (classification == null && isDefinitionQueryText(query))
        val targetConcept = if (isDefinition) (classification?.targetTerm ?: extractConceptFromQuery(query)) else null

        // Metric / Technical specification trigger detection
        val specTriggerWords = setOf(
            "capacity", "specs", "specification", "specifications", "spec", "power", "wattage",
            "battery", "charging", "voltage", "current", "temperature", "temperatures", "dimensions",
            "dimension", "size", "weight", "storage", "ram", "memory", "speed", "frequency", "resolution"
        )
        val isSpecQuery = queryKeywords.any { kw ->
            val stems = queryStemsMap[kw] ?: setOf(kw)
            specTriggerWords.any { trigger -> wordMatchesKeyword(trigger, kw, stems) }
        }

        // BM25 Lexical Scoring
        val docCount = chunks.size.toDouble()
        val chunkWordsMap = chunks.associateWith { chunk ->
            chunk.text.split(WORD_SPLIT_REGEX).filter { it.isNotEmpty() }
        }
        val avgDocLength = chunkWordsMap.values.map { it.size }.average().coerceAtLeast(10.0)

        val idfMap = queryKeywords.associateWith { kw ->
            val stems = queryStemsMap[kw] ?: setOf(kw)
            val containingDocs = chunks.count { chunk ->
                val words = chunkWordsMap[chunk] ?: emptyList()
                words.any { w -> wordMatchesKeyword(w, kw, stems) }
            }
            if (containingDocs > 0) ln((docCount - containingDocs + 0.5) / (containingDocs + 0.5) + 1.0) + 1.0 else 0.5
        }

        val candidates = mutableListOf<CandidateScore>()

        for (chunk in chunks) {
            val chunkText = chunk.text
            val chunkTextLower = chunkText.lowercase()
            val normalizedChunkLower = chunkTextLower.replace(NON_ALPHANUMERIC_REGEX, " ").replace(MULTI_SPACE_REGEX, " ")
            val chunkWords = chunkWordsMap[chunk] ?: emptyList()
            val docLen = chunkWords.size.toDouble()

            // Dense Cosine Similarity
            var denseScore = 0f
            if (chunk.embedding != null && queryEmbedding.isNotEmpty()) {
                val chunkEmbedding = FloatArrayConverter.toFloatArray(chunk.embedding)
                if (chunkEmbedding != null) {
                    denseScore = cosineSimilarity(queryEmbedding, chunkEmbedding)
                }
            }

            // BM25 Lexical Score
            var bm25Score = 0.0
            val k1 = 1.2
            val b = 0.75

            for (kw in queryKeywords) {
                val stems = queryStemsMap[kw] ?: setOf(kw)
                val termOccurrences = chunkWords.count { w -> wordMatchesKeyword(w, kw, stems) }
                if (termOccurrences > 0) {
                    val idf = idfMap[kw] ?: 1.0
                    val tf = (termOccurrences * (k1 + 1)) / (termOccurrences + k1 * (1.0 - b + b * (docLen / avgDocLength)))
                    bm25Score += tf * idf * 8.0
                }
            }

            // Phrase & Compound Identifier Match Bonus
            var phraseBonus = 0.0
            if (queryPhrase.length >= 4 && (chunkTextLower.contains(queryPhrase) || normalizedChunkLower.contains(queryPhrase))) {
                phraseBonus = 20.0
            }
            for (ct in compoundTokens) {
                val normCt = ct.replace("-", " ")
                if (chunkTextLower.contains(ct) || normalizedChunkLower.contains(normCt)) {
                    phraseBonus += 25.0
                }
            }

            val exactDefScore = computeDefinitionScore(chunk, chunkWords, normalizedChunkLower, targetConcept, isDefinition)

            val initialScore = (denseScore * 10f) + bm25Score.toFloat() + phraseBonus.toFloat()
            candidates.add(
                CandidateScore(
                    chunk = chunk,
                    initialScore = initialScore,
                    denseScore = denseScore,
                    bm25Score = bm25Score.toFloat(),
                    exactDefScore = exactDefScore
                )
            )
        }

        // Compute RRF (Reciprocal Rank Fusion) across 3 independent signals:
        // 1. Dense Semantic Similarity Rank
        // 2. BM25 Lexical Rank
        // 3. Exact Concept & Definition Match Rank
        val denseSorted = candidates.sortedByDescending { it.denseScore }
        val denseRankMap = denseSorted.mapIndexed { idx, c -> c.chunk.id to (idx + 1) }.toMap()

        val bm25Sorted = candidates.sortedByDescending { it.bm25Score }
        val bm25RankMap = bm25Sorted.mapIndexed { idx, c -> c.chunk.id to (idx + 1) }.toMap()

        val defSorted = candidates.sortedByDescending { it.exactDefScore }
        val defRankMap = defSorted.mapIndexed { idx, c -> c.chunk.id to (idx + 1) }.toMap()

        val kRrf = 60.0

        for (cand in candidates) {
            val dRank = denseRankMap[cand.chunk.id] ?: candidates.size
            val bRank = bm25RankMap[cand.chunk.id] ?: candidates.size

            val rrf = if (isDefinition) {
                val defRank = defRankMap[cand.chunk.id] ?: candidates.size
                (1.0 / (kRrf + dRank)) + (1.0 / (kRrf + bRank)) + (2.5 / (kRrf + defRank))
            } else {
                (1.0 / (kRrf + dRank)) + (1.0 / (kRrf + bRank))
            }
            cand.rrfScore = rrf.toFloat()
        }

        // Multi-Signal Cross-Reranking
        val defRegex = if (queryPhrase.length >= 4) {
            Regex("""(?i)\b(?:${queryKeywords.joinToString("""\s+""") { Regex.escape(it) }})\b\s+(?:is|refers\s+to|consists\s+of|describes|defines|involves)\b""")
        } else null

        val leadRegex = if (queryKeywords.size >= 2) {
            Regex("""(?i)\b(?:${queryKeywords.joinToString("""\s+""") { Regex.escape(it) }})\b\s*(?:include|are|:|consist)""")
        } else null

        for (cand in candidates) {
            val chunkText = cand.chunk.text
            val chunkLower = chunkText.lowercase()
            val normalizedChunkLower = chunkLower.replace(NON_ALPHANUMERIC_REGEX, " ").replace(MULTI_SPACE_REGEX, " ")
            val chunkWords = chunkWordsMap[cand.chunk] ?: chunkText.split(WORD_SPLIT_REGEX).filter { it.isNotEmpty() }
            val denseScore = cand.denseScore

            var matchedTermCount = 0
            for (kw in queryKeywords) {
                val stems = queryStemsMap[kw] ?: setOf(kw)
                if (chunkWords.any { w -> wordMatchesKeyword(w, kw, stems) }) {
                    matchedTermCount++
                }
            }

            val isAdministrativeBoilerplate = isBoilerplateText(chunkLower)

            if (isDefinition) {
                // DEFINITION-ORIENTED RERANKING:
                // Base score comes from RRF fusion + direct definition evidence.
                // Generic list and entity inflation is suppressed so troubleshooting tables do not outscore definitions.
                var rerank = (cand.rrfScore * 1000.0) + (cand.denseScore * 25.0) + cand.exactDefScore.toDouble()
                if (isAdministrativeBoilerplate) {
                    rerank -= 100.0
                    cand.rejectionReason = "Administrative Boilerplate"
                }
                if (cand.exactDefScore < 0f) {
                    rerank -= 50.0 // Heavy penalty for troubleshooting/failure sections
                }
                cand.rerankScore = rerank.toFloat()

                // Survival & tier for definition query
                cand.survived = cand.rerankScore >= 15.0f && cand.exactDefScore >= 0f && !isAdministrativeBoilerplate
                cand.tier = when {
                    isAdministrativeBoilerplate || cand.exactDefScore < 0f -> RelevanceTier.TIER_C_BOILERPLATE_WEAK
                    cand.exactDefScore >= 50.0f || cand.rerankScore >= 80.0f -> RelevanceTier.TIER_A_DIRECT_ANSWER
                    cand.survived -> RelevanceTier.TIER_B_SUPPORTING_CONTEXT
                    else -> RelevanceTier.TIER_C_BOILERPLATE_WEAK
                }
            } else {
                // STANDARD CROSS-RERANKING (for general queries):
                var rerank = cand.initialScore.toDouble() + (cand.rrfScore * 500.0)

                if (isAdministrativeBoilerplate) {
                    rerank -= 50.0
                    cand.rejectionReason = "Administrative Boilerplate"
                }

                if (queryPhrase.length >= 4 && (chunkLower.contains(queryPhrase) || normalizedChunkLower.contains(queryPhrase))) {
                    rerank += 25.0
                }
                for (ct in compoundTokens) {
                    val normCt = ct.replace("-", " ")
                    if (chunkLower.contains(ct) || normalizedChunkLower.contains(normCt)) {
                        rerank += 30.0
                    }
                }

                if (defRegex != null && defRegex.containsMatchIn(normalizedChunkLower)) {
                    rerank += 50.0
                }
                if (leadRegex != null && leadRegex.containsMatchIn(normalizedChunkLower)) {
                    rerank += 30.0
                }

                var specificEvidenceBonus = 0.0
                if (matchedTermCount > 0 || denseScore >= 0.35f) {
                    val entityMatches = GENERIC_ENTITY_REGEX.findAll(chunkText).toList()
                    if (entityMatches.size >= 2) {
                        specificEvidenceBonus += 15.0 + (entityMatches.size * 3.0).coerceAtMost(20.0)
                    }
                    if (COMMA_LIST_REGEX.containsMatchIn(chunkText)) {
                        specificEvidenceBonus += 20.0
                    }
                    val matchedDimensionRegexes = dimensionUnitMap.entries
                        .filter { entry -> entry.key.any { dimKey -> queryKeywords.any { kw -> wordMatchesKeyword(dimKey, kw, queryStemsMap[kw] ?: setOf(kw)) } } }
                        .map { it.value }

                    if (matchedDimensionRegexes.isNotEmpty()) {
                        val matchingUnitCount = matchedDimensionRegexes.sumOf { regex -> regex.findAll(chunkText).count() }
                        if (matchingUnitCount > 0) {
                            specificEvidenceBonus += 25.0 + (matchingUnitCount * 5.0).coerceAtMost(25.0)
                        }
                    } else if (isSpecQuery) {
                        val anyMatches = ANY_METRIC_REGEX.findAll(chunkText).toList()
                        if (anyMatches.isNotEmpty()) {
                            specificEvidenceBonus += 15.0 + (anyMatches.size * 3.0).coerceAtMost(15.0)
                        }
                    }

                    val listMatches = LIST_EVIDENCE_REGEX.findAll(chunkText).toList()
                    if (listMatches.size >= 2) {
                        val isListRequest = Regex("""(?i)\b(?:what\s+are|list|give\s+me\s+all|name\s+all|which\s+are)\b""").containsMatchIn(cleanQuery)
                        if (isListRequest) {
                            specificEvidenceBonus += 15.0 + (listMatches.size * 3.0).coerceAtMost(25.0)
                        } else {
                            specificEvidenceBonus += 15.0
                        }
                    }
                }

                rerank += specificEvidenceBonus

                if (queryKeywords.isNotEmpty()) {
                    val coverage = matchedTermCount.toDouble() / queryKeywords.size
                    if (coverage >= 1.0) {
                        rerank += 20.0
                    } else if (coverage >= 0.5) {
                        rerank += 10.0
                    } else if (matchedTermCount == 0) {
                        rerank -= 35.0
                    }
                }

                cand.rerankScore = rerank.toFloat()
                cand.survived = cand.rerankScore >= 5.0f && !isAdministrativeBoilerplate

                cand.tier = when {
                    isAdministrativeBoilerplate -> RelevanceTier.TIER_C_BOILERPLATE_WEAK
                    matchedTermCount == 0 && denseScore < 0.35f -> RelevanceTier.TIER_C_BOILERPLATE_WEAK
                    (specificEvidenceBonus >= 15.0 || rerank >= 20.0) && (matchedTermCount > 0 || denseScore >= 0.35f) -> RelevanceTier.TIER_A_DIRECT_ANSWER
                    matchedTermCount > 0 || cand.initialScore >= 5.0f -> RelevanceTier.TIER_B_SUPPORTING_CONTEXT
                    else -> RelevanceTier.TIER_C_BOILERPLATE_WEAK
                }
            }
        }

        // =========================================================================
        // LOG TOP 10 RETRIEVED CHUNKS BEFORE FILTERING
        // =========================================================================
        val top10BeforeFiltering = candidates.sortedByDescending { it.rerankScore }.take(10)
        logInfo(TAG, "========== TOP 10 RETRIEVED CHUNKS BEFORE FILTERING (Query: \"$query\") ==========")
        println("========== TOP 10 RETRIEVED CHUNKS BEFORE FILTERING (Query: \"$query\") ==========")
        for ((idx, cand) in top10BeforeFiltering.withIndex()) {
            val rank = idx + 1
            val preview = cand.chunk.text.replace("\n", " ").take(60).trim()
            val logStr = String.format(
                java.util.Locale.US,
                "[RETRIEVAL_TOP10] Rank %2d | Page %2d | Chunk %4d | Dense: %.4f | Lexical: %6.2f | ExactDef: %6.2f | RRF: %.5f | Final: %6.2f | Survived: %5b | Preview: \"%s\"",
                rank, cand.chunk.pageNumber, cand.chunk.id, cand.denseScore, cand.bm25Score, cand.exactDefScore, cand.rrfScore, cand.rerankScore, cand.survived, preview
            )
            logInfo("RETRIEVAL_TOP10", logStr)
            println(logStr)
        }
        println("================================================================================")

        return candidates
    }

    private fun computeDefinitionScore(
        chunk: ChunkEntity,
        chunkWords: List<String>,
        normalizedChunkLower: String,
        targetConcept: String?,
        isDefinition: Boolean
    ): Float {
        if (!isDefinition || targetConcept.isNullOrBlank()) return 0f
        val conceptLower = targetConcept.lowercase().trim()
        val conceptWords = conceptLower.split(WORD_SPLIT_REGEX).filter { it.isNotEmpty() }

        val conceptRegex = Regex("""(?i)\b${Regex.escape(conceptLower)}\b""")
        val hasConceptPhrase = conceptRegex.containsMatchIn(normalizedChunkLower)
        val hasAllConceptWords = conceptWords.isNotEmpty() && conceptWords.all { cw ->
            chunkWords.any { w -> w.equals(cw, ignoreCase = true) }
        }

        if (!hasConceptPhrase && !hasAllConceptWords) {
            return 0f
        }

        var score = 20.0f

        if (isDefinition) {
            // 1. Direct explicit definition: "<concept> is a/an/the technique/method/process/practice/procedure/..."
            val directDefRegex = Regex("""(?i)\b${Regex.escape(conceptLower)}\b\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*(?:technique|method|process|practice|procedure|system|approach|term|concept|form|type|discipline|art|science|mechanism|structure|tissue|organism|device|tool|component|operation|way|action)\b""")
            if (directDefRegex.containsMatchIn(normalizedChunkLower)) {
                score += 100.0f
            }

            // 2. Definitional copula: "<concept> is/are/was/were/refers to/defined as/means/consists of"
            val copulaDefRegex = Regex("""(?i)\b${Regex.escape(conceptLower)}\b\s+(?:is|are|was|were|refers\s+to|defined\s+as|is\s+defined\s+as|consists\s+of|describes|is\s+known\s+as|means)\b""")
            if (copulaDefRegex.containsMatchIn(normalizedChunkLower)) {
                score += 50.0f
            }

            // 3. Glossary definition: line starting with "<concept>:"
            val glossaryRegex = Regex("""(?im)^\s*${Regex.escape(conceptLower)}\s*[:\-]\s*[A-Z]""")
            if (glossaryRegex.containsMatchIn(chunk.text)) {
                score += 60.0f
            }

            // 4. Section title containing concept and introduction/overview
            val sectionTitleLower = chunk.sectionTitle?.lowercase() ?: ""
            if (sectionTitleLower.contains(conceptLower) && (sectionTitleLower.contains("intro") || sectionTitleLower.contains("overview") || sectionTitleLower.contains("about"))) {
                score += 50.0f
            }

            // 5. Intro header in chunk text
            val introHeaderRegex = Regex("""(?i)\b(?:section\s+1|introduction|overview)\b.*\b${Regex.escape(conceptLower)}\b""")
            if (introHeaderRegex.containsMatchIn(chunk.text)) {
                score += 40.0f
            }

            // 6. Page 1 / First chunk bonus for definition queries
            if (chunk.pageNumber == 1 && chunk.chunkIndex <= 1) {
                score += 30.0f
            }

            // 7. Negative penalties: non-definitional troubleshooting / failure tables
            val troubleshootingRegex = Regex("""(?i)\b(?:troubleshooting|troubleshoot|failure|failures|causes\s+of\s+failure|what\s+to\s+check|likely\s+causes|problem\s+solving|symptoms|defect)\b""")
            if (troubleshootingRegex.containsMatchIn(chunk.text)) {
                score -= 80.0f
            }

            val toolsRegex = Regex("""(?i)\b(?:tools\s+required|equipment\s+needed|supplies|materials\s+list)\b""")
            if (toolsRegex.containsMatchIn(chunk.text)) {
                score -= 40.0f
            }

            val aftercareRegex = Regex("""(?i)\b(?:aftercare|post-graft|maintenance|weekly\s+inspection)\b""")
            if (aftercareRegex.containsMatchIn(chunk.text)) {
                score -= 30.0f
            }
        }

        return score
    }

    private val DEFINITION_REGEX_1 = Regex("""^\s*(?i:what\s+does)\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+(?i:mean)[\s\?\.\!]*$""")
    private val DEFINITION_REGEX_2 = Regex("""^\s*(?i:what\s+is\s+meant\s+by|meaning\s+of|definition\s+of)\s+(?:a\s+|an\s+|the\s+)?(.+?)[\s\?\.\!]*$""")
    private val DEFINITION_REGEX_3 = Regex(
        """^\s*(?i:what\s+is)\s+(?!(?:the|all|every)\b)(?:a\s+|an\s+)?([a-zA-Z0-9\-_]+(?:\s+[a-zA-Z0-9\-_]+){0,3})[\s\?\.\!]*$|^\s*(?i:define|explain|describe)\s+(?:a\s+|an\s+|the\s+)?([a-zA-Z0-9\-_]+(?:\s+[a-zA-Z0-9\-_]+){0,3})[\s\?\.\!]*$"""
    )

    private fun isDefinitionQueryText(query: String): Boolean {
        val concept = extractConceptFromQuery(query)
        if (concept.isNullOrBlank()) return false
        val lower = concept.lowercase()
        val nonDefWords = listOf(
            "recommended", "needed", "required", "used", "better", "best", "difference",
            "why", "how", "when", "where", "temperature", "price", "cost", "salary",
            "capacity", "spec", "specs", "specification", "power", "wattage", "voltage",
            "process", "processes", "step", "steps", "list", "recipe"
        )
        return nonDefWords.none { lower.contains(it) }
    }

    private fun extractConceptFromQuery(query: String): String? {
        val q = query.trim()
        val m1 = DEFINITION_REGEX_1.find(q)
        if (m1 != null) return cleanQueryText(m1.groupValues[1])
        val m2 = DEFINITION_REGEX_2.find(q)
        if (m2 != null) return cleanQueryText(m2.groupValues[1])
        val m3 = DEFINITION_REGEX_3.find(q)
        if (m3 != null) {
            val raw = (m3.groupValues.drop(1).firstOrNull { it.isNotBlank() } ?: "").trim()
            return cleanQueryText(raw)
        }
        return null
    }

    // -------------------------------------------------------------------------
    // UTILITY METHODS
    // -------------------------------------------------------------------------
    private fun isBoilerplateText(text: String): Boolean {
        val lower = text.lowercase()
        return listOf(
            "scientific and technical information (sti)",
            "sti program",
            "advancement of aeronautics and space science",
            "standard form 298",
            "report documentation page",
            "distribution statement",
            "table of contents",
            "all rights reserved",
            "published by the"
        ).any { lower.contains(it) }
    }

    private fun cleanQueryText(query: String): String {
        return query.replace(Regex("""[?.,!;:"]+"""), "").trim()
    }

    private fun getRootStems(word: String): Set<String> {
        val w = word.lowercase().trim()
        val stems = mutableSetOf(w)
        if (w.endsWith("ies") && w.length > 4) stems.add(w.removeSuffix("ies") + "y")
        if (w.endsWith("es") && w.length > 4) stems.add(w.removeSuffix("es"))
        if (w.endsWith("s") && w.length > 3) stems.add(w.removeSuffix("s"))
        if (w.endsWith("ed") && w.length > 4) stems.add(w.removeSuffix("ed"))
        if (w.endsWith("ing") && w.length > 5) stems.add(w.removeSuffix("ing"))
        if (w.endsWith("tion") && w.length > 6) stems.add(w.removeSuffix("tion"))
        if (w.endsWith("ity") && w.length > 5) stems.add(w.removeSuffix("ity"))
        if (w.endsWith("able") && w.length > 6) stems.add(w.removeSuffix("able"))
        if (w.endsWith("ive") && w.length > 5) stems.add(w.removeSuffix("ive"))
        return stems
    }

    private fun wordMatchesKeyword(docWord: String, kw: String, kwStems: Set<String>): Boolean {
        val cleanDocWord = docWord.replace(Regex("[^a-zA-Z0-9]"), "").lowercase()
        if (cleanDocWord.isEmpty()) return false
        if (cleanDocWord == kw || cleanDocWord in kwStems) return true
        if (kwStems.any { (it.length >= 4 && cleanDocWord.startsWith(it)) || (it.length >= 4 && cleanDocWord.contains(it)) }) return true
        val docStems = getRootStems(cleanDocWord)
        if (docStems.any { it in kwStems }) return true
        return false
    }

    private fun cosineSimilarity(vectorA: FloatArray, vectorB: FloatArray): Float {
        if (vectorA.size != vectorB.size) return 0f
        var dotProduct = 0.0
        var normA = 0.0
        var normB = 0.0
        for (i in vectorA.indices) {
            val a = vectorA[i].toDouble()
            val b = vectorB[i].toDouble()
            dotProduct += a * b
            normA += a * a
            normB += b * b
        }
        val denom = sqrt(normA) * sqrt(normB)
        return if (denom == 0.0) 0f else (dotProduct / denom).toFloat()
    }

    private fun logInfo(tag: String, msg: String) {
        try {
            Log.i(tag, msg)
        } catch (_: Throwable) {
            println("[$tag] $msg")
        }
    }

    private fun logDebug(tag: String, msg: String) {
        try {
            Log.d(tag, msg)
        } catch (_: Throwable) {
            println("[$tag] $msg")
        }
    }

    private fun logWarn(tag: String, msg: String) {
        try {
            Log.w(tag, msg)
        } catch (_: Throwable) {
            println("[$tag] $msg")
        }
    }
}
