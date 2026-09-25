package com.siddu.offlinedocai.ai.retrieval

import android.util.Log
import com.siddu.offlinedocai.data.dao.ChunkDao
import com.siddu.offlinedocai.data.entities.ChunkEntity
import com.siddu.offlinedocai.data.FloatArrayConverter
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Enhanced two-stage hybrid retriever with relevance validation and multi-signal cross-reranking.
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
        private val PAGE_LOOKUP_REGEX = Regex("""(?i)\b(?:page|pg|p\.|page\s*no|page\s*number|page\s*#)[\s\.\-\#:]+(\d+)\b|\b(\d+)(?:st|nd|rd|th)?\s+page\b""")
        private val COMPOUND_IDENTIFIER_PATTERN = Regex("""(?i)\b(pre[- ]phase|phase|stage|step|chapter|section|appendix|type|level|tier)\s+([a-zA-Z0-9]+)\b""")
    }

    private val summaryTriggerKeywords = setOf(
        "summarize", "summary", "overview", "what is this document about", "what does this document", "main topics", "brief of", "gist"
    )

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
        var rerankScore: Float = 0f,
        var tier: RelevanceTier = RelevanceTier.TIER_B_SUPPORTING_CONTEXT,
        var rejectionReason: String? = null
    )

    suspend fun retrieve(
        queryEmbedding: FloatArray,
        relevantDocumentIds: List<Long>,
        topK: Int,
        query: String = ""
    ): List<ChunkEntity> {
        if (relevantDocumentIds.isEmpty()) return emptyList()

        val chunks = chunkDao.getAllChunksForDocumentIds(relevantDocumentIds)
        if (chunks.isEmpty()) return emptyList()

        val cleanQuery = query.lowercase().trim()

        // 0. Stage 0: PAGE_LOOKUP Deterministic Intent Detection & Metadata Filtering
        val pageLookupRegex = Regex("""(?i)\b(?:page|pg|p\.|page\s*no|page\s*number|page\s*#)[\s\.\-\#:]+(\d+)\b|\b(\d+)(?:st|nd|rd|th)?\s+page\b""")
        val pageMatch = pageLookupRegex.find(cleanQuery)
        val requestedPage = pageMatch?.let { m ->
            m.groupValues[1].toIntOrNull() ?: m.groupValues[2].toIntOrNull()
        }

        if (requestedPage != null) {
            val pageChunks = chunks.filter { it.pageNumber == requestedPage }
                .sortedBy { it.id } // preserve document/chunk sequential order

            // Diagnostic Logging (Requirement 7)
            Log.i(TAG, "[PAGE_LOOKUP] Detected Intent: PAGE_LOOKUP")
            Log.i(TAG, "[PAGE_LOOKUP] Requested Page: $requestedPage")
            Log.i(TAG, "[PAGE_LOOKUP] Resolved Metadata Page Field: ChunkEntity.pageNumber")
            Log.i(TAG, "[PAGE_LOOKUP] Candidate Chunk IDs: ${chunks.map { it.id }}")
            Log.i(TAG, "[PAGE_LOOKUP] Candidate Pages: ${chunks.map { it.pageNumber }}")
            Log.i(TAG, "[PAGE_LOOKUP] Final Selected Pages: ${pageChunks.map { it.pageNumber }}")

            if (pageChunks.isEmpty()) {
                Log.w(TAG, "[PAGE_LOOKUP] Requested Page $requestedPage not found in document chunks.")
                return emptyList()
            }

            return pageChunks
        }

        val compoundIdentifierPattern = Regex("""(?i)\b(pre[- ]phase|phase|stage|step|chapter|section|appendix|type|level|tier)\s+([a-zA-Z0-9]+)\b""")
        val compoundMatches = compoundIdentifierPattern.findAll(cleanQuery).toList()
        val compoundTokens = compoundMatches.map { it.value.lowercase() }
        val compoundSubTokens = compoundMatches.flatMap { listOf(it.groupValues[1].lowercase(), it.groupValues[2].lowercase()) }

        val rawTokens = cleanQuery.split(Regex("[^a-zA-Z0-9]+")).filter { it.length > 1 || it in compoundSubTokens }

        // Classify as generic summary query ONLY if user asks for document-wide summary without a specific chapter/section/page
        val hasSpecificTarget = Regex("""(?i)\b(?:chapter|section|page|phase|appendix)\b""").containsMatchIn(cleanQuery)
        val isExplicitSummaryRequest = !hasSpecificTarget && (summaryTriggerKeywords.any { cleanQuery.contains(it) } ||
            (rawTokens.size <= 3 && listOf("summary", "summarize", "overview", "about").any { cleanQuery.contains(it) }))

        if (isExplicitSummaryRequest) {
            if (chunks.size <= topK) return chunks
            val step = chunks.size.toDouble() / topK
            val selected = mutableListOf<ChunkEntity>()
            for (i in 0 until topK) {
                val idx = (i * step).toInt().coerceIn(0, chunks.size - 1)
                if (!selected.contains(chunks[idx])) {
                    selected.add(chunks[idx])
                }
            }
            return selected
        }

        // 1. Stage 1: Query Preprocessing & Morphological Term Normalization
        val queryKeywords = rawTokens.filter { it !in stopWords || it in compoundSubTokens }.distinct()
        val queryStemsMap = queryKeywords.associateWith { getRootStems(it) }
        val queryPhrase = queryKeywords.joinToString(" ")

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

        // 2. Stage 1: Initial Hybrid Retrieval Scoring (Dense + BM25 Lexical)
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
            if (chunk.embedding != null) {
                val chunkEmbedding = FloatArrayConverter.toFloatArray(chunk.embedding)
                if (chunkEmbedding != null && queryEmbedding.isNotEmpty()) {
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
                    bm25Score += tf * idf * 8.0 // Scaled for hybrid balance
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

            val initialScore = (denseScore * 10f) + bm25Score.toFloat() + phraseBonus.toFloat()
            candidates.add(CandidateScore(chunk = chunk, initialScore = initialScore, denseScore = denseScore))
        }

        // 3. Stage 2: Relevance Validation & Multi-Signal Cross-Reranking
        // Optimize for large documents: Only perform heavy regex reranking on top scoring candidates
        val candidatesWithSignal = candidates.filter { it.initialScore > 0f || it.denseScore >= 0.2f }
            .sortedByDescending { it.initialScore }
        val topCandidatesToRerank = if (candidatesWithSignal.isNotEmpty()) {
            candidatesWithSignal.take(80)
        } else {
            candidates.sortedByDescending { it.initialScore }.take(20)
        }
        val topCandidateSet = topCandidatesToRerank.toSet()

        val defRegex = if (queryPhrase.length >= 4) {
            Regex("""(?i)\b(?:${queryKeywords.joinToString("""\s+""") { Regex.escape(it) }})\b\s+(?:is|refers\s+to|consists\s+of|describes|defines|involves)\b""")
        } else null

        val leadRegex = if (queryKeywords.size >= 2) {
            Regex("""(?i)\b(?:${queryKeywords.joinToString("""\s+""") { Regex.escape(it) }})\b\s*(?:include|are|:|consist)""")
        } else null

        for (cand in candidates) {
            if (cand !in topCandidateSet) {
                cand.tier = RelevanceTier.TIER_C_BOILERPLATE_WEAK
                cand.rerankScore = cand.initialScore
                continue
            }

            val chunkText = cand.chunk.text
            val chunkLower = chunkText.lowercase()
            val normalizedChunkLower = chunkLower.replace(NON_ALPHANUMERIC_REGEX, " ").replace(MULTI_SPACE_REGEX, " ")
            val chunkWords = chunkWordsMap[cand.chunk] ?: chunkText.split(WORD_SPLIT_REGEX).filter { it.isNotEmpty() }
            val denseScore = cand.denseScore
            var rerank = cand.initialScore.toDouble()

            // Count matching query terms
            var matchedTermCount = 0
            for (kw in queryKeywords) {
                val stems = queryStemsMap[kw] ?: setOf(kw)
                if (chunkWords.any { w -> wordMatchesKeyword(w, kw, stems) }) {
                    matchedTermCount++
                }
            }

            // Signal A: Boilerplate / Administrative Fluff Penalty
            val isAdministrativeBoilerplate = listOf(
                "scientific and technical information (sti)",
                "sti program",
                "advancement of aeronautics and space science",
                "standard form 298",
                "report documentation page",
                "distribution statement",
                "table of contents",
                "all rights reserved",
                "published by the"
            ).any { chunkLower.contains(it) }

            if (isAdministrativeBoilerplate) {
                rerank -= 50.0
                cand.rejectionReason = "Administrative / STI Program Boilerplate"
            }

            // Signal B: Exact Query Phrase & Compound Identifier Bonus
            if (queryPhrase.length >= 4 && (chunkLower.contains(queryPhrase) || normalizedChunkLower.contains(queryPhrase))) {
                rerank += 25.0
            }
            for (ct in compoundTokens) {
                val normCt = ct.replace("-", " ")
                if (chunkLower.contains(ct) || normalizedChunkLower.contains(normCt)) {
                    rerank += 30.0
                }
            }

            // Definition lead bonus (e.g. "SE engine is a...")
            if (defRegex != null && defRegex.containsMatchIn(normalizedChunkLower)) {
                rerank += 25.0
            }

            // Subject lead / enumeration pattern (e.g. "Technical Management Processes include:")
            if (leadRegex != null && leadRegex.containsMatchIn(normalizedChunkLower)) {
                rerank += 30.0
            }

            // Signal C: Subject / Direct Answer Evidence Specificity (Fully Domain-Agnostic)
            var specificEvidenceBonus = 0.0

            if (matchedTermCount > 0 || denseScore >= 0.35f) {
                // 1. Generic Multi-Word Named Entities & Proper Noun Identifiers:
                val entityMatches = GENERIC_ENTITY_REGEX.findAll(chunkText).toList()
                if (entityMatches.size >= 2) {
                    specificEvidenceBonus += 15.0 + (entityMatches.size * 3.0).coerceAtMost(20.0)
                }

                // 2. Generic Comma-Separated Proper Nouns / Entity List Series:
                if (COMMA_LIST_REGEX.containsMatchIn(chunkText)) {
                    specificEvidenceBonus += 20.0
                }

                // 3. Dimensional Technical Specification Unit Matching:
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

                // 4. Structured Colon Headings / Bullet Lists Evidence:
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

            // Signal D: Query Keyword Coverage
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

            // Classify Relevance Tier
            cand.tier = when {
                isAdministrativeBoilerplate -> RelevanceTier.TIER_C_BOILERPLATE_WEAK
                matchedTermCount == 0 && denseScore < 0.35f -> RelevanceTier.TIER_C_BOILERPLATE_WEAK
                (specificEvidenceBonus >= 15.0 || rerank >= 20.0) && (matchedTermCount > 0 || denseScore >= 0.35f) -> RelevanceTier.TIER_A_DIRECT_ANSWER
                matchedTermCount > 0 || cand.initialScore >= 5.0f -> RelevanceTier.TIER_B_SUPPORTING_CONTEXT
                else -> RelevanceTier.TIER_C_BOILERPLATE_WEAK
            }
        }

        // Sort candidates by rerank score descending
        val sortedCandidates = candidates.sortedByDescending { it.rerankScore }

        // Diagnostic Logging for each candidate
        for ((index, cand) in sortedCandidates.withIndex()) {
            val preview = cand.chunk.text.take(65).replace("\n", " ")
            val rankStr = if (cand.tier == RelevanceTier.TIER_C_BOILERPLATE_WEAK && cand.rerankScore < 10f) {
                "FILTERED (${cand.rejectionReason ?: "Low relevance"})"
            } else {
                "#${index + 1}"
            }
            Log.d(TAG, "[RAG Reranker] Chunk ${cand.chunk.id} (Page ${cand.chunk.pageNumber}): InitScore=${"%.2f".format(cand.initialScore)}, RerankScore=${"%.2f".format(cand.rerankScore)}, Tier=${cand.tier}, Rank=$rankStr | Preview: \"$preview...\"")
        }

        // Filter out low-relevance boilerplate candidates
        val relevantCandidates = sortedCandidates.filter {
            it.tier != RelevanceTier.TIER_C_BOILERPLATE_WEAK && it.rerankScore >= 5.0f
        }

        if (relevantCandidates.isEmpty()) {
            Log.w(TAG, "No relevant candidate passed evidence threshold for query: '$query'")
            return emptyList()
        }

        // Strategy A: Comparison / Multi-Hop Retrieval (ensure evidence for all compared entities)
        val comparisonRegex = Regex("""(?i)\b(?:compare|difference\s+between)\s+(.+?)\s+(?:and|with|to|vs|versus)\s+(.+)|\b(.+?)\s+(?:vs|versus)\s+(.+)\b""")
        val compMatch = comparisonRegex.find(cleanQuery)
        if (compMatch != null) {
            val targetA = compMatch.groupValues[1].ifEmpty { compMatch.groupValues[3] }.trim()
            val targetB = compMatch.groupValues[2].ifEmpty { compMatch.groupValues[4] }.trim()

            if (targetA.isNotEmpty() && targetB.isNotEmpty()) {
                val wordsA = targetA.split(Regex("[^a-zA-Z0-9]+")).filter { it.isNotEmpty() }
                val wordsB = targetB.split(Regex("[^a-zA-Z0-9]+")).filter { it.isNotEmpty() }

                val candA = relevantCandidates.filter { c ->
                    val cText = c.chunk.text.lowercase()
                    wordsA.any { cText.contains(it.lowercase()) }
                }
                val candB = relevantCandidates.filter { c ->
                    val cText = c.chunk.text.lowercase()
                    wordsB.any { cText.contains(it.lowercase()) }
                }

                val merged = mutableListOf<ChunkEntity>()
                val halfK = max(1, topK / 2)
                for (c in candA.take(halfK)) {
                    if (c.chunk !in merged) merged.add(c.chunk)
                }
                for (c in candB.take(halfK)) {
                    if (c.chunk !in merged) merged.add(c.chunk)
                }
                for (c in relevantCandidates) {
                    if (c.chunk !in merged && merged.size < topK) {
                        merged.add(c.chunk)
                    }
                }
                if (merged.isNotEmpty()) {
                    return merged
                }
            }
        }

        // Strategy B: Location / Source Query (ensure page diversity across document)
        val locationRegex = Regex("""(?i)\bwhere\s+(?:does|can|in|is|are|do)?\s*(?:the\s+book|the\s+document|we|i)?\s*(?:discuss|mention|talk\s+about|cover|find|address|describe)\b|\bwhere\s+in\s+the\s+book\b|\bwhich\s+(?:pages|sections|chapters)\s+(?:discuss|mention|cover)\b""")
        if (locationRegex.containsMatchIn(cleanQuery)) {
            val pageSeen = mutableSetOf<Int>()
            val diverse = mutableListOf<ChunkEntity>()
            for (c in relevantCandidates) {
                if (pageSeen.add(c.chunk.pageNumber)) {
                    diverse.add(c.chunk)
                }
                if (diverse.size >= topK) break
            }
            if (diverse.isNotEmpty()) {
                return diverse
            }
        }

        // Strategy C: General Document QA & Structured Queries
        return relevantCandidates.take(topK).map { it.chunk }
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
}
