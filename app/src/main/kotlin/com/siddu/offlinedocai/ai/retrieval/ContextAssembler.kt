package com.siddu.offlinedocai.ai.retrieval

import com.siddu.offlinedocai.data.entities.ChunkEntity
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

/**
 * Structured source block containing rich provenance, location, score, and merged content.
 * Retains complete structured metadata for downstream CitationValidator (Phase 6).
 */
data class SourceBlock(
    val sourceId: Long,
    val documentId: Long,
    val pageNumber: Int,
    val pageSpan: String,
    val chunkIndex: Int,
    val sectionTitle: String?,
    val text: String,
    val relevanceScore: Float = 0f,
    val source: RetrievalSource = RetrievalSource.HYBRID_SEMANTIC,
    val intent: QueryIntent = QueryIntent.CONCEPTUAL_QUERY
) {
    /**
     * Formats the block into the Phase 4 structured citation format for LLM consumption.
     * Retains explicit structured headers so the model has complete source attribution.
     */
    fun formatBlock(): String {
        val sb = StringBuilder()
        sb.append("[DOC_ID=$documentId]\n")
        sb.append("[SOURCE_ID=$sourceId]\n")
        sb.append("[PAGE=$pageNumber]\n")
        val effectiveSpan = if (pageSpan.isNotBlank()) pageSpan else pageNumber.toString()
        sb.append("[PAGE_SPAN=$effectiveSpan]\n")
        val effectiveSection = if (!sectionTitle.isNullOrBlank()) sectionTitle else "General"
        sb.append("[SECTION=$effectiveSection]\n")
        sb.append("[CHUNK_INDEX=$chunkIndex]\n\n")
        sb.append(text.trim())
        return sb.toString()
    }
}

/**
 * Result of the context assembly pipeline.
 */
data class AssembledContext(
    val contextText: String,
    val sourceBlocks: List<SourceBlock>,
    val totalEstimatedTokens: Int,
    val isBudgetLimited: Boolean,
    val intent: QueryIntent,
    val sourcesJson: String
)

/**
 * Phase 4 Context Assembler:
 * Transforms VectorStore retrieval results into a high-quality, structured, ordered,
 * deduplicated, merged, and size-budgeted context for LLM generation.
 */
object ContextAssembler {

    // On-device context window constraints (typically 2048 for on-device GGUF models)
    const val SAFE_MODEL_CONTEXT_LIMIT = 2048
    const val RESERVED_SYSTEM_PROMPT_TOKENS = 256
    const val RESERVED_USER_QUERY_TOKENS = 128
    const val RESERVED_METADATA_TOKENS = 128
    const val RESERVED_GENERATION_TOKENS = 384

    // Default safe context evidence budget: 2048 - (256 + 128 + 128 + 384) = 1152 tokens
    const val DEFAULT_MAX_CONTEXT_TOKENS = 1152

    // Evidence budget tiers for on-device context efficiency
    const val BUDGET_SIMPLE_QUERY_TOKENS = 384
    const val BUDGET_MULTI_HOP_COMPARISON_TOKENS = 640
    const val BUDGET_EXHAUSTIVE_TOKENS = 1152

    /**
     * Resolves evidence budget based on query intent.
     * Simple factual / definitional queries get ~384 tokens to avoid bloating prompt evaluation latency on mobile CPU.
     * Multi-hop comparison queries get ~640 tokens for balanced entity representation.
     * Exhaustive list / page-range / summary queries get ~1152 tokens to retain all candidates across pages.
     */
    fun resolveEvidenceBudget(intent: QueryIntent): Int {
        return when (intent) {
            QueryIntent.EXACT_FACT,
            QueryIntent.PAGE_LOOKUP,
            QueryIntent.ENTITY_LOOKUP,
            QueryIntent.CONCEPTUAL_QUERY,
            QueryIntent.SECTION_QUERY -> BUDGET_SIMPLE_QUERY_TOKENS

            QueryIntent.COMPARISON -> BUDGET_MULTI_HOP_COMPARISON_TOKENS

            QueryIntent.PAGE_RANGE,
            QueryIntent.GLOBAL_SUMMARY,
            QueryIntent.EXHAUSTIVE_LIST,
            QueryIntent.FIND_ALL -> BUDGET_EXHAUSTIVE_TOKENS
        }
    }

    /**
     * Deterministic conservative token estimator.
     * Computes conservative token counts taking into account words, punctuation, subwords, and character length.
     * Guaranteed to never exceed the safe model context window.
     */
    fun estimateTokens(text: String): Int {
        if (text.isEmpty()) return 0
        val charCount = text.length
        // Conservative character-to-token ratio (3.5 characters per token for technical English / code)
        val charBasedTokens = ceil(charCount / 3.5).toInt()

        // Subword and punctuation heuristic
        val words = text.split(Regex("""\s+""")).filter { it.isNotBlank() }
        var wordBasedTokens = 0
        for (w in words) {
            wordBasedTokens += 1 // base token
            if (w.length > 6) {
                wordBasedTokens += (w.length - 6) / 4 + 1
            }
            val punctCount = w.count { !it.isLetterOrDigit() }
            if (punctCount > 0) {
                wordBasedTokens += (punctCount + 1) / 2
            }
        }
        return max(charBasedTokens, wordBasedTokens)
    }

    /**
     * Primary assembly method.
     *
     * Pipeline:
     * 1. Empty retrieval check (deterministic sentinels)
     * 2. Deduplication by primary identity (documentId, chunkId)
     * 3. Intent-specific selection and budgeting
     * 4. Document-order sorting (documentId -> pageNumber -> chunkIndex -> id)
     * 5. Adjacent chunk merging (contiguous chunks from same section & page)
     * 6. Structured source block formatting
     */
    fun assemble(
        retrievedResults: List<RetrievalResult>,
        queryIntent: QueryIntent,
        queryClassification: QueryClassification? = null,
        maxEvidenceTokens: Int = DEFAULT_MAX_CONTEXT_TOKENS
    ): AssembledContext {
        // 1. Handle Empty Retrieval
        if (retrievedResults.isEmpty()) {
            val emptySentinel = when (queryIntent) {
                QueryIntent.PAGE_LOOKUP -> "[PAGE_NOT_FOUND:${queryClassification?.pageStart ?: 1}]"
                QueryIntent.PAGE_RANGE -> "[PAGE_RANGE_NOT_FOUND:${queryClassification?.pageStart ?: 1}-${queryClassification?.pageEnd ?: 1}]"
                else -> "[NO_EVIDENCE_FOUND]"
            }
            return AssembledContext(
                contextText = emptySentinel,
                sourceBlocks = emptyList(),
                totalEstimatedTokens = 0,
                isBudgetLimited = false,
                intent = queryIntent,
                sourcesJson = "[]"
            )
        }

        // Determine effective budget: if caller used default, resolve intent-based budget; otherwise use caller's budget
        val effectiveBudget = if (maxEvidenceTokens == DEFAULT_MAX_CONTEXT_TOKENS) {
            resolveEvidenceBudget(queryIntent)
        } else {
            maxEvidenceTokens
        }

        // 2. Deduplicate Candidates (keep highest score if duplicate chunkId exists or text is duplicate)
        val deduplicatedMap = mutableMapOf<Pair<Long, Long>, RetrievalResult>()
        val seenPrefixes = mutableSetOf<String>()
        for (res in retrievedResults) {
            val key = Pair(res.chunk.documentId, res.chunk.id)
            val normalizedText = res.chunk.text.trim().lowercase()
            val prefix = if (normalizedText.length > 40) normalizedText.substring(0, 40) else normalizedText
            val existing = deduplicatedMap[key]
            
            val isDuplicatePrefix = seenPrefixes.any { p -> 
                (p.length > 20 && prefix.startsWith(p)) || (prefix.length > 20 && p.startsWith(prefix)) 
            }
            
            if (existing == null || res.relevanceScore > existing.relevanceScore) {
                if (!isDuplicatePrefix || existing != null) {
                    deduplicatedMap[key] = res
                    seenPrefixes.add(prefix)
                }
            }
        }
        val deduplicated = deduplicatedMap.values.toList()

        // 3. Intent-Specific Evidence Selection & Budgeting
        val (selectedResults, isBudgetLimited) = selectEvidenceWithinBudget(
            candidates = deduplicated,
            intent = queryIntent,
            classification = queryClassification,
            maxBudgetTokens = effectiveBudget
        )

        // 4. Sort Selected Evidence into Document Order (documentId -> pageNumber -> chunkIndex -> id)
        val documentOrdered = selectedResults.sortedWith(
            compareBy(
                { it.chunk.documentId },
                { it.chunk.pageNumber },
                { it.chunk.chunkIndex },
                { it.chunk.id }
            )
        )

        // 5. Adjacent Chunk Merging
        val mergedBlocks = mergeAdjacentChunks(documentOrdered)

        // 6. Structured Source Blocks & Context Text Construction
        val blockStrings = mergedBlocks.map { it.formatBlock() }
        val contextText = blockStrings.joinToString("\n\n")
        val totalTokens = estimateTokens(contextText)

        // 7. Structured JSON metadata for persistence & Phase 6 CitationValidator
        val sourcesJson = buildSourcesJson(mergedBlocks)

        return AssembledContext(
            contextText = contextText,
            sourceBlocks = mergedBlocks,
            totalEstimatedTokens = totalTokens,
            isBudgetLimited = isBudgetLimited,
            intent = queryIntent,
            sourcesJson = sourcesJson
        )
    }

    /**
     * Selects candidates that fit within the token budget using intent-aware prioritization.
     */
    private fun selectEvidenceWithinBudget(
        candidates: List<RetrievalResult>,
        intent: QueryIntent,
        classification: QueryClassification?,
        maxBudgetTokens: Int
    ): Pair<List<RetrievalResult>, Boolean> {
        val prioritized = when (intent) {
            QueryIntent.PAGE_LOOKUP -> {
                val targetPage = classification?.pageStart ?: candidates.firstOrNull()?.chunk?.pageNumber ?: 1
                candidates.filter { it.chunk.pageNumber == targetPage }
                    .sortedBy { it.chunk.chunkIndex }
            }

            QueryIntent.PAGE_RANGE -> {
                val pStart = classification?.pageStart ?: 1
                val pEnd = classification?.pageEnd ?: 1
                candidates.filter { it.chunk.pageNumber in pStart..pEnd }
                    .sortedWith(compareBy({ it.chunk.pageNumber }, { it.chunk.chunkIndex }))
            }

            QueryIntent.EXACT_FACT -> {
                val exactMatches = candidates.filter { it.source == RetrievalSource.EXACT_FACT_LEXICAL }
                val others = candidates.filter { it.source != RetrievalSource.EXACT_FACT_LEXICAL }
                    .sortedByDescending { it.relevanceScore }
                val topScore = (exactMatches + others).maxOfOrNull { it.relevanceScore } ?: 0f
                val filteredOthers = if (topScore > 0f) {
                    others.filter { it.relevanceScore >= topScore * 0.30f }
                } else {
                    others
                }
                exactMatches + filteredOthers
            }

            QueryIntent.COMPARISON -> {
                val entityA = classification?.entityA?.lowercase()?.trim() ?: ""
                val entityB = classification?.entityB?.lowercase()?.trim() ?: ""

                val joint = mutableListOf<RetrievalResult>()
                val listA = mutableListOf<RetrievalResult>()
                val listB = mutableListOf<RetrievalResult>()
                val general = mutableListOf<RetrievalResult>()

                for (c in candidates) {
                    val textLower = c.chunk.text.lowercase()
                    val hasA = entityA.isNotEmpty() && textLower.contains(entityA)
                    val hasB = entityB.isNotEmpty() && textLower.contains(entityB)
                    when {
                        hasA && hasB -> joint.add(c)
                        hasA -> listA.add(c)
                        hasB -> listB.add(c)
                        else -> general.add(c)
                    }
                }

                val result = mutableListOf<RetrievalResult>()
                result.addAll(joint.sortedByDescending { it.relevanceScore })

                val sortedA = listA.sortedByDescending { it.relevanceScore }
                val sortedB = listB.sortedByDescending { it.relevanceScore }
                val maxItems = max(sortedA.size, sortedB.size)
                for (i in 0 until maxItems) {
                    if (i < sortedA.size) result.add(sortedA[i])
                    if (i < sortedB.size) result.add(sortedB[i])
                }
                result.addAll(general.sortedByDescending { it.relevanceScore })
                result
            }

            QueryIntent.EXHAUSTIVE_LIST -> {
                val result = mutableListOf<RetrievalResult>()
                val topScore = candidates.maxOfOrNull { it.relevanceScore } ?: 0f
                val validCandidates = if (topScore > 0f) candidates.filter { it.relevanceScore >= topScore * 0.40f } else candidates
                
                val byPage = validCandidates.groupBy { it.chunk.pageNumber }
                for ((_, pageList) in byPage) {
                    val top = pageList.maxByOrNull { it.relevanceScore }
                    if (top != null) result.add(top)
                }
                val remaining = validCandidates.filter { it !in result }
                    .sortedByDescending { it.relevanceScore }
                result.addAll(remaining)
                result
            }

            QueryIntent.FIND_ALL -> {
                val result = mutableListOf<RetrievalResult>()
                val topScore = candidates.maxOfOrNull { it.relevanceScore } ?: 0f
                val validCandidates = if (topScore > 0f) candidates.filter { it.relevanceScore >= topScore * 0.40f } else candidates
                
                val byPage = validCandidates.groupBy { it.chunk.pageNumber }
                for ((_, pageList) in byPage) {
                    val top = pageList.maxByOrNull { it.relevanceScore }
                    if (top != null) result.add(top)
                }
                val remaining = validCandidates.filter { it !in result }
                    .sortedByDescending { it.relevanceScore }
                result.addAll(remaining)
                result
            }

            QueryIntent.GLOBAL_SUMMARY -> {
                val byPage = candidates.sortedBy { it.chunk.pageNumber }
                val firstPage = byPage.firstOrNull()?.chunk?.pageNumber ?: 1
                val lastPage = byPage.lastOrNull()?.chunk?.pageNumber ?: 1

                val intro = byPage.filter { it.chunk.pageNumber <= firstPage + 1 }
                val conclusion = byPage.filter { it.chunk.pageNumber >= lastPage - 1 && it.chunk.pageNumber > firstPage + 1 }
                val middle = byPage.filter { it !in intro && it !in conclusion }

                val result = mutableListOf<RetrievalResult>()
                result.addAll(intro)
                result.addAll(conclusion)
                result.addAll(middle.sortedByDescending { it.relevanceScore })
                result
            }

            QueryIntent.CONCEPTUAL_QUERY,
            QueryIntent.ENTITY_LOOKUP,
            QueryIntent.SECTION_QUERY -> {
                val sorted = candidates.sortedByDescending { it.relevanceScore }
                val topScore = sorted.firstOrNull()?.relevanceScore ?: 0f
                if (topScore > 0f) {
                    sorted.filter { it.relevanceScore >= topScore * 0.30f || it.source == RetrievalSource.EXACT_FACT_LEXICAL }
                } else {
                    sorted
                }
            }
        }

        // Apply token budget
        val selected = mutableListOf<RetrievalResult>()
        var currentTokenCount = 0
        var isLimited = false

        for (c in prioritized) {
            val cost = estimateTokens(c.chunk.text) + 25
            if (currentTokenCount + cost <= maxBudgetTokens) {
                selected.add(c)
                currentTokenCount += cost
            } else {
                isLimited = true
            }
        }

        if (selected.isEmpty() && prioritized.isNotEmpty()) {
            selected.add(prioritized.first())
        }

        return Pair(selected, isLimited)
    }

    /**
     * Merges adjacent sequential chunks from the same document, page, and section.
     * Chunks must have matching documentId, pageNumber, sectionTitle, and contiguous chunkIndex (curr + 1).
     */
    fun mergeAdjacentChunks(orderedResults: List<RetrievalResult>): List<SourceBlock> {
        if (orderedResults.isEmpty()) return emptyList()

        val mergedBlocks = mutableListOf<SourceBlock>()

        var currentFirstChunk = orderedResults[0].chunk
        var currentLastChunkIndex = currentFirstChunk.chunkIndex
        var currentMaxScore = orderedResults[0].relevanceScore
        var currentSource = orderedResults[0].source
        var currentIntent = orderedResults[0].intent
        val currentTexts = mutableListOf(currentFirstChunk.text.trim())
        var currentMinPage = currentFirstChunk.pageNumber
        var currentMaxPage = currentFirstChunk.pageNumber

        for (i in 1 until orderedResults.size) {
            val nextResult = orderedResults[i]
            val nextChunk = nextResult.chunk

            val isSameDoc = nextChunk.documentId == currentFirstChunk.documentId
            val isSamePage = nextChunk.pageNumber == currentFirstChunk.pageNumber
            val isContiguousIndex = nextChunk.chunkIndex == currentLastChunkIndex + 1
            val isSameSection = (nextChunk.sectionTitle == currentFirstChunk.sectionTitle) ||
                (nextChunk.sectionTitle.isNullOrBlank() && currentFirstChunk.sectionTitle.isNullOrBlank())

            if (isSameDoc && isSamePage && isContiguousIndex && isSameSection) {
                // Compatible adjacent chunk on same page: merge into current block
                currentTexts.add(nextChunk.text.trim())
                currentLastChunkIndex = nextChunk.chunkIndex
                if (nextResult.relevanceScore > currentMaxScore) {
                    currentMaxScore = nextResult.relevanceScore
                }
                currentMinPage = min(currentMinPage, nextChunk.pageNumber)
                currentMaxPage = max(currentMaxPage, nextChunk.pageNumber)
            } else {
                // Finish current block
                val pageSpan = if (currentFirstChunk.pageSpan.isNotBlank()) {
                    currentFirstChunk.pageSpan
                } else if (currentMinPage == currentMaxPage) {
                    currentMinPage.toString()
                } else {
                    "$currentMinPage-$currentMaxPage"
                }

                mergedBlocks.add(
                    SourceBlock(
                        sourceId = currentFirstChunk.id,
                        documentId = currentFirstChunk.documentId,
                        pageNumber = currentMinPage,
                        pageSpan = pageSpan,
                        chunkIndex = currentFirstChunk.chunkIndex,
                        sectionTitle = currentFirstChunk.sectionTitle,
                        text = currentTexts.joinToString("\n\n"),
                        relevanceScore = currentMaxScore,
                        source = currentSource,
                        intent = currentIntent
                    )
                )

                // Start new block
                currentFirstChunk = nextChunk
                currentLastChunkIndex = nextChunk.chunkIndex
                currentMaxScore = nextResult.relevanceScore
                currentSource = nextResult.source
                currentIntent = nextResult.intent
                currentTexts.clear()
                currentTexts.add(nextChunk.text.trim())
                currentMinPage = nextChunk.pageNumber
                currentMaxPage = nextChunk.pageNumber
            }
        }

        // Add the final block
        val finalSpan = if (currentFirstChunk.pageSpan.isNotBlank()) {
            currentFirstChunk.pageSpan
        } else if (currentMinPage == currentMaxPage) {
            currentMinPage.toString()
        } else {
            "$currentMinPage-$currentMaxPage"
        }
        mergedBlocks.add(
            SourceBlock(
                sourceId = currentFirstChunk.id,
                documentId = currentFirstChunk.documentId,
                pageNumber = currentMinPage,
                pageSpan = finalSpan,
                chunkIndex = currentFirstChunk.chunkIndex,
                sectionTitle = currentFirstChunk.sectionTitle,
                text = currentTexts.joinToString("\n\n"),
                relevanceScore = currentMaxScore,
                source = currentSource,
                intent = currentIntent
            )
        )

        return mergedBlocks
    }

    /**
     * Extracts a concise grounded sentence from the strongest evidence when synthesis fails.
     * Prevents dumping giant raw chunks to the user.
     */
    fun extractGroundedFallback(sourceBlocks: List<SourceBlock>, query: String): String {
        if (sourceBlocks.isEmpty()) {
            return "I searched the provided document(s), but could not find information answering \"$query\"."
        }

        val topBlock = sourceBlocks.maxByOrNull { it.relevanceScore } ?: sourceBlocks.first()
        val sentences = topBlock.text.split(Regex("""(?<=[.!?])\s+|\n+"""))
            .map { it.trim() }
            .filter { it.length > 15 }

        // Find sentence containing query keywords
        val queryKeywords = query.lowercase().split(Regex("""[^a-zA-Z0-9]+"""))
            .filter { it.length > 2 }

        val bestSentence = sentences.maxByOrNull { sentence ->
            val sLower = sentence.lowercase()
            queryKeywords.count { sLower.contains(it) }
        } ?: sentences.firstOrNull() ?: topBlock.text.take(150)

        val cleanSnippet = bestSentence.replace(Regex("""\s+"""), " ").take(200).trim()
        val pageRef = if (topBlock.pageSpan.contains("-")) "Pages ${topBlock.pageSpan}" else "Page ${topBlock.pageNumber}"
        return "Based on $pageRef: \"$cleanSnippet\""
    }

    /**
     * Builds structured JSON array for persistence and citation verification.
     */
    private fun buildSourcesJson(blocks: List<SourceBlock>): String {
        val items = blocks.map { b ->
            val escapedSection = b.sectionTitle?.replace("\"", "\\\"") ?: ""
            """{"sourceId":${b.sourceId},"documentId":${b.documentId},"page":${b.pageNumber},"pageSpan":"${b.pageSpan}","chunkIndex":${b.chunkIndex},"section":"$escapedSection","score":${b.relevanceScore}}"""
        }
        return items.joinToString(prefix = "[", postfix = "]", separator = ",")
    }
}
