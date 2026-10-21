package com.siddu.offlinedocai.ai.retrieval

import com.siddu.offlinedocai.ai.chunking.TextChunker
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.ceil

/**
 * One numbered evidence block given to the LLM. [citationTag] is the exact string the model is
 * instructed to copy when citing, so context labels and requested citation format are identical.
 */
data class SourceBlock(
    val number: Int,
    val documentId: Long,
    val documentName: String?,
    val pageNumber: Int,
    val pageSpan: String,
    val sectionTitle: String?,
    val text: String,
    val chunkIds: List<Long>,
    val relevanceScore: Float,
    val source: RetrievalSource
) {
    val citationTag: String get() = "[Source $number, Page ${pageSpan.ifBlank { pageNumber.toString() }}]"

    fun format(): String {
        val meta = buildList {
            documentName?.takeIf { it.isNotBlank() }?.let { add("Document: $it") }
            sectionTitle?.takeIf { it.isNotBlank() }?.let { add("Section: $it") }
        }
        val header = if (meta.isEmpty()) citationTag else "$citationTag (${meta.joinToString(" | ")})"
        return "$header\n${text.trim()}"
    }
}

data class AssembledContext(
    val sourceBlocks: List<SourceBlock>,
    val contextText: String,
    val usedTokens: Int,
    val budgetTokens: Int,
    /** True when relevant evidence was left out because it did not fit the budget. */
    val isBudgetLimited: Boolean,
    val intent: QueryIntent,
    val sourcesJson: String
) {
    val isEmpty: Boolean get() = sourceBlocks.isEmpty()
}

/**
 * Turns ranked retrieval results into the evidence section of the prompt:
 *  1. de-duplicates,
 *  2. greedily packs results in priority order into the token budget (every block's header is
 *     counted, and later smaller blocks may still fit after a large one is skipped),
 *  3. restores document order and merges consecutive chunks, removing their shared overlap,
 *  4. numbers the blocks; for open questions the most relevant block comes first.
 */
object ContextAssembler {

    /** Separator between blocks plus the header line, measured once with a worst-case header. */
    private const val WORST_CASE_HEADER = "[Source 99, Page 9999-9999] (Document: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx | Section: xxxxxxxxxxxxxxxxxxxxxxxx)\n\n\n"

    /** Conservative fallback when the LLM tokenizer is unavailable (~3.2 chars per token for mixed text). */
    fun estimateTokens(text: String): Int = if (text.isEmpty()) 0 else ceil(text.length / 3.2).toInt()

    fun assemble(
        results: List<RetrievalResult>,
        classification: QueryClassification,
        budgetTokens: Int,
        tokenCounter: (String) -> Int = ::estimateTokens,
        documentNames: Map<Long, String> = emptyMap()
    ): AssembledContext {
        val intent = classification.intent
        if (results.isEmpty() || budgetTokens <= 0) return empty(intent, budgetTokens)

        // 1. De-duplicate (first occurrence = highest priority wins).
        val unique = LinkedHashMap<Long, RetrievalResult>()
        for (r in results) unique.putIfAbsent(r.chunk.id, r)

        // 2. Greedy packing in priority order (with fair-share representation for sub-queries).
        val headerCost = tokenCounter(WORST_CASE_HEADER)
        val selected = mutableListOf<RetrievalResult>()
        val selectedIds = mutableSetOf<Long>()
        var used = 0
        var skippedRelevant = false

        // Predecessor companion lookup map: allow high-priority chunks to bring along their immediate
        // predecessor/successor chunks from the candidate pool to complete boundaries and headers.
        val candidatesByPosition = unique.values.associateBy { it.chunk.documentId to it.chunk.chunkIndex }

        // Fair-share reservation: when multiple sub-queries exist (e.g. comparison sides or clauses),
        // guarantee balanced representation so neither side gets crowded out by the other.
        if (classification.subQueries.size >= 2) {
            val perSubQueryTarget = maxOf(2, (budgetTokens / classification.subQueries.size) / 350)
            for (sq in classification.subQueries) {
                val sqTerms = TextAnalyzer.terms(sq).toSet()
                if (sqTerms.isEmpty()) continue
                val matches = unique.values
                    .filter { it.chunk.id !in selectedIds && it.source != RetrievalSource.NEIGHBOR }
                    .map { r ->
                        val chunkTerms = TextAnalyzer.terms(listOfNotNull(r.chunk.sectionTitle, r.chunk.text).joinToString("\n")).toSet()
                        val overlap = sqTerms.count { it in chunkTerms }
                        r to overlap
                    }
                    .filter { it.second > 0 }
                    .sortedWith(compareByDescending<Pair<RetrievalResult, Int>> { it.second }.thenByDescending { it.first.relevanceScore })

                var count = 0
                for ((cand, _) in matches) {
                    if (count >= perSubQueryTarget) break
                    if (cand.chunk.id in selectedIds) continue
                    val cost = tokenCounter(cand.chunk.text) + headerCost
                    if (used + cost <= budgetTokens) {
                        selected.add(cand)
                        selectedIds.add(cand.chunk.id)
                        used += cost
                        count++

                        // Pull predecessor if present to restore missing headers/speaker/context
                        val predecessor = candidatesByPosition[cand.chunk.documentId to cand.chunk.chunkIndex - 1]
                        if (predecessor != null && predecessor.chunk.id !in selectedIds) {
                            val predCost = tokenCounter(predecessor.chunk.text) + headerCost
                            if (used + predCost <= budgetTokens) {
                                selected.add(predecessor)
                                selectedIds.add(predecessor.chunk.id)
                                used += predCost
                            }
                        }
                    }
                }
            }
        }

        for (r in unique.values) {
            if (r.chunk.id in selectedIds) continue
            val cost = tokenCounter(r.chunk.text) + headerCost
            if (used + cost <= budgetTokens) {
                selected.add(r)
                selectedIds.add(r.chunk.id)
                used += cost

                // When pulling primary hits, prioritize pulling their predecessor neighbor if available
                if (r.source != RetrievalSource.NEIGHBOR) {
                    val predecessor = candidatesByPosition[r.chunk.documentId to r.chunk.chunkIndex - 1]
                    if (predecessor != null && predecessor.chunk.id !in selectedIds) {
                        val predCost = tokenCounter(predecessor.chunk.text) + headerCost
                        if (used + predCost <= budgetTokens) {
                            selected.add(predecessor)
                            selectedIds.add(predecessor.chunk.id)
                            used += predCost
                        }
                    }
                }
            } else if (r.source != RetrievalSource.NEIGHBOR) {
                skippedRelevant = true
            }
        }
        if (selected.isEmpty()) return empty(intent, budgetTokens).copy(isBudgetLimited = true)

        // 3. Document order + merge consecutive chunks of the same document.
        val ordered = selected.sortedWith(compareBy({ it.chunk.documentId }, { it.chunk.chunkIndex }))
        val groups = mutableListOf<MutableList<RetrievalResult>>()
        for (r in ordered) {
            val last = groups.lastOrNull()?.last()
            if (last != null && last.chunk.documentId == r.chunk.documentId && r.chunk.chunkIndex == last.chunk.chunkIndex + 1) {
                groups.last().add(r)
            } else {
                groups.add(mutableListOf(r))
            }
        }

        // 4. Block order: open questions (GENERAL) lead with the best evidence;
        // sequential, comparison, and structural intents read in natural document order so cause-and-effect
        // and timelines flow chronologically.
        val blockOrder = if (intent == QueryIntent.GENERAL) {
            groups.sortedByDescending { g -> g.maxOf { it.relevanceScore } }
        } else groups

        val blocks = blockOrder.mapIndexed { index, group -> toBlock(index + 1, group, documentNames) }
        val contextText = blocks.joinToString("\n\n") { it.format() }
        return AssembledContext(
            sourceBlocks = blocks,
            contextText = contextText,
            usedTokens = tokenCounter(contextText),
            budgetTokens = budgetTokens,
            isBudgetLimited = skippedRelevant,
            intent = intent,
            sourcesJson = sourcesJson(blocks)
        )
    }

    private fun toBlock(number: Int, group: List<RetrievalResult>, names: Map<Long, String>): SourceBlock {
        val first = group.first().chunk
        val text = group.drop(1).fold(first.text.trim()) { acc, r -> joinRemovingOverlap(acc, r.chunk.text.trim()) }
        val pages = group.flatMap { pageBounds(it.chunk.pageSpan, it.chunk.pageNumber) }
        val startPage = pages.minOrNull() ?: first.pageNumber
        val endPage = pages.maxOrNull() ?: first.pageNumber
        return SourceBlock(
            number = number,
            documentId = first.documentId,
            documentName = names[first.documentId],
            pageNumber = startPage,
            pageSpan = if (startPage == endPage) "$startPage" else "$startPage-$endPage",
            sectionTitle = group.firstNotNullOfOrNull { it.chunk.sectionTitle },
            text = text,
            chunkIds = group.map { it.chunk.id },
            relevanceScore = group.maxOf { it.relevanceScore },
            source = group.first().source
        )
    }

    /**
     * Consecutive chunks share 1–2 overlap sentences (see [TextChunker]). Drop the leading sentences
     * of [next] that already appear at the end of [previous] so the model never reads them twice.
     */
    fun joinRemovingOverlap(previous: String, next: String): String {
        val tail = normalizeSpace(previous.takeLast(1200))
        val sentences = TextChunker.splitIntoSentences(next)
        var drop = 0
        while (drop < sentences.size && drop < 3) {
            val s = normalizeSpace(sentences[drop])
            if (s.length >= 12 && tail.contains(s)) drop++ else break
        }
        if (drop == 0) return "$previous\n\n$next"
        val remainder = sentences.drop(drop).joinToString(" ")
        return if (remainder.isBlank()) previous else "$previous $remainder"
    }

    private fun normalizeSpace(s: String) = s.replace(Regex("\\s+"), " ").trim()

    private fun pageBounds(pageSpan: String, fallback: Int): List<Int> {
        val parts = pageSpan.split("-").mapNotNull { it.trim().toIntOrNull() }
        return if (parts.isEmpty()) listOf(fallback) else parts
    }

    private fun sourcesJson(blocks: List<SourceBlock>): String {
        val array = JSONArray()
        for (b in blocks) {
            array.put(
                JSONObject()
                    .put("source", b.number)
                    .put("documentId", b.documentId)
                    .put("documentName", b.documentName ?: JSONObject.NULL)
                    .put("page", b.pageNumber)
                    .put("pageSpan", b.pageSpan)
                    .put("section", b.sectionTitle ?: JSONObject.NULL)
                    .put("chunkIds", JSONArray(b.chunkIds))
                    .put("score", b.relevanceScore.toDouble())
                    .put("text", b.text)
            )
        }
        return array.toString()
    }

    private fun empty(intent: QueryIntent, budget: Int) =
        AssembledContext(emptyList(), "", 0, budget, isBudgetLimited = false, intent = intent, sourcesJson = "[]")
}
