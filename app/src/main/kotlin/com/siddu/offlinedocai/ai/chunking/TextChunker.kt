package com.siddu.offlinedocai.ai.chunking

import com.siddu.offlinedocai.domain.rag.ExtractedPage
import java.text.BreakIterator
import java.util.Locale

/**
 * Structure-preserving, token-bounded document chunker.
 *
 * 1. Each page is parsed into structural blocks (headings, list items, table rows, paragraphs);
 *    paragraphs that continue across a page break are re-joined.
 * 2. Blocks are packed into chunks measured in **embedding-model tokens** (the same WordPiece
 *    tokenizer the embedder uses), so no chunk is ever silently truncated at embedding time.
 * 3. Oversized blocks are split at sentence boundaries (and, for run-on "sentences", at word
 *    boundaries), never mid-word.
 * 4. Consecutive chunks of the same section share 1–2 trailing sentences of overlap so facts that
 *    straddle a boundary remain retrievable. A new section always starts a new chunk.
 */
object TextChunker {

    /** Bump when chunking behaviour changes; part of the document index signature. */
    const val VERSION = 4

    const val DEFAULT_TARGET_TOKENS = 128
    const val DEFAULT_MAX_TOKENS = 180
    const val DEFAULT_OVERLAP_TOKENS = 32
    private const val HEADING_RESERVE_TOKENS = 24

    val DOT_LEADER = Regex("""(?:\.\s*){3,}\s*\d+|(?:\.\s*){5,}""")
    val TRAILING_PAGE_NUM = Regex("""\s+\d{1,4}$""")
    val TOC_HEADING = Regex("""^(?:table\s+of\s+contents|contents|index|table\s+of\s+figures|list\s+of\s+tables|list\s+of\s+figures)$""", RegexOption.IGNORE_CASE)

    /**
     * Detects structural navigation blocks (Table of Contents, Index, List of Figures) that contain
     * high-density keyword lists with dot leaders or page numbers, which pollute retrieval if indexed.
     */
    fun isTableOfContents(text: String, sectionTitle: String? = null): Boolean {
        val lines = text.lines().map { it.trim() }.filter { it.isNotEmpty() }
        if (lines.isEmpty()) return false
        val dotCount = lines.count { DOT_LEADER.containsMatchIn(it) }
        if (dotCount >= 2) return true
        if (dotCount >= 1 && lines.size <= 3) return true
        val isTocTitle = sectionTitle != null && TOC_HEADING.matches(sectionTitle.trim())
        val trailingPages = lines.count { TRAILING_PAGE_NUM.containsMatchIn(it) }
        if (isTocTitle && (trailingPages.toDouble() / lines.size >= 0.3 || dotCount > 0)) return true
        if (lines.size >= 4 && (trailingPages.toDouble() / lines.size >= 0.5)) return true
        return false
    }

    data class Chunk(
        val index: Int,
        val pageNumber: Int,
        val text: String,
        val pageSpan: String = "$pageNumber",
        val sectionTitle: String? = null,
        val paragraphIndex: Int = 0,
        val tokenCount: Int = 0
    )

    /** Fallback counter (~1.3 WordPiece tokens per word) for callers without a tokenizer. */
    val approximateTokenCounter: (String) -> Int = { text ->
        val words = text.split(WHITESPACE).count { it.isNotEmpty() }
        val punctuation = text.count { !it.isLetterOrDigit() && !it.isWhitespace() }
        ((words * 1.3) + punctuation * 0.5).toInt()
    }

    private enum class BlockType { HEADING, LIST_ITEM, TABLE_ROW, PARAGRAPH }

    private data class Block(
        val startPage: Int,
        val endPage: Int,
        val sectionTitle: String?,
        val type: BlockType,
        val text: String
    )

    /** Smallest packable piece: a whole block, or one sentence (or word window) of an oversized block. */
    private data class Unit(
        val text: String,
        val tokens: Int,
        val startPage: Int,
        val endPage: Int,
        val sectionTitle: String?,
        val type: BlockType,
        val blockId: Int
    )

    private val WHITESPACE = Regex("\\s+")
    private val MARKDOWN_HEADING = Regex("^#{1,6}\\s+\\S")
    private val OUTLINE_HEADING = Regex(
        """^(?:(?:chapter|section|appendix|part)\s+[0-9ivxlcdmIVXLCDM]+[.:]?|[0-9]+(?:\.[0-9]+){0,3}\.?)\s+\p{Lu}[^.!?]{0,70}$""",
        RegexOption.IGNORE_CASE
    )
    private val LABEL_HEADING = Regex("""^[\p{L}0-9][\p{L}0-9\s/&_\-]{1,48}:$""")
    private val LIST_ITEM = Regex("""^(?:[•\-*‣◦⁃∙▪●○]|\d{1,3}[.)]|[a-zA-Z][.)]|\([0-9a-zA-Z]{1,3}\))\s+\S""")

    fun chunkDocument(
        pages: List<ExtractedPage>,
        tokenCounter: (String) -> Int = approximateTokenCounter,
        targetTokens: Int = DEFAULT_TARGET_TOKENS,
        maxTokens: Int = DEFAULT_MAX_TOKENS,
        overlapTokens: Int = DEFAULT_OVERLAP_TOKENS
    ): List<Chunk> {
        require(maxTokens > 0 && targetTokens in 1..maxTokens) { "Invalid chunk token limits" }
        if (pages.isEmpty()) return emptyList()
        val rawBlocks = mergeCrossPageParagraphs(parseBlocks(pages))
        val blocks = rawBlocks.filterNot { isTableOfContents(it.text, it.sectionTitle) }
        // Leave room for a section heading to share the chunk with the unit that follows it.
        val unitLimit = (maxTokens - HEADING_RESERVE_TOKENS).coerceAtLeast(maxTokens / 2)
        val units = toUnits(blocks, tokenCounter, unitLimit)
        return pack(units, tokenCounter, targetTokens, maxTokens, overlapTokens)
    }

    /** Backward-compatible single-page API. */
    fun chunkText(text: String, pageNumber: Int, tokenCounter: (String) -> Int = approximateTokenCounter): List<Chunk> =
        chunkDocument(listOf(ExtractedPage(pageNumber, text)), tokenCounter)

    // ---------------------------------------------------------------------------------------------
    // 1. Structural parsing
    // ---------------------------------------------------------------------------------------------

    private fun parseBlocks(pages: List<ExtractedPage>): List<Block> {
        val blocks = mutableListOf<Block>()
        var section: String? = null

        for (page in pages) {
            val lines = page.text.lines()
            var i = 0
            while (i < lines.size) {
                val line = lines[i].trim()
                if (line.isEmpty()) { i++; continue }

                val prevEmpty = i == 0 || lines[i - 1].isBlank()
                val nextEmpty = i + 1 >= lines.size || lines[i + 1].isBlank()

                if (isHeading(line, prevEmpty, nextEmpty)) {
                    section = line.removePrefix("#").trimStart('#', ' ').trimEnd(':').trim().ifEmpty { line }
                    blocks.add(Block(page.pageNumber, page.pageNumber, section, BlockType.HEADING, line))
                    i++
                    continue
                }

                if (isTableRow(line)) {
                    val rows = mutableListOf<String>()
                    while (i < lines.size && isTableRow(lines[i].trim())) {
                        rows.add(lines[i].trim()); i++
                    }
                    blocks.add(Block(page.pageNumber, page.pageNumber, section, BlockType.TABLE_ROW, rows.joinToString("\n")))
                    continue
                }

                if (LIST_ITEM.containsMatchIn(line)) {
                    val itemLines = mutableListOf(line)
                    i++
                    while (i < lines.size) {
                        val raw = lines[i]
                        val next = raw.trim()
                        if (next.isEmpty() || LIST_ITEM.containsMatchIn(next) || isTableRow(next) ||
                            isHeading(next, prevEmpty = false, nextEmpty = false)) break
                        val indented = raw.startsWith("  ") || raw.startsWith("\t")
                        if (indented || !next[0].isUpperCase()) { itemLines.add(next); i++ } else break
                    }
                    blocks.add(Block(page.pageNumber, page.pageNumber, section, BlockType.LIST_ITEM, joinLines(itemLines)))
                    continue
                }

                val paraLines = mutableListOf<String>()
                while (i < lines.size) {
                    val next = lines[i].trim()
                    if (next.isEmpty()) break
                    if (paraLines.isNotEmpty() &&
                        (isHeading(next, prevEmpty = false, nextEmpty = false) || LIST_ITEM.containsMatchIn(next) || isTableRow(next))
                    ) break
                    paraLines.add(next); i++
                }
                if (paraLines.isNotEmpty()) {
                    blocks.add(Block(page.pageNumber, page.pageNumber, section, BlockType.PARAGRAPH, joinLines(paraLines)))
                }
            }
        }
        return blocks
    }

    private fun mergeCrossPageParagraphs(blocks: List<Block>): List<Block> {
        val merged = mutableListOf<Block>()
        for (block in blocks) {
            val last = merged.lastOrNull()
            if (last != null && last.endPage < block.startPage &&
                last.type == BlockType.PARAGRAPH && block.type == BlockType.PARAGRAPH &&
                continuesSentence(last.text, block.text)
            ) {
                val joined = if (last.text.endsWith("-")) last.text.dropLast(1) + block.text else "${last.text} ${block.text}"
                merged[merged.size - 1] = last.copy(endPage = block.endPage, text = joined)
            } else {
                merged.add(block)
            }
        }
        return merged
    }

    /**
     * Conservative heading detection. A false positive is costly (it splits chunks and mislabels
     * sections), so only clearly heading-shaped lines qualify.
     */
    private fun isHeading(line: String, prevEmpty: Boolean, nextEmpty: Boolean): Boolean {
        if (line.length > 90) return false
        if (MARKDOWN_HEADING.containsMatchIn(line)) return true
        val endsLikeSentence = line.endsWith(".") || line.endsWith(",") || line.endsWith(";")
        if (OUTLINE_HEADING.matches(line) && !endsLikeSentence) return true
        if (LABEL_HEADING.matches(line) && line.split(WHITESPACE).size <= 6) return true

        val letters = line.count { it.isLetter() }
        if (letters >= 3 && line.length <= 60 && !endsLikeSentence && line.none { it.isLowerCase() } &&
            (prevEmpty || nextEmpty)
        ) return true

        // Short Title Case line standing on its own between blank lines.
        val words = line.split(WHITESPACE)
        val titleCase = words.size in 1..6 && words.all { w -> w.first().isUpperCase() || !w.first().isLetter() || w.length <= 3 }
        return titleCase && prevEmpty && nextEmpty && !endsLikeSentence && line.length <= 50 &&
            words.first().first().isUpperCase()
    }

    private fun isTableRow(line: String): Boolean =
        (line.startsWith("|") && line.endsWith("|") && line.length > 2) || line.count { it == '\t' } >= 2

    private fun joinLines(lines: List<String>): String {
        if (lines.size == 1) return lines[0]
        val sb = StringBuilder(lines[0])
        for (idx in 1 until lines.size) {
            val prev = lines[idx - 1]
            val curr = lines[idx]
            when {
                prev.endsWith("-") && curr.firstOrNull()?.isLowerCase() == true -> {
                    sb.setLength(sb.length - 1); sb.append(curr)
                }
                // Hard-wrapped prose: previous line does not end a sentence, or is a long full-width line.
                (!prev.endsWithAny(".", ":", ";", "!", "?") && curr.firstOrNull()?.isLowerCase() == true) ||
                    (prev.length > 45 && !prev.endsWith(":")) -> sb.append(' ').append(curr)
                else -> sb.append('\n').append(curr)
            }
        }
        return sb.toString()
    }

    private fun continuesSentence(prev: String, next: String): Boolean {
        val p = prev.trimEnd()
        val n = next.trimStart()
        if (p.isEmpty() || n.isEmpty()) return false
        if (p.endsWith("-") && n[0].isLowerCase()) return true
        return !p.endsWithAny(".", "?", "!", ":", ";", "|") && n[0].isLowerCase()
    }

    // ---------------------------------------------------------------------------------------------
    // 2. Units (token-measured)
    // ---------------------------------------------------------------------------------------------

    private fun toUnits(blocks: List<Block>, count: (String) -> Int, maxTokens: Int): List<Unit> {
        val units = mutableListOf<Unit>()
        blocks.forEachIndexed { id, block ->
            val tokens = count(block.text)
            if (tokens <= maxTokens) {
                units.add(Unit(block.text, tokens, block.startPage, block.endPage, block.sectionTitle, block.type, id))
                return@forEachIndexed
            }
            val pieces = if (block.type == BlockType.TABLE_ROW || block.type == BlockType.LIST_ITEM) {
                block.text.lines().filter { it.isNotBlank() }
            } else {
                splitIntoSentences(block.text)
            }
            for (piece in pieces) {
                val pieceTokens = count(piece)
                if (pieceTokens <= maxTokens) {
                    units.add(Unit(piece, pieceTokens, block.startPage, block.endPage, block.sectionTitle, block.type, id))
                } else {
                    for (window in splitByWords(piece, count, maxTokens)) {
                        units.add(Unit(window, count(window), block.startPage, block.endPage, block.sectionTitle, block.type, id))
                    }
                }
            }
        }
        return units
    }

    private fun splitByWords(text: String, count: (String) -> Int, maxTokens: Int): List<String> {
        val words = text.split(WHITESPACE).filter { it.isNotEmpty() }
        val windows = mutableListOf<String>()
        val current = mutableListOf<String>()
        // WordPiece splits on whitespace first, so a window's token count is the sum of its words' counts:
        // count each word once instead of re-tokenizing the growing window (which was quadratic).
        var running = 0
        for (word in words) {
            val t = count(word)
            if (current.isNotEmpty() && running + t > maxTokens) {
                windows.add(current.joinToString(" "))
                current.clear()
                running = 0
            }
            current.add(word)
            running += t
        }
        if (current.isNotEmpty()) windows.add(current.joinToString(" "))
        return windows
    }

    // ---------------------------------------------------------------------------------------------
    // 3. Packing with overlap
    // ---------------------------------------------------------------------------------------------

    private fun pack(
        units: List<Unit>,
        count: (String) -> Int,
        targetTokens: Int,
        maxTokens: Int,
        overlapTokens: Int
    ): List<Chunk> {
        val chunks = mutableListOf<Chunk>()
        var overlap: List<String> = emptyList()       // carried sentences from the previous chunk
        var overlapPage = 0                            // page those sentences were printed on
        val current = mutableListOf<Unit>()            // new content of the chunk being built
        var currentTokens = 0
        var paragraphIndex = 0

        fun render(): String {
            val sb = StringBuilder()
            if (overlap.isNotEmpty()) sb.append(overlap.joinToString(" "))
            for ((i, u) in current.withIndex()) {
                if (sb.isNotEmpty()) {
                    val prev = current.getOrNull(i - 1)
                    val sameBlock = prev != null && prev.blockId == u.blockId
                    sb.append(
                        when {
                            sameBlock && u.type == BlockType.PARAGRAPH -> " "
                            sameBlock -> "\n"
                            else -> "\n\n"
                        }
                    )
                }
                sb.append(u.text)
            }
            return sb.toString().trim()
        }

        fun emit(carryOverlap: Boolean) {
            if (current.isEmpty()) return
            val text = render()
            val first = current.first()
            // Carried-over sentences come from the previous chunk's last page; include it in the span.
            val startPage = if (overlap.isNotEmpty()) minOf(overlapPage, first.startPage) else first.startPage
            val endPage = current.maxOf { it.endPage }
            val section = current.firstNotNullOfOrNull { it.sectionTitle }
            chunks.add(
                Chunk(
                    index = chunks.size,
                    pageNumber = startPage,
                    text = text,
                    pageSpan = if (startPage == endPage) "$startPage" else "$startPage-$endPage",
                    sectionTitle = section,
                    paragraphIndex = paragraphIndex,
                    tokenCount = count(text)
                )
            )
            paragraphIndex += current.map { it.blockId }.distinct().size
            val last = current.last()
            overlapPage = last.endPage
            overlap = if (carryOverlap && overlapTokens > 0 && last.type != BlockType.HEADING && last.type != BlockType.TABLE_ROW) {
                trailingSentences(current, count, overlapTokens)
            } else emptyList()
            current.clear()
            currentTokens = overlap.sumOf { count(it) }
        }

        for (unit in units) {
            if (unit.type == BlockType.HEADING && current.any { it.type != BlockType.HEADING }) {
                emit(carryOverlap = false) // a new section never inherits the previous section's tail
                overlap = emptyList(); currentTokens = 0
            }
            val sectionChanged = current.isEmpty() && overlap.isNotEmpty() &&
                unit.sectionTitle != null && chunks.lastOrNull()?.sectionTitle != unit.sectionTitle
            if (sectionChanged) { overlap = emptyList(); currentTokens = 0 }

            val separatorCost = if (current.isEmpty() && overlap.isEmpty()) 0 else 1
            val hasContent = current.any { it.type != BlockType.HEADING }
            if (hasContent && currentTokens + separatorCost + unit.tokens > targetTokens) {
                emit(carryOverlap = true)
            } else if (!hasContent && current.isNotEmpty() && currentTokens + separatorCost + unit.tokens > maxTokens) {
                emit(carryOverlap = false) // rare: heading(s) too long to share a chunk with the next unit
            }
            if (currentTokens + unit.tokens > maxTokens) {
                overlap = emptyList(); currentTokens = 0 // overlap never pushes a chunk past the hard limit
            }
            current.add(unit)
            currentTokens += unit.tokens + if (currentTokens > 0) 1 else 0
        }
        emit(carryOverlap = false)
        return chunks
    }

    private fun trailingSentences(units: List<Unit>, count: (String) -> Int, budget: Int): List<String> {
        val tail = units.last { it.type != BlockType.HEADING }
        val sentences = splitIntoSentences(tail.text)
        val picked = ArrayDeque<String>()
        var used = 0
        for (sentence in sentences.asReversed()) {
            val t = count(sentence)
            if (used + t > budget || picked.size >= 2) break
            picked.addFirst(sentence)
            used += t
        }
        return picked.toList()
    }

    // ---------------------------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------------------------

    fun splitIntoSentences(text: String): List<String> {
        val clean = text.trim()
        if (clean.isEmpty()) return emptyList()
        val iterator = BreakIterator.getSentenceInstance(Locale.US)
        iterator.setText(clean)
        val raw = mutableListOf<String>()
        var start = iterator.first()
        var end = iterator.next()
        while (end != BreakIterator.DONE) {
            clean.substring(start, end).trim().takeIf { it.isNotEmpty() }?.let { raw.add(it) }
            start = end
            end = iterator.next()
        }
        // Re-join splits that happened after common abbreviations ("e.g. the", "Dr. Smith").
        val merged = mutableListOf<String>()
        for (sentence in raw) {
            val prev = merged.lastOrNull()
            val lastToken = prev?.substringAfterLast(' ')?.lowercase()
            if (prev != null && lastToken != null && lastToken in ABBREVIATIONS) {
                merged[merged.size - 1] = "$prev $sentence"
            } else {
                merged.add(sentence)
            }
        }
        return merged.ifEmpty { listOf(clean) }
    }

    private val ABBREVIATIONS = setOf(
        "dr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "jr.", "e.g.", "i.e.", "vs.", "etc.", "fig.", "no.", "approx."
    )

    private fun String.endsWithAny(vararg suffixes: String) = suffixes.any { endsWith(it) }
}
