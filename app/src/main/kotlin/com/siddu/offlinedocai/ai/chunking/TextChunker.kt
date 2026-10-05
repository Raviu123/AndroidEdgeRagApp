package com.siddu.offlinedocai.ai.chunking

import com.siddu.offlinedocai.domain.rag.ExtractedPage

/**
 * Structure-Preserving Hierarchical Document Chunker.
 * Preserves paragraphs, section headings, bullet lists, numbered items, table rows,
 * and individual line boundaries without flattening text into a single run-on line.
 */
object TextChunker {

    data class Chunk(
        val index: Int,
        val pageNumber: Int,
        val text: String,
        val pageSpan: String = "$pageNumber",
        val sectionTitle: String? = null,
        val paragraphIndex: Int = 0
    )

    private enum class BlockType {
        HEADING,
        LIST_ITEM,
        TABLE_ROW,
        PARAGRAPH
    }

    private data class SemanticBlock(
        val startPage: Int,
        val endPage: Int,
        val sectionTitle: String?,
        val blockType: BlockType,
        val text: String,
        val followedByEmptyLine: Boolean = true
    )

    /**
     * Chunk a full multi-page document while preserving structure, headings, lists, tables,
     * stable document ordering, and cross-page semantic continuity.
     *
     * @param pages list of extracted pages in chronological document order.
     * @param targetWords target word count per retrieval chunk (~350 words).
     * @param maxWords maximum word threshold before splitting at sentence boundaries (~500 words).
     * @param overlapWords word overlap for semantic continuity across chunk boundaries (~60 words).
     */
    fun chunkDocument(
        pages: List<ExtractedPage>,
        targetWords: Int = 350,
        maxWords: Int = 500,
        overlapWords: Int = 60
    ): List<Chunk> {
        if (pages.isEmpty()) return emptyList()

        // 1. Parse each page into structural semantic blocks
        val rawBlocks = mutableListOf<SemanticBlock>()
        var currentSectionTitle: String? = null

        for (page in pages) {
            val pageLines = page.text.lines()
            var i = 0

            while (i < pageLines.size) {
                val rawLine = pageLines[i]
                val trimmed = rawLine.trim()

                if (trimmed.isEmpty()) {
                    i++
                    continue
                }

                val prevLineEmpty = (i == 0) || pageLines[i - 1].trim().isEmpty()
                val nextLineEmpty = (i + 1 >= pageLines.size) || pageLines[i + 1].trim().isEmpty()

                // Check for generic structural heading
                val headingTitle = detectHeading(trimmed, prevLineEmpty, nextLineEmpty)
                if (headingTitle != null) {
                    currentSectionTitle = headingTitle
                    rawBlocks.add(
                        SemanticBlock(
                            startPage = page.pageNumber,
                            endPage = page.pageNumber,
                            sectionTitle = currentSectionTitle,
                            blockType = BlockType.HEADING,
                            text = trimmed,
                            followedByEmptyLine = nextLineEmpty
                        )
                    )
                    i++
                    continue
                }

                // Check for table row (| col1 | col2 | or tab-separated)
                if (isTableRow(trimmed)) {
                    val tableLines = mutableListOf<String>()
                    while (i < pageLines.size && isTableRow(pageLines[i].trim())) {
                        tableLines.add(pageLines[i].trim())
                        i++
                    }
                    val tableNextEmpty = (i >= pageLines.size) || pageLines[i].trim().isEmpty()
                    rawBlocks.add(
                        SemanticBlock(
                            startPage = page.pageNumber,
                            endPage = page.pageNumber,
                            sectionTitle = currentSectionTitle,
                            blockType = BlockType.TABLE_ROW,
                            text = tableLines.joinToString("\n"),
                            followedByEmptyLine = tableNextEmpty
                        )
                    )
                    continue
                }

                // Check for list item (bullet or numbered)
                if (isListItem(trimmed)) {
                    val listLines = mutableListOf<String>()
                    listLines.add(trimmed)
                    i++
                    // Collect following indented lines or continuation of list item
                    while (i < pageLines.size) {
                        val nextTrim = pageLines[i].trim()
                        if (nextTrim.isEmpty()) break
                        if (isListItem(nextTrim) || isTableRow(nextTrim) || detectHeading(nextTrim, prevLineEmpty = false, nextLineEmpty = false) != null) {
                            break
                        }
                        if (pageLines[i].startsWith("  ") || pageLines[i].startsWith("\t") || !nextTrim[0].isUpperCase()) {
                            listLines.add(nextTrim)
                            i++
                        } else {
                            break
                        }
                    }
                    val listNextEmpty = (i >= pageLines.size) || pageLines[i].trim().isEmpty()
                    rawBlocks.add(
                        SemanticBlock(
                            startPage = page.pageNumber,
                            endPage = page.pageNumber,
                            sectionTitle = currentSectionTitle,
                            blockType = BlockType.LIST_ITEM,
                            text = listLines.joinToString("\n"),
                            followedByEmptyLine = listNextEmpty
                        )
                    )
                    continue
                }

                // Standard paragraph / line group: accumulate until blank line, heading, list, or table
                val paraLines = mutableListOf<String>()
                while (i < pageLines.size) {
                    val pLine = pageLines[i].trim()
                    if (pLine.isEmpty()) break
                    if (detectHeading(pLine, prevLineEmpty = false, nextLineEmpty = false) != null || isListItem(pLine) || isTableRow(pLine)) {
                        break
                    }
                    paraLines.add(pLine)
                    i++
                }

                val paraNextEmpty = (i >= pageLines.size) || pageLines[i].trim().isEmpty()
                if (paraLines.isNotEmpty()) {
                    rawBlocks.add(
                        SemanticBlock(
                            startPage = page.pageNumber,
                            endPage = page.pageNumber,
                            sectionTitle = currentSectionTitle,
                            blockType = BlockType.PARAGRAPH,
                            text = joinParagraphLines(paraLines),
                            followedByEmptyLine = paraNextEmpty
                        )
                    )
                }
            }
        }

        // 2. Cross-Page Semantic Continuity: merge blocks across page transitions if sentences continue
        val mergedBlocks = mutableListOf<SemanticBlock>()
        var bIdx = 0
        while (bIdx < rawBlocks.size) {
            val current = rawBlocks[bIdx]
            if (bIdx + 1 < rawBlocks.size) {
                val next = rawBlocks[bIdx + 1]
                val isCrossPageContinuation = current.endPage < next.startPage &&
                    current.blockType == BlockType.PARAGRAPH &&
                    next.blockType == BlockType.PARAGRAPH &&
                    isSentenceContinuation(current.text, next.text)

                if (isCrossPageContinuation) {
                    val combinedText = if (current.text.endsWith("-")) {
                        current.text.removeSuffix("-") + next.text
                    } else {
                        "${current.text} ${next.text}"
                    }
                    mergedBlocks.add(
                        SemanticBlock(
                            startPage = current.startPage,
                            endPage = next.endPage,
                            sectionTitle = current.sectionTitle ?: next.sectionTitle,
                            blockType = BlockType.PARAGRAPH,
                            text = combinedText,
                            followedByEmptyLine = next.followedByEmptyLine
                        )
                    )
                    bIdx += 2
                    continue
                }
            }
            mergedBlocks.add(current)
            bIdx++
        }

        // 3. Assemble Semantic Blocks into Structure-Preserving Retrieval Chunks
        val chunks = mutableListOf<Chunk>()
        var currentChunkBlocks = mutableListOf<SemanticBlock>()
        var currentWordCount = 0
        var globalChunkIndex = 0
        var paragraphIndex = 0

        fun emitChunk(blocks: List<SemanticBlock>) {
            if (blocks.isEmpty()) return

            val firstBlock = blocks.first()
            val lastBlock = blocks.last()
            val startPage = firstBlock.startPage
            val endPage = lastBlock.endPage
            val pageSpan = if (startPage == endPage) "$startPage" else "$startPage-$endPage"
            val section = blocks.mapNotNull { it.sectionTitle }.firstOrNull()

            // Format chunk text preserving block types and line breaks
            val textBuilder = StringBuilder()
            for (j in blocks.indices) {
                val b = blocks[j]
                if (j > 0) {
                    val prev = blocks[j - 1]
                    if (prev.followedByEmptyLine) {
                        textBuilder.append("\n\n")
                    } else {
                        textBuilder.append("\n")
                    }
                }
                textBuilder.append(b.text)
            }

            val chunkText = textBuilder.toString().trim()
            if (chunkText.isNotEmpty()) {
                chunks.add(
                    Chunk(
                        index = globalChunkIndex++,
                        pageNumber = startPage,
                        text = chunkText,
                        pageSpan = pageSpan,
                        sectionTitle = section,
                        paragraphIndex = paragraphIndex
                    )
                )
                paragraphIndex += blocks.count { it.blockType == BlockType.PARAGRAPH }
            }
        }

        for (block in mergedBlocks) {
            val blockWords = countWords(block.text)

            // If a single block is huge, split at sentence boundaries
            if (blockWords > maxWords) {
                if (currentChunkBlocks.isNotEmpty()) {
                    emitChunk(currentChunkBlocks)
                    currentChunkBlocks = mutableListOf()
                    currentWordCount = 0
                }

                val sentenceSplits = splitIntoSentences(block.text)
                var sentAcc = mutableListOf<String>()
                var sentWords = 0

                for (sent in sentenceSplits) {
                    val sw = countWords(sent)
                    if (sentWords + sw > maxWords && sentAcc.isNotEmpty()) {
                        val splitBlock = block.copy(text = sentAcc.joinToString(" "))
                        emitChunk(listOf(splitBlock))
                        sentAcc = mutableListOf()
                        sentWords = 0
                    }
                    sentAcc.add(sent)
                    sentWords += sw
                }
                if (sentAcc.isNotEmpty()) {
                    val splitBlock = block.copy(text = sentAcc.joinToString(" "))
                    currentChunkBlocks.add(splitBlock)
                    currentWordCount = sentWords
                }
                continue
            }

            if (currentWordCount + blockWords > targetWords && currentChunkBlocks.isNotEmpty()) {
                emitChunk(currentChunkBlocks)

                // Overlap: keep the last block if it provides useful context within budget
                val lastBlock = currentChunkBlocks.lastOrNull()
                currentChunkBlocks = mutableListOf()
                currentWordCount = 0

                if (lastBlock != null && countWords(lastBlock.text) <= overlapWords && lastBlock.blockType != BlockType.HEADING) {
                    currentChunkBlocks.add(lastBlock)
                    currentWordCount = countWords(lastBlock.text)
                }
            }

            currentChunkBlocks.add(block)
            currentWordCount += blockWords
        }

        if (currentChunkBlocks.isNotEmpty()) {
            emitChunk(currentChunkBlocks)
        }

        return chunks
    }

    /**
     * Backward-compatible API for single-page chunking.
     * Preserves structure, line breaks, and lists instead of flat-collapsing whitespace.
     */
    fun chunkText(
        text: String,
        pageNumber: Int,
        chunkSize: Int = 400,
        overlap: Int = 80
    ): List<Chunk> {
        val extractedPage = ExtractedPage(pageNumber = pageNumber, text = text)
        return chunkDocument(
            pages = listOf(extractedPage),
            targetWords = chunkSize,
            maxWords = (chunkSize * 1.25).toInt(),
            overlapWords = overlap
        )
    }

    private fun detectHeading(line: String, prevLineEmpty: Boolean, nextLineEmpty: Boolean): String? {
        val trimmed = line.trim()
        if (trimmed.isEmpty() || trimmed.length > 90) return null

        // 1. Markdown heading: "# Section", "## Section"
        if (trimmed.startsWith("#")) {
            return trimmed.replace(Regex("^#+\\s*"), "").trim()
        }

        // 2. Numbered outline / Chapter / Section
        val outlineMatch = Regex(
            """^(?:(?:chapter|section|appendix|part)\s+[0-9a-zA-Z\.]+|[0-9]+(?:\.[0-9]+)+|[A-Z]\.[0-9]+)\s*[:\-]?\s*(.*)$""",
            RegexOption.IGNORE_CASE
        ).find(trimmed)
        if (outlineMatch != null) {
            val titlePart = outlineMatch.groupValues[1].trim()
            return if (titlePart.isNotEmpty()) trimmed else outlineMatch.value.trim()
        }

        // 3. Category label ending with colon: "Languages:", "Projects:", "Requirements:"
        val colonMatch = Regex("""^([A-Za-z0-9\s/&_\-]{3,50}):$""").find(trimmed)
        if (colonMatch != null) {
            return colonMatch.groupValues[1].trim()
        }

        // 4. Isolated short uppercase line (e.g. TECHNICAL SKILLS, EDUCATION)
        val isAllUpper = trimmed.length in 3..60 && trimmed.all { it.isUpperCase() || !it.isLetter() } && trimmed.any { it.isLetter() }
        val endsWithPunctuation = trimmed.endsWith(".") || trimmed.endsWith(",") || trimmed.endsWith(";")
        if (isAllUpper && !endsWithPunctuation && (prevLineEmpty || trimmed.length <= 40)) {
            return trimmed
        }

        // 5. Short title-case phrase isolated by newlines
        val words = trimmed.split(Regex("\\s+"))
        val isTitleCase = words.size in 1..6 && words.all { it.isNotEmpty() && (it[0].isUpperCase() || it.all { c -> !c.isLetter() }) }
        if (isTitleCase && prevLineEmpty && !endsWithPunctuation && trimmed.length <= 45) {
            return trimmed
        }

        return null
    }

    private fun isTableRow(line: String): Boolean {
        return (line.startsWith("|") && line.endsWith("|")) ||
            (line.contains("|") && line.count { it == '|' } >= 2) ||
            (line.contains("\t") && line.count { it == '\t' } >= 2)
    }

    private fun isListItem(line: String): Boolean {
        val trimmed = line.trim()
        return Regex("""^(?:[•\-\*‣◦⁃∙]|\d+[\.\)]|[a-zA-Z][\.\)]|\([0-9a-zA-Z]\))\s+""").containsMatchIn(trimmed)
    }

    private fun joinParagraphLines(lines: List<String>): String {
        if (lines.isEmpty()) return ""
        if (lines.size == 1) return lines[0].trim()

        val sb = StringBuilder()
        for (idx in lines.indices) {
            val line = lines[idx]
            if (idx > 0) {
                val prevLine = lines[idx - 1].trim()
                val currLine = line.trim()

                // 1. Soft-hyphen wrap: previous line ends with hyphen, current starts with lowercase
                if (prevLine.endsWith("-") && currLine.isNotEmpty() && currLine[0].isLowerCase()) {
                    sb.setLength(sb.length - 1)
                    sb.append(currLine)
                    continue
                }

                // 2. Prose sentence continuation:
                // Previous line does not end with colon/terminal punctuation AND current starts with lowercase
                // OR previous line is long full-width line (> 45 chars) in running paragraph
                val isProseContinuation = (!prevLine.endsWith(".") && !prevLine.endsWith(":") &&
                    !prevLine.endsWith(";") && !prevLine.endsWith("!") && !prevLine.endsWith("?") &&
                    currLine.isNotEmpty() && currLine[0].isLowerCase()) ||
                    (prevLine.length > 45 && !prevLine.endsWith(":") && !prevLine.endsWith("|"))

                if (isProseContinuation) {
                    sb.append(" ").append(currLine)
                } else {
                    // Short line, list element, item, or key-value line -> PRESERVE NEWLINE
                    sb.append("\n").append(currLine)
                }
            } else {
                sb.append(line.trim())
            }
        }
        return sb.toString().trim()
    }

    private fun isSentenceContinuation(prevText: String, nextText: String): Boolean {
        val pTrim = prevText.trim()
        val nTrim = nextText.trim()
        if (pTrim.isEmpty() || nTrim.isEmpty()) return false

        // Dangle hyphen word continuation (e.g. "configu-", "ration")
        if (pTrim.endsWith("-") && nTrim[0].isLowerCase()) return true

        // Previous text does not end with terminal punctuation
        val endsWithTerminal = pTrim.endsWith(".") || pTrim.endsWith("?") || pTrim.endsWith("!") ||
            pTrim.endsWith(":") || pTrim.endsWith("|") || pTrim.endsWith(";")

        return !endsWithTerminal && nTrim[0].isLowerCase()
    }

    fun splitIntoSentences(text: String): List<String> {
        val clean = text.trim()
        if (clean.isEmpty()) return emptyList()

        val iterator = java.text.BreakIterator.getSentenceInstance(java.util.Locale.US)
        iterator.setText(clean)

        val rawSentences = mutableListOf<String>()
        var start = iterator.first()
        var end = iterator.next()

        while (end != java.text.BreakIterator.DONE) {
            val sentence = clean.substring(start, end).trim()
            if (sentence.isNotEmpty()) {
                rawSentences.add(sentence)
            }
            start = end
            end = iterator.next()
        }

        val abbreviations = setOf("dr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "jr.", "e.g.", "i.e.", "vs.", "etc.")
        val merged = mutableListOf<String>()
        var i = 0
        while (i < rawSentences.size) {
            var current = rawSentences[i]
            while (i + 1 < rawSentences.size) {
                val lastToken = current.split(Regex("\\s+")).lastOrNull()?.lowercase()?.trimEnd('.', ',') + "."
                if (abbreviations.contains(lastToken)) {
                    i++
                    current = "$current ${rawSentences[i]}"
                } else {
                    break
                }
            }
            merged.add(current)
            i++
        }

        return if (merged.isNotEmpty()) merged else listOf(clean)
    }

    private fun countWords(text: String): Int {
        return text.split(Regex("""\s+""")).count { it.isNotEmpty() }
    }
}
