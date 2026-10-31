package com.offlinedocai.domain.rag

import android.content.Context
import android.net.Uri
import android.util.Log
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.text.PDFTextStripper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.apache.poi.xwpf.usermodel.XWPFDocument
import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader
import java.lang.Exception

/**
 * Simple data class representing a single extracted page.
 */
data class ExtractedPage(
    val pageNumber: Int,
    val text: String
)

/**
 * Represents a document that has been fully extracted.
 * @property pageCount total number of pages extracted
 * @property pages list of pages with their extracted text
 */
data class ExtractedDocument(
    val pageCount: Int,
    val pages: List<ExtractedPage>
)

/**
 * Exception hierarchy for document processing errors.
 */
sealed class DocumentProcessingException(message: String, cause: Throwable? = null) : Exception(message, cause) {
    class FileNotFound(message: String) : DocumentProcessingException(message)
    class UriOpeningFailed(message: String) : DocumentProcessingException(message)
    class PdfParsingFailed(message: String, cause: Throwable) : DocumentProcessingException(message, cause)
    class DocxParsingFailed(message: String, cause: Throwable) : DocumentProcessingException(message, cause)
    class TxtParsingFailed(message: String, cause: Throwable) : DocumentProcessingException(message, cause)
    class EmptyDocument(message: String) : DocumentProcessingException(message)
}

/**
 * Core processor responsible for extracting text from supported document types.
 * Supports PDF (via PDFBox‑Android), DOCX (via Apache POI) and plain TXT files.
 */
object DocumentProcessor {

    private const val TAG = "DocumentProcessor"
    private val MULTI_SPACE = Regex("[ \\t]{2,}")

    /**
     * Public entry point – determines the file type from the Uri and forwards to the appropriate extractor.
     */
    suspend fun extractDocument(context: Context, uri: Uri): ExtractedDocument = withContext(Dispatchers.IO) {
        val resolver = context.contentResolver
        val kind = detectKind(resolver.getType(uri) ?: "", uri.path ?: "")
        extract(context, kind) { resolver.openInputStream(uri) }
    }

    /** Extracts from the app's private copy of an imported file. */
    suspend fun extractDocument(context: Context, file: java.io.File, mimeType: String?): ExtractedDocument =
        withContext(Dispatchers.IO) {
            val kind = detectKind(mimeType ?: "", file.name)
            extract(context, kind) { file.inputStream() }
        }

    enum class Kind { PDF, DOCX, TXT }

    fun detectKind(mimeType: String, nameOrPath: String): Kind {
        val lowerName = nameOrPath.lowercase()
        return when {
            mimeType.endsWith("pdf") || lowerName.endsWith(".pdf") -> Kind.PDF
            mimeType.endsWith("vnd.openxmlformats-officedocument.wordprocessingml.document") || lowerName.endsWith(".docx") -> Kind.DOCX
            mimeType == "application/msword" || lowerName.endsWith(".doc") ->
                throw DocumentProcessingException.FileNotFound("Legacy .doc files are not supported. Save the file as .docx or PDF and import it again.")
            mimeType.startsWith("text/") || lowerName.endsWith(".txt") || lowerName.endsWith(".md") -> Kind.TXT
            else -> throw DocumentProcessingException.FileNotFound("Unsupported file type. Import a PDF, DOCX or TXT file.")
        }
    }

    private fun extract(context: Context, kind: Kind, open: () -> InputStream?): ExtractedDocument {
        val document = when (kind) {
            Kind.PDF -> {
                PDFBoxResourceLoader.init(context.applicationContext)
                extractPdf(open(), context.cacheDir)
            }
            Kind.DOCX -> extractDocx(open())
            Kind.TXT -> extractTxt(open())
        }
        if (document.pages.none { it.text.isNotBlank() }) {
            throw DocumentProcessingException.EmptyDocument(
                if (kind == Kind.PDF) "No selectable text was found. This looks like a scanned or image-only PDF; OCR is not supported yet."
                else "The document contains no readable text."
            )
        }
        return document
    }

    /**
     * Data structures for spatial token and line layout reconstruction.
     */
    internal data class PositionedToken(
        val text: String,
        val xMin: Float,
        val xMax: Float,
        val yMin: Float,
        val yMax: Float,
        val fontSize: Float
    )

    private data class LineFragment(
        val text: String,
        val xMin: Float,
        val xMax: Float,
        val yMin: Float,
        val yMax: Float,
        val yBase: Float,
        val fontSize: Float
    )

    /**
     * Extract text page‑by‑page from a PDF InputStream with multi-column spatial layout analysis.
     */
    private fun extractPdf(inputStream: InputStream?, tempDir: java.io.File?): ExtractedDocument {
        if (inputStream == null) throw DocumentProcessingException.UriOpeningFailed("Cannot open InputStream for PDF")
        try {
            inputStream.use { stream ->
                // Large PDFs are buffered in a temp file instead of RAM (only 16 MB is kept in memory).
                val memory = com.tom_roush.pdfbox.io.MemoryUsageSetting.setupMixed(16L * 1024 * 1024).also { s -> tempDir?.let { s.setTempDir(it) } }
                PDDocument.load(stream, memory).use { document ->
                    val pageCount = document.numberOfPages
                    if (pageCount == 0) throw DocumentProcessingException.EmptyDocument("PDF contains no pages")
                    
                    val pages = mutableListOf<ExtractedPage>()
                    val stripper = object : PDFTextStripper() {
                        private var currentPageTokens = mutableListOf<PositionedToken>()
                        private var currentPageFallbackText = StringBuilder()
                        private var currentPageNum = 1

                        override fun startPage(page: com.tom_roush.pdfbox.pdmodel.PDPage?) {
                            currentPageTokens.clear()
                            currentPageFallbackText.clear()
                            currentPageNum = currentPageNo
                        }

                        override fun writeString(text: String?, textPositions: MutableList<com.tom_roush.pdfbox.text.TextPosition>?) {
                            if (text != null && text.isNotBlank()) {
                                currentPageFallbackText.append(text).append(' ')
                                if (!textPositions.isNullOrEmpty()) {
                                    val xMin = textPositions.minOf { it.xDirAdj }
                                    val xMax = textPositions.maxOf { it.xDirAdj + it.widthDirAdj }
                                    val yMin = textPositions.minOf { it.yDirAdj - it.heightDir }
                                    val yMax = textPositions.maxOf { it.yDirAdj }
                                    val fontSize = textPositions.firstOrNull()?.fontSizeInPt ?: 10f
                                    currentPageTokens.add(PositionedToken(text.trim(), xMin, xMax, yMin, yMax, fontSize))
                                }
                            }
                        }

                        override fun writeWordSeparator() {
                            currentPageFallbackText.append(' ')
                        }

                        override fun writeLineSeparator() {
                            currentPageFallbackText.append("\n")
                        }

                        override fun writeParagraphSeparator() {
                            currentPageFallbackText.append("\n\n")
                        }

                        override fun endPage(page: com.tom_roush.pdfbox.pdmodel.PDPage?) {
                            val extractedText = reconstructPageText(
                                tokens = currentPageTokens,
                                rawStreamText = currentPageFallbackText.toString()
                            ).replace("\r\n", "\n").replace(MULTI_SPACE, " ").trim()

                            if (extractedText.isNotEmpty()) {
                                pages.add(ExtractedPage(pageNumber = currentPageNum, text = extractedText))
                            }
                        }
                    }
                    
                    // Single-pass O(N) extraction across all pages
                    stripper.getText(document)

                    if (pages.isEmpty()) {
                        // Fallback in case document has non-standard page stream
                        val allText = PDFTextStripper().getText(document).trim()
                        if (allText.isNotEmpty()) {
                            pages.add(ExtractedPage(pageNumber = 1, text = allText))
                        }
                    }

                    return ExtractedDocument(pageCount = pageCount, pages = pages)
                }
            }
        } catch (e: DocumentProcessingException) {
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "PDF extraction failed", e)
            throw DocumentProcessingException.PdfParsingFailed("Failed to parse PDF", e)
        }
    }

    /**
     * Reconstructs page text in correct reading order by analyzing horizontal line baselines
     * and detecting multi-column vertical gutters (e.g. 2-column papers and articles).
     */
    internal fun reconstructPageText(tokens: List<PositionedToken>, rawStreamText: String): String {
        if (tokens.isEmpty()) return rawStreamText.trim()

        // 1. Group tokens by baseline into visual line fragments
        val sortedTokens = tokens.sortedWith(compareBy({ it.yMax }, { it.xMin }))
        val lineFragments = mutableListOf<LineFragment>()
        
        var curTokens = mutableListOf<PositionedToken>()
        var curYBase = -1f
        var curFontSize = 10f

        fun flushLine() {
            if (curTokens.isEmpty()) return
            val sorted = curTokens.sortedBy { it.xMin }
            
            // Subdivide line if there are large horizontal gaps (> 16pt, e.g. column margins)
            val fragTokens = mutableListOf<PositionedToken>()
            fun flushFrag() {
                if (fragTokens.isEmpty()) return
                val text = fragTokens.joinToString(" ") { it.text.trim() }.trim()
                if (text.isNotEmpty()) {
                    val xMin = fragTokens.minOf { it.xMin }
                    val xMax = fragTokens.maxOf { it.xMax }
                    val yMin = fragTokens.minOf { it.yMin }
                    val yMax = fragTokens.maxOf { it.yMax }
                    lineFragments.add(LineFragment(text, xMin, xMax, yMin, yMax, curYBase, curFontSize))
                }
                fragTokens.clear()
            }

            for (t in sorted) {
                if (fragTokens.isNotEmpty()) {
                    val gap = t.xMin - fragTokens.last().xMax
                    if (gap > 16f) {
                        flushFrag()
                    }
                }
                fragTokens.add(t)
            }
            flushFrag()
            curTokens.clear()
        }

        for (t in sortedTokens) {
            if (curTokens.isEmpty()) {
                curTokens.add(t)
                curYBase = t.yMax
                curFontSize = t.fontSize
            } else {
                val yDiff = kotlin.math.abs(t.yMax - curYBase)
                val tolerance = kotlin.math.max(2.5f, curFontSize * 0.4f)
                if (yDiff <= tolerance) {
                    curTokens.add(t)
                } else {
                    flushLine()
                    curTokens.add(t)
                    curYBase = t.yMax
                    curFontSize = t.fontSize
                }
            }
        }
        flushLine()

        if (lineFragments.isEmpty()) return rawStreamText.trim()

        // 2. Compute spatial bounding box of the page text
        val pageMinX = lineFragments.minOf { it.xMin }
        val pageMaxX = lineFragments.maxOf { it.xMax }
        val pageWidth = pageMaxX - pageMinX

        // If page is narrow or too few fragments, return simple vertical order
        if (pageWidth < 120f || lineFragments.size < 6) {
            return formatFragments(lineFragments.sortedWith(compareBy({ it.yBase }, { it.xMin })))
        }

        // 3. Multi-Column Gutter Detection
        // Candidate gutter search window: middle 35% to 65% of the page text width
        val gutterStart = pageMinX + pageWidth * 0.35f
        val gutterEnd = pageMinX + pageWidth * 0.65f
        
        var bestGutterX = -1f
        var minCrossing = Int.MAX_VALUE
        var maxSeparation = -1

        val steps = 30
        val stepSize = (gutterEnd - gutterStart) / steps

        for (step in 0..steps) {
            val candidateX = gutterStart + step * stepSize
            val gutterMargin = 8f
            
            var leftCount = 0
            var rightCount = 0
            var crossCount = 0

            for (f in lineFragments) {
                // If the fragment is full-width spanning (e.g. title/banner), don't count as crossing column noise
                val isFullWidth = (f.xMax - f.xMin) > (pageWidth * 0.65f)
                if (isFullWidth) continue

                if (f.xMax < candidateX - gutterMargin) {
                    leftCount++
                } else if (f.xMin > candidateX + gutterMargin) {
                    rightCount++
                } else {
                    crossCount++
                }
            }

            val totalColFragments = leftCount + rightCount + crossCount
            if (totalColFragments > 0) {
                val minCol = minOf(leftCount, rightCount)
                // A valid 2-column layout has significant content on both left and right (at least 15% each)
                if (minCol >= totalColFragments * 0.15) {
                    if (crossCount < minCrossing || (crossCount == minCrossing && minCol > maxSeparation)) {
                        minCrossing = crossCount
                        maxSeparation = minCol
                        bestGutterX = candidateX
                    }
                }
            }
        }

        val totalNonSpanning = lineFragments.count { (it.xMax - it.xMin) <= (pageWidth * 0.65f) }
        val isTwoColumn = bestGutterX > 0f && minCrossing <= (totalNonSpanning * 0.10)

        if (!isTwoColumn) {
            // Single-column: sort top-to-bottom
            return formatFragments(lineFragments.sortedWith(compareBy({ it.yBase }, { it.xMin })))
        }

        // 4. Partition into Top Spanning, Left Column, Right Column, Bottom Spanning
        val gutterX = bestGutterX
        val spanning = mutableListOf<LineFragment>()
        val leftCol = mutableListOf<LineFragment>()
        val rightCol = mutableListOf<LineFragment>()

        for (f in lineFragments) {
            val isFullSpan = (f.xMax - f.xMin) > (pageWidth * 0.65f) || (f.xMin < gutterX - 15f && f.xMax > gutterX + 15f)
            if (isFullSpan) {
                spanning.add(f)
            } else {
                val midX = (f.xMin + f.xMax) / 2f
                if (midX < gutterX) {
                    leftCol.add(f)
                } else {
                    rightCol.add(f)
                }
            }
        }

        // Determine column vertical boundaries
        val colTopY = listOfNotNull(leftCol.minOfOrNull { it.yMin }, rightCol.minOfOrNull { it.yMin }).minOrNull() ?: 0f
        val colBottomY = listOfNotNull(leftCol.maxOfOrNull { it.yMax }, rightCol.maxOfOrNull { it.yMax }).maxOrNull() ?: Float.MAX_VALUE

        val topSpanning = spanning.filter { it.yMax <= colTopY + 10f }.sortedWith(compareBy({ it.yBase }, { it.xMin }))
        val bottomSpanning = spanning.filter { it.yMin >= colBottomY - 10f }.sortedWith(compareBy({ it.yBase }, { it.xMin }))
        val midSpanning = spanning.filter { it !in topSpanning && it !in bottomSpanning }.sortedWith(compareBy({ it.yBase }, { it.xMin }))

        val leftSorted = leftCol.sortedWith(compareBy({ it.yBase }, { it.xMin }))
        val rightSorted = rightCol.sortedWith(compareBy({ it.yBase }, { it.xMin }))

        val resultBlocks = mutableListOf<String>()
        if (topSpanning.isNotEmpty()) {
            resultBlocks.add(formatFragments(topSpanning))
        }
        if (leftSorted.isNotEmpty()) {
            resultBlocks.add(formatFragments(leftSorted))
        }
        if (rightSorted.isNotEmpty()) {
            resultBlocks.add(formatFragments(rightSorted))
        }
        if (midSpanning.isNotEmpty()) {
            resultBlocks.add(formatFragments(midSpanning))
        }
        if (bottomSpanning.isNotEmpty()) {
            resultBlocks.add(formatFragments(bottomSpanning))
        }

        return resultBlocks.filter { it.isNotBlank() }.joinToString("\n\n")
    }

    private fun formatFragments(fragments: List<LineFragment>): String {
        if (fragments.isEmpty()) return ""
        val sb = StringBuilder()
        var lastYBase = -1f
        var lastHeight = 12f

        for (f in fragments) {
            if (sb.isNotEmpty()) {
                val yGap = f.yBase - lastYBase
                if (yGap > lastHeight * 1.6f) {
                    sb.append("\n\n")
                } else if (yGap > lastHeight * 0.4f) {
                    sb.append("\n")
                } else {
                    sb.append(" ")
                }
            }
            sb.append(f.text)
            lastYBase = f.yBase
            lastHeight = kotlin.math.max(8f, f.yMax - f.yMin)
        }
        return sb.toString().trim()
    }

    private val HEADING_STYLE = Regex("""(?i)^(heading\s*(\d)|title|subtitle)$""")

    /**
     * Extract text and tables from a DOCX InputStream in document order.
     *
     * Pages: Word records where it last laid out page boundaries (`w:lastRenderedPageBreak`) plus
     * explicit page breaks; those are used as page numbers, so citations and "page N" questions work
     * for DOCX too. Files saved without layout information form a single page.
     * Headings: paragraphs with a Heading/Title style are emitted as Markdown headings, which the
     * chunker uses as reliable section boundaries.
     */
    private fun extractDocx(inputStream: InputStream?): ExtractedDocument {
        if (inputStream == null) throw DocumentProcessingException.UriOpeningFailed("Cannot open InputStream for DOCX")
        try {
            inputStream.use { stream ->
                XWPFDocument(stream).use { doc ->
                    val bodyElements = doc.bodyElements
                    if (bodyElements.isEmpty()) throw DocumentProcessingException.EmptyDocument("DOCX contains no readable text")
                    val pages = mutableListOf<StringBuilder>(StringBuilder())
                    fun newPage() { if (pages.last().isNotBlank()) pages.add(StringBuilder()) }

                    for (elem in bodyElements) {
                        when (elem) {
                            is org.apache.poi.xwpf.usermodel.XWPFParagraph -> {
                                val xml = runCatching { elem.ctp.xmlText() }.getOrDefault("")
                                val breaks = countOccurrences(xml, "lastRenderedPageBreak") + countOccurrences(xml, "w:type=\"page\"")
                                val breakBefore = elem.isPageBreak ||
                                    (breaks > 0 && elem.runs.firstOrNull()?.let { r -> runCatching { r.ctr.xmlText() }.getOrDefault("").let { "lastRenderedPageBreak" in it || "w:type=\"page\"" in it } } == true)
                                if (breakBefore) newPage()
                                val pText = elem.text.trim()
                                if (pText.isNotEmpty()) {
                                    val level = HEADING_STYLE.find(elem.style.orEmpty())?.let { m -> m.groupValues[2].toIntOrNull() ?: 1 }
                                    val line = if (level != null && pText.length <= 120 && !pText.contains('\n')) "#".repeat(level.coerceIn(1, 6)) + " " + pText else pText
                                    pages.last().append(line).append("\n\n")
                                }
                                if (breaks > 0 && !breakBefore) newPage()
                            }
                            is org.apache.poi.xwpf.usermodel.XWPFTable -> {
                                for (row in elem.rows) {
                                    val cells = row.tableCells.map { cell -> cell.text.trim().replace("\n", " ") }
                                    if (cells.any { it.isNotEmpty() }) {
                                        pages.last().append("| ").append(cells.joinToString(" | ")).append(" |\n")
                                    }
                                }
                                pages.last().append("\n")
                            }
                        }
                    }
                    val extracted = pages.mapIndexedNotNull { i, sb ->
                        sb.toString().trim().takeIf { it.isNotEmpty() }?.let { ExtractedPage(pageNumber = i + 1, text = it) }
                    }
                    if (extracted.isEmpty()) throw DocumentProcessingException.EmptyDocument("DOCX contains no readable text")
                    return ExtractedDocument(pageCount = pages.size, pages = extracted)
                }
            }
        } catch (e: DocumentProcessingException) {
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "DOCX extraction failed", e)
            throw DocumentProcessingException.DocxParsingFailed("Failed to parse DOCX", e)
        }
    }

    private fun countOccurrences(text: String, needle: String): Int {
        var count = 0
        var from = text.indexOf(needle)
        while (from >= 0) {
            count++
            from = text.indexOf(needle, from + needle.length)
        }
        return count
    }

    /**
     * Decodes text files: honours UTF-8/UTF-16 byte-order marks, accepts valid UTF-8, and falls back to
     * Windows-1252 (the usual encoding of legacy Western text files) instead of producing garbage.
     */
    fun decodeText(bytes: ByteArray): String {
        fun startsWith(vararg b: Int) = bytes.size >= b.size && b.indices.all { bytes[it] == b[it].toByte() }
        return when {
            startsWith(0xEF, 0xBB, 0xBF) -> String(bytes, 3, bytes.size - 3, Charsets.UTF_8)
            startsWith(0xFF, 0xFE) -> String(bytes, 2, bytes.size - 2, Charsets.UTF_16LE)
            startsWith(0xFE, 0xFF) -> String(bytes, 2, bytes.size - 2, Charsets.UTF_16BE)
            else -> try {
                Charsets.UTF_8.newDecoder()
                    .onMalformedInput(java.nio.charset.CodingErrorAction.REPORT)
                    .onUnmappableCharacter(java.nio.charset.CodingErrorAction.REPORT)
                    .decode(java.nio.ByteBuffer.wrap(bytes)).toString()
            } catch (e: java.nio.charset.CharacterCodingException) {
                String(bytes, java.nio.charset.Charset.forName("windows-1252"))
            }
        }
    }

    /**
     * Extract text from a plain TXT InputStream preserving line and paragraph structure. Form feeds
     * (\f, used by printed/exported text) separate pages.
     */
    private fun extractTxt(inputStream: InputStream?): ExtractedDocument {
        if (inputStream == null) throw DocumentProcessingException.UriOpeningFailed("Cannot open InputStream for TXT")
        try {
            inputStream.use { stream ->
                val fullText = decodeText(stream.readBytes()).replace("\r\n", "\n").replace('\r', '\n')
                val rawPages = fullText.split('\u000C')
                val pages = rawPages.mapIndexedNotNull { i, t -> t.trim().takeIf { it.isNotEmpty() }?.let { ExtractedPage(i + 1, it) } }
                if (pages.isEmpty()) throw DocumentProcessingException.EmptyDocument("TXT file is empty")
                return ExtractedDocument(pageCount = rawPages.size, pages = pages)
            }
        } catch (e: DocumentProcessingException) {
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "TXT extraction failed", e)
            throw DocumentProcessingException.TxtParsingFailed("Failed to read TXT file", e)
        }
    }
}
