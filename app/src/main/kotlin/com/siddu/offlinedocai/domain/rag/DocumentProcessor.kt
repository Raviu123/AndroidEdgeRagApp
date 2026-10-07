package com.siddu.offlinedocai.domain.rag

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
     * Extract text page‑by‑page from a PDF InputStream in a single O(N) pass.
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
                        private var currentPageText = StringBuilder()
                        private var currentPageNum = 1

                        override fun startPage(page: com.tom_roush.pdfbox.pdmodel.PDPage?) {
                            currentPageText.clear()
                            currentPageNum = currentPageNo
                        }

                        override fun writeString(text: String?, textPositions: MutableList<com.tom_roush.pdfbox.text.TextPosition>?) {
                            if (text != null) {
                                currentPageText.append(text)
                            }
                        }

                        // PDFBox emits inter-word gaps through writeWordSeparator(); without this override,
                        // PDFs that position words instead of storing space glyphs come out as "Thequickbrownfox".
                        override fun writeWordSeparator() {
                            currentPageText.append(' ')
                        }

                        override fun writeLineSeparator() {
                            currentPageText.append("\n")
                        }

                        override fun writeParagraphSeparator() {
                            currentPageText.append("\n\n")
                        }

                        override fun endPage(page: com.tom_roush.pdfbox.pdmodel.PDPage?) {
                            val trimmed = currentPageText.toString().replace("\r\n", "\n").replace(MULTI_SPACE, " ").trim()
                            if (trimmed.isNotEmpty()) {
                                pages.add(ExtractedPage(pageNumber = currentPageNum, text = trimmed))
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
