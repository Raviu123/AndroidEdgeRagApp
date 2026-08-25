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

    /**
     * Public entry point – determines the file type from the Uri and forwards to the appropriate extractor.
     */
    suspend fun extractDocument(context: Context, uri: Uri): ExtractedDocument = withContext(Dispatchers.IO) {
        // Initialise PDFBox once – safe to call multiple times.
        PDFBoxResourceLoader.init(context.applicationContext)

        val resolver = context.contentResolver
        val mimeType = resolver.getType(uri) ?: ""
        val path = uri.path ?: ""
        return@withContext when {
            mimeType.endsWith("pdf") || path.endsWith(".pdf", ignoreCase = true) -> extractPdf(resolver.openInputStream(uri))
            mimeType.endsWith("msword") || mimeType.endsWith("vnd.openxmlformats-officedocument.wordprocessingml.document") || path.endsWith(".docx", ignoreCase = true) -> extractDocx(resolver.openInputStream(uri))
            mimeType.endsWith("plain") || path.endsWith(".txt", ignoreCase = true) -> extractTxt(resolver.openInputStream(uri))
            else -> throw DocumentProcessingException.FileNotFound("Unsupported file type or unable to determine type for Uri: $uri")
        }
    }

    /**
     * Extract text page‑by‑page from a PDF InputStream.
     */
    private fun extractPdf(inputStream: InputStream?): ExtractedDocument {
        if (inputStream == null) throw DocumentProcessingException.UriOpeningFailed("Cannot open InputStream for PDF")
        try {
            inputStream.use { stream ->
                PDDocument.load(stream).use { document ->
                    val pageCount = document.numberOfPages
                    if (pageCount == 0) throw DocumentProcessingException.EmptyDocument("PDF contains no pages")
                    val stripper = PDFTextStripper()
                    val pages = mutableListOf<ExtractedPage>()
                    for (i in 1..pageCount) {
                        stripper.startPage = i
                        stripper.endPage = i
                        val rawText = stripper.getText(document).trim()
                        if (rawText.isNotEmpty()) {
                            pages.add(ExtractedPage(pageNumber = i, text = rawText))
                        }
                    }
                    return ExtractedDocument(pageCount = pageCount, pages = pages)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "PDF extraction failed", e)
            throw DocumentProcessingException.PdfParsingFailed("Failed to parse PDF", e)
        }
    }

    /**
     * Extract text from a DOCX InputStream.
     * DOCX does not expose page boundaries; we treat the whole document as a single logical page (page 1).
     */
    private fun extractDocx(inputStream: InputStream?): ExtractedDocument {
        if (inputStream == null) throw DocumentProcessingException.UriOpeningFailed("Cannot open InputStream for DOCX")
        try {
            inputStream.use { stream ->
                XWPFDocument(stream).use { doc ->
                    val paragraphs = doc.paragraphs
                    if (paragraphs.isEmpty()) throw DocumentProcessingException.EmptyDocument("DOCX contains no readable text")
                    val textBuilder = StringBuilder()
                    for (p in paragraphs) {
                        textBuilder.append(p.text).append('\n')
                    }
                    val fullText = textBuilder.toString().trim()
                    val page = ExtractedPage(pageNumber = 1, text = fullText)
                    return ExtractedDocument(pageCount = 1, pages = listOf(page))
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "DOCX extraction failed", e)
            throw DocumentProcessingException.DocxParsingFailed("Failed to parse DOCX", e)
        }
    }

    /**
     * Extract text from a plain TXT InputStream.
     * The whole file is regarded as a single page.
     */
    private fun extractTxt(inputStream: InputStream?): ExtractedDocument {
        if (inputStream == null) throw DocumentProcessingException.UriOpeningFailed("Cannot open InputStream for TXT")
        try {
            inputStream.use { stream ->
                BufferedReader(InputStreamReader(stream)).use { reader ->
                    val lines = reader.readLines()
                    if (lines.isEmpty()) throw DocumentProcessingException.EmptyDocument("TXT file is empty")
                    val fullText = lines.joinToString("\n").trim()
                    val page = ExtractedPage(pageNumber = 1, text = fullText)
                    return ExtractedDocument(pageCount = 1, pages = listOf(page))
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "TXT extraction failed", e)
            throw DocumentProcessingException.TxtParsingFailed("Failed to read TXT file", e)
        }
    }
}
