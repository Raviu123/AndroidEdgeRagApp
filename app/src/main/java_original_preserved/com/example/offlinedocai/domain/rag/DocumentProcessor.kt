package com.example.offlinedocai.domain.rag

import android.content.Context
import android.net.Uri
import android.util.Log
import com.example.offlinedocai.ai.chunking.TextChunker
import com.example.offlinedocai.ai.embeddings.EmbeddingModel
import com.example.offlinedocai.ai.retrieval.VectorStore
import com.example.offlinedocai.data.database.entities.DocumentChunkEntity
import com.example.offlinedocai.domain.model.Document
import com.example.offlinedocai.domain.repository.DocumentRepository
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.text.PDFTextStripper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import java.io.File
import java.io.FileOutputStream
import java.util.UUID

class DocumentProcessor(
    private val context: Context,
    private val documentRepository: DocumentRepository,
    private val embeddingModel: EmbeddingModel
) {

    private val chunker = TextChunker()

    sealed interface ProcessingState {

        data object Reading : ProcessingState

        data object Extracting : ProcessingState

        data object Chunking : ProcessingState

        data class Embedding(val progress: Float) : ProcessingState

        data object Saving : ProcessingState

        data class SavingProgress(
            val currentBatch: Int,
            val totalBatches: Int,
            val savedVectors: Int,
            val totalVectors: Int
        ) : ProcessingState

        data class Completed(
            val document: Document
        ) : ProcessingState

        data class Error(
            val message: String
        ) : ProcessingState
    }

    companion object {

        private const val TAG = "DocumentProcessor"

        private const val DOCUMENTS_DIRECTORY = "documents"

        private const val WORDS_PER_PAGE = 700
    }

    init {
        try {
            PDFBoxResourceLoader.init(
                context.applicationContext
            )
        } catch (e: Exception) {
            Log.e(
                TAG,
                "Failed to initialize PDFBox",
                e
            )
        }
    }

    /**
     * Processes a PDF, DOCX or TXT document completely locally.
     *
     * Pipeline:
     *
     * URI
     *  ↓
     * Copy to private storage
     *  ↓
     * Extract text
     *  ↓
     * Preserve page numbers
     *  ↓
     * Chunk text
     *  ↓
     * Generate embeddings
     *  ↓
     * Store chunks in Room
     */
    fun processDocument(
        uri: Uri,
        fileName: String,
        fileSize: Long,
        type: String,
        chunkSize: Int,
        chunkOverlap: Int,
        documentId: String = UUID.randomUUID().toString()
    ): Flow<ProcessingState> = flow {

        val startTime = System.currentTimeMillis()
        val mimeType = context.contentResolver.getType(uri)
        val fileExtension = fileName.substringAfterLast(".", "").uppercase()

        val actualType = when {
            type.uppercase() == "PDF" || mimeType == "application/pdf" || fileExtension == "PDF" -> "PDF"
            type.uppercase() == "DOCX" || mimeType == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" || fileExtension == "DOCX" -> "DOCX"
            type.uppercase() == "TXT" || mimeType == "text/plain" || fileExtension == "TXT" -> "TXT"
            else -> {
                emit(ProcessingState.Error("Unsupported file type. Please select a PDF, DOCX or TXT file."))
                return@flow
            }
        }
        Log.d(TAG, "Detected document type: $actualType")

        emit(ProcessingState.Reading)

        // ---------------------------------------------------------
        // 1. Prepare private document directory
        // ---------------------------------------------------------

        val documentsDirectory = File(
            context.filesDir,
            DOCUMENTS_DIRECTORY
        )

        if (!documentsDirectory.exists()) {
            documentsDirectory.mkdirs()
        }

        // Remove path information from the original filename.
        val safeFileName = fileName
            .substringAfterLast('/')
            .substringAfterLast('\\')

        val destinationFile = File(
            documentsDirectory,
            "${documentId}_$safeFileName"
        )

        // ---------------------------------------------------------
        // 2. Copy selected file to private app storage
        // ---------------------------------------------------------

        try {

            val inputStream =
                context.contentResolver.openInputStream(uri)

            if (inputStream == null) {

                emit(
                    ProcessingState.Error(
                        "Unable to open the selected file."
                    )
                )

                return@flow
            }

            inputStream.use { input ->

                FileOutputStream(
                    destinationFile
                ).use { output ->

                    input.copyTo(output)
                }
            }

        } catch (e: Exception) {

            Log.e(
                TAG,
                "Failed to copy document",
                e
            )

            emit(
                ProcessingState.Error(
                    "Failed to save the document locally: ${
                        e.message ?: "Unknown error"
                    }"
                )
            )

            return@flow
        }

        // ---------------------------------------------------------
        // 3. Create initial document record
        // ---------------------------------------------------------

        val currentTime =
            System.currentTimeMillis()

        val initialDocument = Document(
            id = documentId,
            name = safeFileName,
            type = actualType,
            filePath = destinationFile.absolutePath,
            createdAt = currentTime,
            updatedAt = currentTime,
            pageCount = 0,
            processingStatus = "PROCESSING",
            fileSize = fileSize,
            chunkCount = 0
        )

        try {

            documentRepository.insertDocument(
                initialDocument
            )

        } catch (e: Exception) {

            Log.e(
                TAG,
                "Failed to insert document",
                e
            )

            emit(
                ProcessingState.Error(
                    "Failed to save document information: ${
                        e.message ?: "Unknown error"
                    }"
                )
            )

            return@flow
        }

        // ---------------------------------------------------------
        // 4. Extract text
        // ---------------------------------------------------------

        emit(ProcessingState.Extracting)

        val pages =
            mutableListOf<Pair<Int, String>>()

        try {

            destinationFile.inputStream().use { inputStream ->

                when (actualType) {

                    // =================================================
                    // PDF
                    // =================================================

                    "PDF" -> {
                        Log.d(TAG, "Starting PDF extraction")
                        PDDocument.load(
                            inputStream
                        ).use { pdfDocument ->

                            val numberOfPages =
                                pdfDocument.numberOfPages
                            Log.d(TAG, "PDF page count: $numberOfPages")

                            if (numberOfPages <= 0) {

                                throw IllegalStateException(
                                    "The PDF contains no pages."
                                )
                            }

                            val stripper =
                                PDFTextStripper()

                            for (
                            pageNumber in
                            1..numberOfPages
                            ) {

                                stripper.startPage =
                                    pageNumber

                                stripper.endPage =
                                    pageNumber

                                val text =
                                    stripper
                                        .getText(pdfDocument)
                                        .trim()

                                if (text.isNotBlank()) {

                                    pages.add(
                                        Pair(
                                            pageNumber,
                                            text
                                        )
                                    )
                                    Log.d(TAG, "Extracted page: $pageNumber")
                                }
                            }
                        }
                        Log.d(TAG, "PDF extraction completed")
                    }

                    // =================================================
                    // DOCX
                    // =================================================

                    "DOCX" -> {

                        val text =
                            DocxExtractor
                                .extractText(inputStream)
                                .trim()

                        if (text.isNotBlank()) {

                            addTextAsPages(
                                text = text,
                                pages = pages
                            )
                        }
                    }

                    // =================================================
                    // TXT
                    // =================================================

                    "TXT" -> {

                        val text =
                            inputStream
                                .bufferedReader()
                                .use { reader ->
                                    reader.readText()
                                }
                                .trim()

                        if (text.isNotBlank()) {

                            addTextAsPages(
                                text = text,
                                pages = pages
                            )
                        }
                    }

                    else -> {

                        throw IllegalArgumentException(
                            "Unsupported document type: $type"
                        )
                    }
                }
            }

        } catch (e: Exception) {

            Log.e(
                TAG,
                "Document extraction failed",
                e
            )

            markDocumentFailed(
                initialDocument
            )

            emit(
                ProcessingState.Error(
                    "Document extraction failed: ${
                        e.message ?: "Unknown error"
                    }"
                )
            )

            return@flow
        }

        // ---------------------------------------------------------
        // 5. Verify extracted text
        // ---------------------------------------------------------

        if (pages.isEmpty()) {

            markDocumentFailed(
                initialDocument
            )

            emit(
                ProcessingState.Error(
                    "No extractable text was found in the document."
                )
            )

            return@flow
        }

        // ---------------------------------------------------------
        // 6. Chunk document
        // ---------------------------------------------------------

        emit(ProcessingState.Chunking)

        val textChunks = try {

            chunker.chunkDocument(
                pages = pages,
                wordCount = chunkSize,
                wordOverlap = chunkOverlap
            )

        } catch (e: Exception) {

            Log.e(
                TAG,
                "Chunking failed",
                e
            )

            markDocumentFailed(
                initialDocument
            )

            emit(
                ProcessingState.Error(
                    "Failed to split document into chunks: ${
                        e.message ?: "Unknown error"
                    }"
                )
            )

            return@flow
        }

        if (textChunks.isEmpty()) {

            markDocumentFailed(
                initialDocument
            )

            emit(
                ProcessingState.Error(
                    "No text chunks were generated."
                )
            )

            return@flow
        }

        Log.d(TAG, "Chunk count: ${textChunks.size}")

        val characterCount = pages.sumOf { it.second.length }

        // ---------------------------------------------------------
        // 7. Generate embeddings
        // ---------------------------------------------------------

        val embeddingStartTime = System.currentTimeMillis()
        emit(ProcessingState.Embedding(0.30f))

        val databaseChunks =
            mutableListOf<DocumentChunkEntity>()

        try {
            var embeddingIndex = 0
            val totalChunks = textChunks.size

            for (chunk in textChunks) {

                val embeddingVector =
                    embeddingModel.getEmbedding(
                        chunk.text
                    )

                if (embeddingVector.isEmpty()) {

                    throw IllegalStateException(
                        "Embedding model returned an empty vector."
                    )
                }

                val embeddingBytes =
                    VectorStore.floatArrayToByteArray(
                        embeddingVector
                    )

                databaseChunks.add(
                    DocumentChunkEntity(
                        id = chunk.id,
                        documentId = documentId,
                        chunkIndex = chunk.chunkIndex,
                        pageNumber = chunk.pageNumber,
                        text = chunk.text,
                        embedding = embeddingBytes
                    )
                )

                embeddingIndex++
                if (embeddingIndex % 5 == 0 || embeddingIndex == totalChunks) {
                    val progress = 0.30f + 0.60f * (embeddingIndex.toFloat() / totalChunks.toFloat())
                    emit(ProcessingState.Embedding(progress))
                }
            }

            val embeddingDuration = System.currentTimeMillis() - embeddingStartTime
            Log.d(TAG, "Embedding generation completed in $embeddingDuration ms for $totalChunks chunks")

            val dbInsertStartTime = System.currentTimeMillis()
            val batchSize = 50
            val chunksList = databaseChunks.toList()
            val totalBatches = (chunksList.size + batchSize - 1) / batchSize

            val totalEmbeddingsGenerated = chunksList.size
            val totalVectorsToSave = chunksList.size
            var totalVectorsSaved = 0
            var totalIndexed = 0

            Log.i("VECTOR_DEBUG", "TOTAL EMBEDDINGS GENERATED = $totalEmbeddingsGenerated")
            Log.i("VECTOR_DEBUG", "TOTAL VECTORS TO SAVE = $totalVectorsToSave")

            Log.i("VECTOR_DEBUG", "BEFORE VECTOR STORE SAVE. Chunks: ${chunksList.size}")
            if (chunksList.isNotEmpty()) {
                val dim = chunksList[0].embedding.size / 4
                val totalBytes = chunksList.sumOf { it.embedding.size }
                Log.i("VECTOR_DEBUG", "embedding.size = $dim, embeddingByteArray.size = $totalBytes")
            }

            for (batchIndex in 0 until totalBatches) {
                val batchNum = batchIndex + 1
                Log.i("VECTOR_DEBUG", "BATCH START: $batchNum/$totalBatches")

                val startIdx = batchIndex * batchSize
                val endIdx = minOf(startIdx + batchSize, chunksList.size)
                val batch = chunksList.subList(startIdx, endIdx)

                Log.i("VECTOR_DEBUG", "BATCH EMBEDDINGS READY: count=${batch.size}")

                Log.i("VECTOR_DEBUG", "BATCH BYTEARRAY CONVERSION START")
                // Bytes are already serialized in step 7, log trace for diagnostics
                Log.i("VECTOR_DEBUG", "BATCH BYTEARRAY CONVERSION END")

                Log.i("VECTOR_DEBUG", "BATCH DAO INSERT START")
                val daoStart = System.currentTimeMillis()
                documentRepository.insertChunks(batch)
                val daoDuration = System.currentTimeMillis() - daoStart
                Log.i("VECTOR_DEBUG", "BATCH DAO INSERT END in $daoDuration ms")

                totalVectorsSaved += batch.size

                Log.i("VECTOR_DEBUG", "BATCH VECTOR INDEX START")
                // Custom index construction is computed dynamically via dynamic cosine similarity
                Log.i("VECTOR_DEBUG", "BATCH VECTOR INDEX END")

                totalIndexed += batch.size

                Log.i("VECTOR_DEBUG", "BATCH STATUS UPDATE START")
                emit(ProcessingState.SavingProgress(batchNum, totalBatches, endIdx, totalChunks))
                Log.i("VECTOR_DEBUG", "BATCH STATUS UPDATE END")

                Log.i("VECTOR_DEBUG", "BATCH COMPLETE")
            }

            val dbInsertDuration = System.currentTimeMillis() - dbInsertStartTime
            Log.i("VECTOR_DEBUG", "AFTER VECTOR STORE SAVE")

            Log.i("VECTOR_DEBUG", "TOTAL VECTORS SAVED = $totalVectorsSaved")
            Log.i("VECTOR_DEBUG", "TOTAL INDEXED = $totalIndexed")

            Log.i("VECTOR_DEBUG", "BEFORE VECTOR INDEX")
            // No custom index construction, similarity computed dynamically on findTopK
            Log.i("VECTOR_DEBUG", "AFTER VECTOR INDEX")

            val totalDuration = System.currentTimeMillis() - startTime
            val runtime = Runtime.getRuntime()
            val usedMemory = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)

            Log.i(TAG, """
                === DOCUMENT PROCESSING COMPLETED ===
                documentId: $documentId
                file size: $fileSize bytes
                extracted character count: $characterCount
                chunk count: $totalChunks
                embedding duration: $embeddingDuration ms
                database insertion duration: $dbInsertDuration ms
                total processing duration: $totalDuration ms
                approximate memory usage: ${usedMemory} MB
            """.trimIndent())

            Log.i("VECTOR_DEBUG", "VECTOR SAVE COMPLETE")

        } catch (e: Exception) {

            Log.e(
                TAG,
                "Embedding generation failed",
                e
            )

            markDocumentFailed(
                initialDocument
            )

            emit(
                ProcessingState.Error(
                    "Failed to generate embeddings: ${
                        e.message ?: "Unknown error"
                    }"
                )
            )

            return@flow
        }

        // ---------------------------------------------------------
        // 8. Mark document completed
        // ---------------------------------------------------------

        val completedDocument =
            initialDocument.copy(
                pageCount = pages.size,
                processingStatus = "COMPLETED",
                chunkCount = databaseChunks.size,
                updatedAt = System.currentTimeMillis()
            )

        Log.i("VECTOR_DEBUG", "BEFORE STATUS UPDATE")
        val statusUpdateStart = System.currentTimeMillis()
        try {
            documentRepository.updateDocument(
                completedDocument
            )
            val statusUpdateDuration = System.currentTimeMillis() - statusUpdateStart
            Log.i("VECTOR_DEBUG", "AFTER STATUS UPDATE in $statusUpdateDuration ms")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to update completed document", e)
            emit(
                ProcessingState.Error(
                    "Document was processed, but its final status could not be saved."
                )
            )
            return@flow
        }

        Log.i("VECTOR_DEBUG", "PROCESSING COMPLETE")

        // ---------------------------------------------------------
        // 9. Finished
        // ---------------------------------------------------------

        emit(
            ProcessingState.Completed(
                completedDocument
            )
        )

    }.flowOn(Dispatchers.IO)

    // =============================================================
    // Helper: Split DOCX/TXT into artificial pages
    // =============================================================

    private fun addTextAsPages(
        text: String,
        pages: MutableList<Pair<Int, String>>
    ) {

        val words = text
            .split(Regex("\\s+"))
            .filter { it.isNotBlank() }

        if (words.isEmpty()) {
            return
        }

        var pageNumber = 1

        for (
        startIndex in
        words.indices step WORDS_PER_PAGE
        ) {

            val endIndex =
                minOf(
                    startIndex + WORDS_PER_PAGE,
                    words.size
                )

            val pageText =
                words
                    .subList(
                        startIndex,
                        endIndex
                    )
                    .joinToString(" ")
                    .trim()

            if (pageText.isNotBlank()) {

                pages.add(
                    Pair(
                        pageNumber,
                        pageText
                    )
                )
            }

            pageNumber++
        }
    }

    // =============================================================
    // Helper: Mark document as failed
    // =============================================================

    private suspend fun markDocumentFailed(
        document: Document
    ) {

        try {

            documentRepository.updateDocument(
                document.copy(
                    processingStatus = "FAILED",
                    updatedAt = System.currentTimeMillis()
                )
            )

        } catch (e: Exception) {

            Log.e(
                TAG,
                "Failed to update document status",
                e
            )
        }
    }
}