package com.siddu.offlinedocai.domain.repository

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Log
import com.siddu.offlinedocai.ai.chunking.TextChunker
import com.siddu.offlinedocai.ai.embeddings.EmbeddingManager
import com.siddu.offlinedocai.ai.embeddings.EmbeddingModel
import com.siddu.offlinedocai.ai.embeddings.ONNXEmbeddingEngine
import com.siddu.offlinedocai.ai.embeddings.WordPieceTokenizer
import com.siddu.offlinedocai.data.AppDatabase
import com.siddu.offlinedocai.data.FloatArrayConverter
import com.siddu.offlinedocai.data.entities.ChunkEntity
import com.siddu.offlinedocai.data.entities.DocumentEntity
import com.siddu.offlinedocai.domain.rag.DocumentProcessor
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.isActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File
import java.util.concurrent.ConcurrentHashMap

/**
 * Document import / indexing pipeline:
 *   1. Register the document (status PROCESSING) and copy the file into private storage.
 *   2. Extract text per page (PDF / DOCX / TXT).
 *   3. Chunk with [TextChunker], measured in the embedding model's own tokens.
 *   4. Embed the chunks in batches with the **active** embedding model ([EmbeddingManager]).
 *   5. Atomically replace the document's chunks and mark it COMPLETED with the index signature
 *      (chunker version + embedding model id). A different signature means "re-index".
 */
class DocumentRepository(
    private val context: Context,
    private val db: AppDatabase,
    private val embeddings: EmbeddingManager = EmbeddingManager.getInstance(context),
    private val tokenizer: () -> WordPieceTokenizer = { embeddings.chunkTokenizer }
) {
    private val TAG = "DocumentRepository"

    companion object {
        const val STATUS_PROCESSING = "PROCESSING"
        const val STATUS_COMPLETED = "COMPLETED"
        const val STATUS_FAILED = "FAILED"

        private const val EMBED_BATCH = 16

        fun indexSignature(embeddingModelId: String) = "chunker-v${TextChunker.VERSION}|$embeddingModelId"

        /** Signature of documents indexed with the default (built-in) embedding model. */
        val CURRENT_INDEX_SIGNATURE = indexSignature(ONNXEmbeddingEngine.MODEL_IDENTIFIER)

        /** Marks legacy chunks that are kept because the source file is no longer available. */
        private const val LEGACY_SIGNATURE_PREFIX = "legacy|"

        /** True when [doc]'s chunks were embedded by [embeddingModelId] (so they can be searched with it). */
        fun isSearchable(doc: DocumentEntity, embeddingModelId: String): Boolean {
            if (doc.status != STATUS_COMPLETED) return false
            val signature = doc.indexSignature?.removePrefix(LEGACY_SIGNATURE_PREFIX) ?: return false
            return signature.substringAfter('|') == embeddingModelId
        }
    }

    private val _importState = MutableStateFlow<ImportState>(ImportState.Idle)
    val importState: StateFlow<ImportState> = _importState

    private val chunkDao = db.chunkDao()
    private val documentDao = db.documentDao()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    /** Indexing is memory-heavy (whole document + model); run one at a time. */
    private val indexMutex = Mutex()
    private val activeDocuments = ConcurrentHashMap.newKeySet<Long>()
    private val jobs = ConcurrentHashMap<Long, Job>()

    private val documentsDir: File get() = File(context.filesDir, "documents").apply { mkdirs() }

    /** The signature a document must have to be searchable with the active embedding model. */
    fun currentSignature(): String = indexSignature(embeddings.modelId)

    fun isSearchable(doc: DocumentEntity): Boolean = isSearchable(doc, embeddings.modelId)

    fun importDocument(uri: Uri) {
        scope.launch {
            val fileName = queryDisplayName(uri)
            val mimeType = context.contentResolver.getType(uri)
            var docId = 0L
            try {
                docId = documentDao.insert(
                    DocumentEntity(uri = uri.toString(), name = fileName, type = mimeType ?: fileName.substringAfterLast('.', ""), status = STATUS_PROCESSING)
                )
                _importState.value = ImportState.Reading(fileName)
                val localFile = copyToPrivateStorage(uri, docId, fileName)
                val document = documentDao.getById(docId)!!.copy(localPath = localFile.absolutePath)
                documentDao.update(document)
                indexDocument(document)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "Import failed for $uri", e)
                markFailed(docId, e)
            }
        }
    }

    /** Re-chunk and re-embed an existing document (manual "Re-index" or stale index). */
    fun reprocessDocument(document: DocumentEntity) {
        scope.launch {
            try {
                indexDocument(document)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "Re-index failed for ${document.name}", e)
                markFailed(document.id, e)
            }
        }
    }

    /**
     * Finishes imports interrupted by process death and re-indexes documents built with an older
     * chunker or with a different embedding model (e.g. after the user switched search models).
     */
    fun reindexStaleDocuments() {
        scope.launch {
            val target = currentSignature()
            val stale = documentDao.getAllDocumentsOnce().filter { doc ->
                val signature = doc.indexSignature
                when {
                    doc.status == STATUS_FAILED -> false
                    signature != null && signature.startsWith(LEGACY_SIGNATURE_PREFIX) -> signature.removePrefix(LEGACY_SIGNATURE_PREFIX) != target && doc.localPath != null
                    else -> doc.status == STATUS_PROCESSING || signature != target
                }
            }
            for (doc in stale) {
                try {
                    indexDocument(doc)
                } catch (e: CancellationException) {
                    throw e
                } catch (e: SourceUnavailableException) {
                    // The original file can no longer be read (pre-v3 imports were not copied). Keep the
                    // existing, still-compatible chunks searchable instead of destroying them.
                    if (chunkDao.countForDocument(doc.id) > 0) {
                        documentDao.update(doc.copy(status = STATUS_COMPLETED, indexSignature = LEGACY_SIGNATURE_PREFIX + doc.indexSignature, errorMessage = e.message))
                    } else {
                        markFailed(doc.id, e)
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Re-index failed for ${doc.name}", e)
                    markFailed(doc.id, e)
                }
            }
        }
    }

    suspend fun deleteDocument(documentId: Long) {
        jobs.remove(documentId)?.cancel() // stop indexing work for a document that is being deleted
        val doc = documentDao.getById(documentId) ?: return
        chunkDao.deleteChunksForDocument(documentId) // explicit; ON DELETE CASCADE also covers it
        documentDao.delete(doc)
        doc.localPath?.let { runCatching { File(it).delete() } }
    }

    private suspend fun indexDocument(document: DocumentEntity) {
        if (!activeDocuments.add(document.id)) {
            Log.w(TAG, "Document ${document.id} is already being indexed")
            return
        }
        // async (not launch): failures are rethrown by await() to the caller instead of crashing the scope.
        val job = scope.async {
            indexMutex.withLock {
                // Re-read: the row may have changed (or been deleted) while waiting for the lock.
                val fresh = documentDao.getById(document.id) ?: return@withLock
                runIndexing(fresh)
            }
        }
        jobs[document.id] = job
        try {
            job.await()
        } catch (e: CancellationException) {
            // Only this document's job was cancelled (it was deleted): carry on with the caller's work.
            if (!kotlinx.coroutines.currentCoroutineContext().isActive) throw e
            return
        } finally {
            jobs.remove(document.id, job)
            activeDocuments.remove(document.id)
        }
        // The search model changed while this document was being indexed: index it again with the new one.
        val after = documentDao.getById(document.id)
        if (after != null && after.status == STATUS_COMPLETED && after.indexSignature != currentSignature() &&
            after.indexSignature?.startsWith(LEGACY_SIGNATURE_PREFIX) != true
        ) {
            indexDocument(after)
        }
    }

    private suspend fun runIndexing(original: DocumentEntity) {
        val name = original.name
        val doc = original.copy(status = STATUS_PROCESSING, errorMessage = null)
        try {
            documentDao.update(doc)

            _importState.value = ImportState.Reading(name)
            val extracted = extract(doc)
            documentDao.update(doc.copy(pageCount = extracted.pageCount))

            _importState.value = ImportState.Chunking(name, extracted.pageCount)
            val wordPiece = tokenizer()
            val chunks = TextChunker.chunkDocument(extracted.pages, tokenCounter = wordPiece::countTokens)
            if (chunks.isEmpty()) throw IllegalStateException("No text chunks could be produced from this document.")

            // One engine for the whole document, so every vector comes from the same model.
            val engine: EmbeddingModel = embeddings.currentEngine()
            val modelId = engine.modelId
            _importState.value = ImportState.Embedding(name, current = 0, total = chunks.size)
            val vectors = ArrayList<FloatArray>(chunks.size)
            for (batch in chunks.chunked(EMBED_BATCH)) {
                vectors.addAll(engine.embedDocuments(batch.map(::embeddingInput)))
                _importState.value = ImportState.Embedding(name, current = vectors.size, total = chunks.size)
            }
            val entities = chunks.mapIndexed { index, chunk ->
                val vector = vectors[index]
                ChunkEntity(
                    documentId = doc.id,
                    pageNumber = chunk.pageNumber,
                    text = chunk.text,
                    embedding = FloatArrayConverter.fromFloatArray(vector),
                    chunkIndex = chunk.index,
                    pageSpan = chunk.pageSpan,
                    sectionTitle = chunk.sectionTitle,
                    paragraphIndex = chunk.paragraphIndex,
                    embeddingModelId = modelId,
                    embeddingDim = vector.size
                )
            }

            _importState.value = ImportState.Saving(name)
            if (documentDao.getById(doc.id) == null) return // deleted meanwhile
            chunkDao.replaceChunksForDocument(doc.id, entities)
            documentDao.update(
                doc.copy(pageCount = extracted.pageCount, status = STATUS_COMPLETED, indexSignature = indexSignature(modelId), errorMessage = null)
            )
            _importState.value = ImportState.Completed(name)
            Log.i(TAG, "Indexed '$name': ${extracted.pageCount} pages, ${entities.size} chunks with $modelId")
        } catch (e: CancellationException) {
            throw e
        } catch (e: SourceUnavailableException) {
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "Indexing failed for $name", e)
            markFailed(doc.id, e)
        }
    }

    /** The section title gives short chunks their topic, which materially improves dense retrieval. */
    private fun embeddingInput(chunk: TextChunker.Chunk): String {
        val title = chunk.sectionTitle?.trim().orEmpty()
        return if (title.isNotEmpty() && !chunk.text.startsWith(title)) "$title\n${chunk.text}" else chunk.text
    }

    private suspend fun extract(doc: DocumentEntity) = run {
        val local = doc.localPath?.let(::File)
        when {
            local != null && local.exists() -> DocumentProcessor.extractDocument(context, local, doc.type)
            else -> try {
                DocumentProcessor.extractDocument(context, Uri.parse(doc.uri))
            } catch (e: SecurityException) {
                throw SourceUnavailableException()
            } catch (e: com.siddu.offlinedocai.domain.rag.DocumentProcessingException.PdfParsingFailed) {
                if (e.cause is SecurityException || e.cause is java.io.FileNotFoundException) throw SourceUnavailableException() else throw e
            } catch (e: java.io.FileNotFoundException) {
                throw SourceUnavailableException()
            }
        }
    }

    private class SourceUnavailableException :
        Exception("The original file is no longer accessible. Remove the document and import it again.")

    private fun copyToPrivateStorage(uri: Uri, docId: Long, fileName: String): File {
        val safeName = fileName.replace(Regex("[^A-Za-z0-9._-]"), "_").takeLast(80).ifEmpty { "document" }
        val target = File(documentsDir, "$docId-$safeName")
        val tmp = File(documentsDir, "$docId-$safeName.tmp")
        val input = context.contentResolver.openInputStream(uri)
            ?: throw IllegalStateException("Cannot open the selected file.")
        input.use { src -> tmp.outputStream().use { src.copyTo(it) } }
        if (target.exists()) target.delete()
        if (!tmp.renameTo(target)) throw IllegalStateException("Could not store the imported file.")
        return target
    }

    private fun queryDisplayName(uri: Uri): String {
        if (uri.scheme == "content") {
            context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    if (index != -1) cursor.getString(index)?.let { if (it.isNotBlank()) return it }
                }
            }
        }
        return uri.lastPathSegment?.substringAfterLast('/') ?: "document"
    }

    private suspend fun markFailed(docId: Long, error: Throwable) {
        val message = error.message ?: "Unknown error"
        if (docId > 0) {
            runCatching {
                documentDao.getById(docId)?.let { documentDao.update(it.copy(status = STATUS_FAILED, errorMessage = message)) }
            }
        }
        _importState.value = ImportState.Error(message)
    }

    sealed class ImportState {
        object Idle : ImportState()
        data class Reading(val documentName: String) : ImportState()
        data class Chunking(val documentName: String, val pageCount: Int) : ImportState()
        data class Embedding(val documentName: String, val current: Int, val total: Int) : ImportState() {
            val percent: Int get() = if (total > 0) ((current * 100) / total).coerceIn(0, 100) else 0
        }
        data class Saving(val documentName: String) : ImportState()
        data class Completed(val documentName: String) : ImportState()
        data class Error(val message: String) : ImportState()
    }
}
