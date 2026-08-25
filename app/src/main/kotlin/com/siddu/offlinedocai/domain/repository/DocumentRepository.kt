package com.siddu.offlinedocai.domain.repository

import android.content.Context
import android.net.Uri
import android.util.Log
import com.siddu.offlinedocai.data.AppDatabase
import com.siddu.offlinedocai.data.dao.ChunkDao
import com.siddu.offlinedocai.data.entities.ChunkEntity
import com.siddu.offlinedocai.data.entities.DocumentEntity
import com.siddu.offlinedocai.domain.rag.DocumentProcessor
import com.siddu.offlinedocai.ai.chunking.TextChunker
import com.siddu.offlinedocai.ai.embeddings.EmbeddingModel
import com.siddu.offlinedocai.ai.embeddings.GGUFEmbeddingEngine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import java.util.UUID

/**
 * Repository handling the full import pipeline:
 *   1. Extract raw text from the Uri (PDF/DOCX/TXT) via DocumentProcessor
 *   2. Chunk the text per page using TextChunker
 *   3. Generate embeddings for each chunk via EmbeddingModel
 *   4. Persist DocumentEntity and ChunkEntity records in Room
 *
 * The repository exposes a StateFlow of ImportState for UI consumption.
 */
class DocumentRepository(
    private val context: Context,
    private val db: AppDatabase
) {
    private val TAG = "DocumentRepository"
    private val _importState = MutableStateFlow<ImportState>(ImportState.Idle)
    val importState: StateFlow<ImportState> = _importState

    private val chunkDao: ChunkDao = db.chunkDao()
    private val documentDao = db.documentDao()
    private val embeddingEngine: EmbeddingModel = GGUFEmbeddingEngine(context)

    /** Public API to start an import */
    fun importDocument(uri: Uri) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                _importState.value = ImportState.Reading
                val result = DocumentProcessor.extractDocument(context, uri)
                // Insert DocumentEntity
                val document = DocumentEntity(
                    id = 0, // auto‑generated
                    name = result.fileName,
                    uri = uri.toString(),
                    pageCount = result.pageCount,
                    importedAt = System.currentTimeMillis()
                )
                val docId = documentDao.insert(document)

                _importState.value = ImportState.Chunking
                val allChunks = mutableListOf<ChunkEntity>()
                result.pages.forEachIndexed { pageIdx, pageText ->
                    val chunks = TextChunker.chunkText(pageText, pageNumber = pageIdx + 1)
                    chunks.forEach { chunk ->
                        allChunks.add(
                            ChunkEntity(
                                id = 0,
                                documentId = docId,
                                pageNumber = chunk.pageNumber,
                                text = chunk.text,
                                embedding = null // placeholder, will be filled after embedding
                            )
                        )
                    }
                }

                _importState.value = ImportState.Embedding
                // Generate embeddings for each chunk sequentially (could be parallelized)
                for (chunk in allChunks) {
                    val embedding = embeddingEngine.getEmbedding(chunk.text)
                    // Convert FloatArray -> ByteArray using the converter
                    val bytes = com.siddu.offlinedocai.data.FloatArrayConverter.fromFloatArray(embedding)
                    chunk.embedding = bytes
                }

                _importState.value = ImportState.Saving
                chunkDao.insertAll(allChunks)
                _importState.value = ImportState.Completed
            } catch (e: Exception) {
                Log.e(TAG, "Import failed", e)
                _importState.value = ImportState.Error(e.message ?: "Unknown error")
            }
        }
    }

    /** Simple sealed class representing the current import step */
    sealed class ImportState {
        object Idle : ImportState()
        object Reading : ImportState()
        object Chunking : ImportState()
        object Embedding : ImportState()
        object Saving : ImportState()
        object Completed : ImportState()
        data class Error(val message: String) : ImportState()
    }
}
