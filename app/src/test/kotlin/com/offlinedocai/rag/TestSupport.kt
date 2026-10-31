package com.offlinedocai.rag

import com.offlinedocai.ai.embeddings.WordPieceTokenizer
import com.offlinedocai.data.FloatArrayConverter
import com.offlinedocai.data.dao.ChunkDao
import com.offlinedocai.data.dao.ChunkSignature
import com.offlinedocai.data.entities.ChunkEntity
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOf
import java.io.File
import kotlin.math.sqrt

/** In-memory ChunkDao for JVM tests. */
class FakeChunkDao : ChunkDao {
    val rows = mutableListOf<ChunkEntity>()
    private var maxIdEverUsed = 0L // like AUTOINCREMENT: ids are never reused

    override suspend fun insertAll(chunks: List<ChunkEntity>) {
        for (c in chunks) {
            val row = if (c.id == 0L) c.copy(id = maxIdEverUsed + 1) else c
            maxIdEverUsed = maxOf(maxIdEverUsed, row.id)
            rows.add(row)
        }
    }

    override fun getChunksForDocument(docId: Long): Flow<List<ChunkEntity>> = flowOf(sorted(rows.filter { it.documentId == docId }))

    override suspend fun getAllChunksForDocumentIds(docIds: List<Long>) = sorted(rows.filter { it.documentId in docIds })

    override suspend fun getAllChunksForDocument(docId: Long) = sorted(rows.filter { it.documentId == docId })

    override suspend fun getSignatures(docIds: List<Long>): List<ChunkSignature> =
        rows.filter { it.documentId in docIds }.groupBy { it.documentId }
            .map { (id, list) -> ChunkSignature(id, list.size, list.maxOf { it.id }) }

    override suspend fun countForDocument(docId: Long) = rows.count { it.documentId == docId }

    override suspend fun deleteChunksForDocument(docId: Long) {
        rows.removeAll { it.documentId == docId }
    }

    private fun sorted(list: List<ChunkEntity>) = list.sortedWith(compareBy({ it.documentId }, { it.chunkIndex }, { it.id }))
}

object TestVectors {
    const val DIM = 384

    /** Unit vector built from (dimension, weight) pairs — lets tests express "same topic, different words". */
    fun topic(vararg weights: Pair<Int, Float>): FloatArray {
        val v = FloatArray(DIM)
        for ((d, w) in weights) v[d] += w
        val norm = sqrt(v.sumOf { (it * it).toDouble() }).toFloat()
        return FloatArray(DIM) { v[it] / norm }
    }

    fun blob(v: FloatArray): ByteArray = FloatArrayConverter.fromFloatArray(v)!!
}

object TestTokenizer {
    /** The real MiniLM vocabulary shipped in assets (unit tests run with the module dir as working dir). */
    val instance: WordPieceTokenizer by lazy {
        val candidates = listOf(
            File("src/main/assets/models/embedding/vocab.txt"),
            File("app/src/main/assets/models/embedding/vocab.txt")
        )
        WordPieceTokenizer(candidates.first { it.exists() })
    }
}

fun chunk(
    id: Long,
    documentId: Long,
    index: Int,
    text: String,
    embedding: FloatArray? = null,
    page: Int = index + 1,
    pageSpan: String = "$page",
    section: String? = null
) = ChunkEntity(
    id = id,
    documentId = documentId,
    pageNumber = page,
    text = text,
    embedding = embedding?.let { TestVectors.blob(it) },
    chunkIndex = index,
    pageSpan = pageSpan,
    sectionTitle = section,
    embeddingModelId = "test",
    embeddingDim = embedding?.size ?: 0
)
