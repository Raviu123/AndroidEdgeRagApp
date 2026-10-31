package com.offlinedocai.data.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Transaction
import com.offlinedocai.data.entities.ChunkEntity
import kotlinx.coroutines.flow.Flow

/** Cheap fingerprint of a document's chunk set, used to invalidate the in-memory search index. */
data class ChunkSignature(val documentId: Long, val chunkCount: Int, val maxChunkId: Long)

@Dao
interface ChunkDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(chunks: List<ChunkEntity>)

    @Query("SELECT * FROM chunks WHERE documentId = :docId ORDER BY chunkIndex ASC, id ASC")
    fun getChunksForDocument(docId: Long): Flow<List<ChunkEntity>>

    @Query("SELECT * FROM chunks WHERE documentId IN (:docIds) ORDER BY documentId ASC, chunkIndex ASC, id ASC")
    suspend fun getAllChunksForDocumentIds(docIds: List<Long>): List<ChunkEntity>

    @Query("SELECT * FROM chunks WHERE documentId = :docId ORDER BY chunkIndex ASC, id ASC")
    suspend fun getAllChunksForDocument(docId: Long): List<ChunkEntity>

    @Query(
        "SELECT documentId, COUNT(*) AS chunkCount, MAX(id) AS maxChunkId FROM chunks " +
            "WHERE documentId IN (:docIds) GROUP BY documentId"
    )
    suspend fun getSignatures(docIds: List<Long>): List<ChunkSignature>

    @Query("SELECT COUNT(*) FROM chunks WHERE documentId = :docId")
    suspend fun countForDocument(docId: Long): Int

    @Query("DELETE FROM chunks WHERE documentId = :docId")
    suspend fun deleteChunksForDocument(docId: Long)

    /** Atomically replaces every chunk of a document (used for first import and re-indexing). */
    @Transaction
    suspend fun replaceChunksForDocument(docId: Long, chunks: List<ChunkEntity>) {
        deleteChunksForDocument(docId)
        insertAll(chunks)
    }
}
