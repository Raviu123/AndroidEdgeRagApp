package com.siddu.offlinedocai.data.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.siddu.offlinedocai.data.entities.ChunkEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface ChunkDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(chunks: List<ChunkEntity>)

    @Query("SELECT * FROM chunks WHERE documentId = :docId ORDER BY chunkIndex ASC, id ASC")
    fun getChunksForDocument(docId: Long): Flow<List<ChunkEntity>>

    // Retrieve all chunks for documents in deterministic document sequence
    @Query("SELECT * FROM chunks WHERE documentId IN (:docIds) ORDER BY chunkIndex ASC, id ASC")
    suspend fun getAllChunksForDocumentIds(docIds: List<Long>): List<ChunkEntity>

    @Query("SELECT * FROM chunks WHERE documentId IN (:docIds) AND pageNumber = :pageNumber ORDER BY chunkIndex ASC, id ASC")
    suspend fun getChunksForPage(docIds: List<Long>, pageNumber: Int): List<ChunkEntity>

    @Query("SELECT * FROM chunks WHERE documentId IN (:docIds) AND pageNumber BETWEEN :startPage AND :endPage ORDER BY chunkIndex ASC, id ASC")
    suspend fun getChunksForPageRange(docIds: List<Long>, startPage: Int, endPage: Int): List<ChunkEntity>
}
