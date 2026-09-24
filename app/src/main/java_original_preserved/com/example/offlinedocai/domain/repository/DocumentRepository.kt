package com.example.offlinedocai.domain.repository

import com.example.offlinedocai.data.database.entities.DocumentChunkEntity
import com.example.offlinedocai.domain.model.Document
import kotlinx.coroutines.flow.Flow

interface DocumentRepository {
    fun getAllDocumentsFlow(): Flow<List<Document>>
    suspend fun getDocumentById(id: String): Document?
    suspend fun insertDocument(document: Document)
    suspend fun updateDocument(document: Document)
    suspend fun deleteDocument(document: Document)
    suspend fun getChunksForDocument(documentId: String): List<DocumentChunkEntity>
    suspend fun insertChunks(chunks: List<DocumentChunkEntity>)
    suspend fun deleteChunksForDocument(documentId: String)
}
