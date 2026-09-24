package com.example.offlinedocai.data.repository

import com.example.offlinedocai.data.database.dao.DocumentDao
import com.example.offlinedocai.data.database.entities.DocumentChunkEntity
import com.example.offlinedocai.data.database.entities.DocumentEntity
import com.example.offlinedocai.domain.model.Document
import com.example.offlinedocai.domain.repository.DocumentRepository
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

class DocumentRepositoryImpl(private val documentDao: DocumentDao) : DocumentRepository {

    override fun getAllDocumentsFlow(): Flow<List<Document>> {
        return documentDao.getAllDocumentsFlow().map { entities ->
            entities.map { it.toDomain() }
        }
    }

    override suspend fun getDocumentById(id: String): Document? {
        return documentDao.getDocumentById(id)?.toDomain()
    }

    override suspend fun insertDocument(document: Document) {
        documentDao.insertDocument(document.toEntity())
    }

    override suspend fun updateDocument(document: Document) {
        documentDao.updateDocument(document.toEntity())
    }

    override suspend fun deleteDocument(document: Document) {
        documentDao.deleteDocument(document.toEntity())
    }

    override suspend fun getChunksForDocument(documentId: String): List<DocumentChunkEntity> {
        return documentDao.getChunksForDocument(documentId)
    }

    override suspend fun insertChunks(chunks: List<DocumentChunkEntity>) {
        documentDao.insertChunks(chunks)
    }

    override suspend fun deleteChunksForDocument(documentId: String) {
        documentDao.deleteChunksForDocument(documentId)
    }

    private fun DocumentEntity.toDomain(): Document = Document(
        id = id,
        name = name,
        type = type,
        filePath = filePath,
        createdAt = createdAt,
        updatedAt = updatedAt,
        pageCount = pageCount,
        processingStatus = processingStatus,
        fileSize = fileSize,
        chunkCount = chunkCount
    )

    private fun Document.toEntity(): DocumentEntity = DocumentEntity(
        id = id,
        name = name,
        type = type,
        filePath = filePath,
        createdAt = createdAt,
        updatedAt = updatedAt,
        pageCount = pageCount,
        processingStatus = processingStatus,
        fileSize = fileSize,
        chunkCount = chunkCount
    )
}
