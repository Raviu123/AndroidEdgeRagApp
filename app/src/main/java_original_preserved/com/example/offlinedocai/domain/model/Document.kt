package com.example.offlinedocai.domain.model

data class Document(
    val id: String,
    val name: String,
    val type: String, // PDF, DOCX, TEXT
    val filePath: String,
    val createdAt: Long,
    val updatedAt: Long,
    val pageCount: Int,
    val processingStatus: String, // PENDING, PROCESSING, COMPLETED, FAILED
    val fileSize: Long,
    val chunkCount: Int
)
