package com.example.offlinedocai.data.database.entities

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "documents")
data class DocumentEntity(
    @PrimaryKey val id: String,
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
