package com.siddu.offlinedocai.data.entities

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * One retrieval unit of a document.
 *
 * [embedding] is a big-endian float32 BLOB produced by the model named in [embeddingModelId];
 * [embeddingDim] is stored alongside so vectors from a different model are never compared.
 * Chunks are removed automatically when their parent document is deleted (ON DELETE CASCADE).
 */
@Entity(
    tableName = "chunks",
    foreignKeys = [
        ForeignKey(
            entity = DocumentEntity::class,
            parentColumns = ["id"],
            childColumns = ["documentId"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [Index("documentId")]
)
data class ChunkEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val documentId: Long,
    val pageNumber: Int,
    val text: String,
    var embedding: ByteArray? = null,
    val chunkIndex: Int = 0,
    val pageSpan: String = "$pageNumber",
    val sectionTitle: String? = null,
    val paragraphIndex: Int = 0,
    val embeddingModelId: String? = null,
    val embeddingDim: Int = 0
)
