package com.offlinedocai.data.entities

import androidx.room.Entity
import androidx.room.PrimaryKey
import java.util.Date

@Entity(tableName = "documents")
data class DocumentEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val uri: String,
    val name: String,
    val type: String,
    val pageCount: Int = 0,
    val addedAt: Long = Date().time,
    val status: String = "PROCESSING",
    /** Private copy of the imported file, so the document can be re-indexed without the original URI grant. */
    val localPath: String? = null,
    /** Chunker + embedding-model signature the stored chunks were built with; null = needs (re)indexing. */
    val indexSignature: String? = null,
    /** Human-readable reason when [status] is FAILED. */
    val errorMessage: String? = null
)
