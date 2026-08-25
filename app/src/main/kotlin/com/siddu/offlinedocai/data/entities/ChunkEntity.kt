package com.siddu.offlinedocai.data.entities

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "chunks")
data class ChunkEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val documentId: Long,
    val pageNumber: Int,
    val text: String,
    var embedding: ByteArray? = null
)
