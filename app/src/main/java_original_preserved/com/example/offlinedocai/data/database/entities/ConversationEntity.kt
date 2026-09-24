package com.example.offlinedocai.data.database.entities

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "conversations",
    foreignKeys = [
        ForeignKey(
            entity = DocumentEntity::class,
            parentColumns = ["id"],
            childColumns = ["documentId"],
            onDelete = ForeignKey.SET_NULL // Keep conversation even if doc is deleted
        )
    ],
    indices = [Index(value = ["documentId"])]
)
data class ConversationEntity(
    @PrimaryKey val id: String,
    val title: String,
    val documentId: String?, // Nullable for general AI chat (no specific doc)
    val createdAt: Long,
    val updatedAt: Long
)
