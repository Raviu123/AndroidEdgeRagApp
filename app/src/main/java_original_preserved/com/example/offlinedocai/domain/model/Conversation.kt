package com.example.offlinedocai.domain.model

data class Conversation(
    val id: String,
    val title: String,
    val documentId: String?,
    val createdAt: Long,
    val updatedAt: Long
)
