package com.example.offlinedocai.domain.model

data class Message(
    val id: String,
    val conversationId: String,
    val role: String, // user, assistant
    val content: String,
    val createdAt: Long
)
