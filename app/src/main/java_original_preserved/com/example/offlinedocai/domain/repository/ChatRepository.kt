package com.example.offlinedocai.domain.repository

import com.example.offlinedocai.domain.model.Conversation
import com.example.offlinedocai.domain.model.Message
import kotlinx.coroutines.flow.Flow

interface ChatRepository {
    fun getAllConversationsFlow(): Flow<List<Conversation>>
    suspend fun getConversationById(id: String): Conversation?
    suspend fun insertConversation(conversation: Conversation)
    suspend fun updateConversation(conversation: Conversation)
    suspend fun deleteConversation(conversation: Conversation)
    fun getMessagesFlowForConversation(conversationId: String): Flow<List<Message>>
    suspend fun getMessagesForConversation(conversationId: String): List<Message>
    suspend fun insertMessage(message: Message)
    suspend fun deleteMessagesForConversation(conversationId: String)
}
