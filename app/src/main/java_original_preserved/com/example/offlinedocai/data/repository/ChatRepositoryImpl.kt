package com.example.offlinedocai.data.repository

import com.example.offlinedocai.data.database.dao.ChatDao
import com.example.offlinedocai.data.database.entities.ConversationEntity
import com.example.offlinedocai.data.database.entities.MessageEntity
import com.example.offlinedocai.domain.model.Conversation
import com.example.offlinedocai.domain.model.Message
import com.example.offlinedocai.domain.repository.ChatRepository
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

class ChatRepositoryImpl(private val chatDao: ChatDao) : ChatRepository {

    override fun getAllConversationsFlow(): Flow<List<Conversation>> {
        return chatDao.getAllConversationsFlow().map { entities ->
            entities.map { it.toDomain() }
        }
    }

    override suspend fun getConversationById(id: String): Conversation? {
        return chatDao.getConversationById(id)?.toDomain()
    }

    override suspend fun insertConversation(conversation: Conversation) {
        chatDao.insertConversation(conversation.toEntity())
    }

    override suspend fun updateConversation(conversation: Conversation) {
        chatDao.updateConversation(conversation.toEntity())
    }

    override suspend fun deleteConversation(conversation: Conversation) {
        chatDao.deleteConversation(conversation.toEntity())
    }

    override fun getMessagesFlowForConversation(conversationId: String): Flow<List<Message>> {
        return chatDao.getMessagesFlowForConversation(conversationId).map { entities ->
            entities.map { it.toDomain() }
        }
    }

    override suspend fun getMessagesForConversation(conversationId: String): List<Message> {
        return chatDao.getMessagesForConversation(conversationId).map { it.toDomain() }
    }

    override suspend fun insertMessage(message: Message) {
        chatDao.insertMessage(message.toEntity())
    }

    override suspend fun deleteMessagesForConversation(conversationId: String) {
        chatDao.deleteMessagesForConversation(conversationId)
    }

    private fun ConversationEntity.toDomain(): Conversation = Conversation(
        id = id,
        title = title,
        documentId = documentId,
        createdAt = createdAt,
        updatedAt = updatedAt
    )

    private fun Conversation.toEntity(): ConversationEntity = ConversationEntity(
        id = id,
        title = title,
        documentId = documentId,
        createdAt = createdAt,
        updatedAt = updatedAt
    )

    private fun MessageEntity.toDomain(): Message = Message(
        id = id,
        conversationId = conversationId,
        role = role,
        content = content,
        createdAt = createdAt
    )

    private fun Message.toEntity(): MessageEntity = MessageEntity(
        id = id,
        conversationId = conversationId,
        role = role,
        content = content,
        createdAt = createdAt
    )
}
