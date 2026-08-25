package com.siddu.offlinedocai

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.room.Room
import com.siddu.offlinedocai.data.AppDatabase
import com.siddu.offlinedocai.data.entities.DocumentEntity
import com.siddu.offlinedocai.data.entities.ConversationEntity
import com.siddu.offlinedocai.data.entities.MessageEntity
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

/**
 * Central ViewModel for the OfflineDocAI app.
 * Provides access to the Room database and exposes StateFlows for UI observation.
 */
class MainViewModel(application: Application) : AndroidViewModel(application) {
    private val database: AppDatabase = Room.databaseBuilder(
        application,
        AppDatabase::class.java,
        "offline_docai.db"
    ).fallbackToDestructiveMigration().build()

    private val documentDao = database.documentDao()
    private val conversationDao = database.conversationDao()
    private val messageDao = database.messageDao()

    private val _documents = MutableStateFlow<List<DocumentEntity>>(emptyList())
    val documents: StateFlow<List<DocumentEntity>> = _documents

    private val _conversations = MutableStateFlow<List<ConversationEntity>>(emptyList())
    val conversations: StateFlow<List<ConversationEntity>> = _conversations

    init {
        // Load existing data from DB
        viewModelScope.launch {
            documentDao.getAllDocuments().collect { _documents.value = it }
        }
        viewModelScope.launch {
            conversationDao.getAllConversations().collect { _conversations.value = it }
        }
    }

    /** Insert a new processed document into the DB. */
    // Repositories
    private val documentRepository = DocumentRepository(application, database)
    private val vectorStore = VectorStore(database.chunkDao())
    private val chatRepository = ChatRepository(application, database, vectorStore)

    val importState: StateFlow<DocumentRepository.ImportState> = documentRepository.importState

    /** Trigger document import */
    fun importDocument(uri: Uri) {
        documentRepository.importDocument(uri)
    }

    /** Start a new conversation linked to selected document IDs */
    fun startConversation(title: String = "New Chat", documentIds: List<Long> = emptyList()): Long {
        return chatRepository.startConversation(title)
    }

    /** Send a message in a conversation */
    fun sendMessage(convId: Long, text: String, docIds: List<Long>) {
        chatRepository.sendMessage(convId, text, docIds)
    }

    val chatState: StateFlow<ChatRepository.ChatState> = chatRepository.chatState
        viewModelScope.launch {
            val doc = DocumentEntity(
                uri = uri,
                name = name,
                type = type,
                pageCount = pageCount,
                status = "COMPLETED"
            )
            documentDao.insert(doc)
        }
    }

    /** Create a new conversation, optionally linked to a document. */
    fun startConversation(title: String, documentId: Long? = null) {
        viewModelScope.launch {
            val conv = ConversationEntity(
                title = title,
                documentId = documentId
            )
            conversationDao.insert(conv)
        }
    }

    /** Persist a chat message. */
    // Delete a document
    fun deleteDocument(documentId: Long) {
        viewModelScope.launch {
            documentDao.delete(documentDao.getById(documentId)!!)
        }
    }

    // Get messages for a conversation as StateFlow
    fun messagesForConversation(convId: Long): StateFlow<List<MessageEntity>> {
        return messageDao.getMessagesForConversation(convId).stateIn(viewModelScope, SharingStarted.Eagerly, emptyList())
    }
        viewModelScope.launch {
            val msg = MessageEntity(
                conversationId = conversationId,
                role = role,
                content = content,
                sources = sources
            )
            messageDao.insert(msg)
        }
    }
}
