package com.siddu.offlinedocai

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.room.Room
import com.siddu.offlinedocai.data.AppDatabase
import com.siddu.offlinedocai.data.entities.DocumentEntity
import com.siddu.offlinedocai.data.entities.ConversationEntity
import com.siddu.offlinedocai.data.entities.MessageEntity
import com.siddu.offlinedocai.domain.repository.DocumentRepository
import com.siddu.offlinedocai.domain.repository.ChatRepository
import com.siddu.offlinedocai.domain.repository.ModelDownloader
import com.siddu.offlinedocai.domain.repository.DownloadState
import com.siddu.offlinedocai.ai.retrieval.VectorStore
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import java.io.File

/**
 * Central ViewModel for the OfflineDocAI app.
 * Provides access to the Room database and exposes StateFlows for UI observation.
 */
class MainViewModel(application: Application) : AndroidViewModel(application) {
    private val database: AppDatabase = Room.databaseBuilder(
        application,
        AppDatabase::class.java,
        "offline_docai.db"
    ).addMigrations(AppDatabase.MIGRATION_1_2).fallbackToDestructiveMigration().build()

    private val documentDao = database.documentDao()
    private val conversationDao = database.conversationDao()
    private val messageDao = database.messageDao()

    private val _documents = MutableStateFlow<List<DocumentEntity>>(emptyList())
    val documents: StateFlow<List<DocumentEntity>> = _documents

    private val _conversations = MutableStateFlow<List<ConversationEntity>>(emptyList())
    val conversations: StateFlow<List<ConversationEntity>> = _conversations

    val isEmbeddingModelDownloaded = MutableStateFlow(false)
    val isLlmModelDownloaded = MutableStateFlow(false)

    private val embeddingDownloader = ModelDownloader(application)
    val embeddingDownloadState: StateFlow<DownloadState> = embeddingDownloader.downloadState

    private val llmDownloader = ModelDownloader(application)
    val llmDownloadState: StateFlow<DownloadState> = llmDownloader.downloadState

    val downloadedLlmModels = MutableStateFlow<List<File>>(emptyList())
    val selectedLlmModel = MutableStateFlow<File?>(null)

    // Repositories
    private val documentRepository = DocumentRepository(application, database)
    private val vectorStore = VectorStore(database.chunkDao())
    private val chatRepository = ChatRepository(application, database, vectorStore)

    val importState: StateFlow<DocumentRepository.ImportState> = documentRepository.importState
    val chatState: StateFlow<ChatRepository.ChatState> = chatRepository.chatState

    init {
        // Load existing data from DB
        viewModelScope.launch {
            documentDao.getAllDocuments().collect { _documents.value = it }
        }
        viewModelScope.launch {
            conversationDao.getAllConversations().collect { _conversations.value = it }
        }
        checkModels()
    }

    fun checkModels() {
        val app = getApplication<Application>()
        // 1. Check local ONNX embedding model (or bundled in assets)
        val embedOnnxDir = File(app.filesDir, "models/embedding")
        val hasLocalEmbed = File(embedOnnxDir, "all-MiniLM-L6-v2.onnx").exists()
        val hasAssetEmbed = try {
            app.assets.list("models/embedding")?.contains("all-MiniLM-L6-v2.onnx") == true
        } catch (_: Exception) { false }
        isEmbeddingModelDownloaded.value = hasLocalEmbed || hasAssetEmbed

        // 2. Check local GGUF LLM model (or bundled in assets)
        val llmDir = File(app.filesDir, "models/llm")
        val llmFiles = llmDir.listFiles()?.filter { it.extension.equals("gguf", ignoreCase = true) } ?: emptyList()
        val hasAssetLlm = try {
            app.assets.list("models/llm")?.any { it.endsWith(".gguf", ignoreCase = true) } == true
        } catch (_: Exception) { false }

        downloadedLlmModels.value = llmFiles
        isLlmModelDownloaded.value = llmFiles.isNotEmpty() || hasAssetLlm
        if (selectedLlmModel.value == null || !selectedLlmModel.value!!.exists()) {
            selectedLlmModel.value = llmFiles.firstOrNull { it.name.contains("1.5b", ignoreCase = true) }
                ?: llmFiles.firstOrNull()
        }
    }

    fun selectLlmModel(file: File) {
        selectedLlmModel.value = file
        val model = com.siddu.offlinedocai.ai.llm.GGUFLLM.SupportedModel.fromFilename(file.name)
        if (model != null) {
            chatRepository.selectLlmModel(model)
        }
    }

    fun selectSupportedModel(model: com.siddu.offlinedocai.ai.llm.GGUFLLM.SupportedModel): Boolean {
        return chatRepository.selectLlmModel(model)
    }

    fun deleteLlmModel(file: File) {
        viewModelScope.launch {
            if (file.exists()) {
                file.delete()
            }
            checkModels()
        }
    }

    fun downloadEmbeddingModel() {
        // Embedding model is bundled locally in assets; verify extraction
        checkModels()
    }

    fun downloadLlmModel(urlStr: String, fileName: String) {
        val destFile = File(getApplication<Application>().filesDir, "models/llm/$fileName")
        viewModelScope.launch {
            llmDownloader.downloadModel(urlStr, destFile)
            checkModels()
        }
    }

    fun resetLlmDownloadState() {
        llmDownloader.resetState()
    }

    fun resetEmbeddingDownloadState() {
        embeddingDownloader.resetState()
    }

    /** Trigger document import */
    fun importDocument(uri: Uri) {
        documentRepository.importDocument(uri)
    }

    /** Start a new conversation linked to selected document IDs */
    suspend fun startConversation(title: String = "New Chat", documentIds: List<Long> = emptyList()): Long {
        return chatRepository.startConversation(title, documentIds.firstOrNull())
    }

    /** Send a message in a conversation */
    fun sendMessage(convId: Long, text: String, docIds: List<Long>) {
        chatRepository.sendMessage(convId, text, docIds)
    }

    // Delete a document
    fun deleteDocument(documentId: Long) {
        viewModelScope.launch {
            documentDao.getById(documentId)?.let {
                documentDao.delete(it)
            }
        }
    }

    // Delete a conversation and its messages
    fun deleteConversation(convId: Long) {
        viewModelScope.launch {
            conversationDao.getById(convId)?.let {
                conversationDao.delete(it)
            }
            messageDao.deleteMessagesForConversation(convId)
        }
    }

    // Get messages for a conversation as StateFlow
    fun messagesForConversation(convId: Long): StateFlow<List<MessageEntity>> {
        return messageDao.getMessagesForConversation(convId).stateIn(viewModelScope, SharingStarted.Eagerly, emptyList())
    }
}
