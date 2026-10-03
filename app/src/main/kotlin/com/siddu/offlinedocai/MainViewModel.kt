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
import com.siddu.offlinedocai.ai.llm.AnswerLength
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
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

    val modelManager = com.siddu.offlinedocai.data.ai.ModelManager.getInstance(application)
    val installedModels: StateFlow<List<com.siddu.offlinedocai.data.ai.ModelItem>> = modelManager.models
    val activeLlmModelItem: StateFlow<com.siddu.offlinedocai.data.ai.ModelItem?> = modelManager.activeModel
    val modelImportProgress: StateFlow<com.siddu.offlinedocai.data.ai.ImportProgress?> = modelManager.importProgress
    val modelImportError: StateFlow<String?> = modelManager.importError

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
        // Synchronize persisted active model with ChatRepository
        modelManager.activeModel.value?.let {
            chatRepository.selectLlmModel(it.file)
        }
    }

    enum class ThemeMode { SYSTEM, LIGHT, DARK }
    private val _themeMode = MutableStateFlow(ThemeMode.SYSTEM)
    val themeMode: StateFlow<ThemeMode> = _themeMode.asStateFlow()

    fun setThemeMode(mode: ThemeMode) {
        _themeMode.value = mode
    }

    fun checkModels() {
        modelManager.refreshModels()
        val app = getApplication<Application>()
        // 1. Check local ONNX embedding model (or bundled in assets)
        val embedOnnxDir = File(app.filesDir, "models/embedding")
        val hasLocalEmbed = File(embedOnnxDir, "all-MiniLM-L6-v2.onnx").exists()
        val hasAssetEmbed = try {
            app.assets.list("models/embedding")?.contains("all-MiniLM-L6-v2.onnx") == true
        } catch (_: Exception) { false }
        isEmbeddingModelDownloaded.value = hasLocalEmbed || hasAssetEmbed

        // 2. Check local GGUF LLM models
        val allLlmFiles = modelManager.models.value.map { it.file }
        downloadedLlmModels.value = allLlmFiles
        isLlmModelDownloaded.value = allLlmFiles.isNotEmpty()
        val active = modelManager.activeModel.value
        selectedLlmModel.value = active?.file ?: allLlmFiles.firstOrNull()
        if (active != null) {
            chatRepository.selectLlmModel(active.file)
        }
    }

    fun importLlmModel(uri: Uri) {
        viewModelScope.launch {
            val result = modelManager.importModelFromUri(uri)
            if (result.isSuccess) {
                val item = result.getOrNull()
                if (item != null) {
                    selectLlmModel(item.file)
                }
            }
            checkModels()
        }
    }

    fun selectLlmModel(file: File) {
        selectedLlmModel.value = file
        modelManager.setActiveModel(file.name)
        chatRepository.selectLlmModel(file)
    }

    fun selectSupportedModel(model: com.siddu.offlinedocai.ai.llm.GGUFLLM.SupportedModel): Boolean {
        val ok = chatRepository.selectLlmModel(model)
        if (ok) {
            modelManager.setActiveModel(model.filename)
            checkModels()
        }
        return ok
    }

    fun deleteLlmModel(file: File) {
        viewModelScope.launch {
            val removed = modelManager.removeModel(file.name)
            if (removed) {
                checkModels()
            }
        }
    }

    fun deleteModelItem(item: com.siddu.offlinedocai.data.ai.ModelItem) {
        viewModelScope.launch {
            val removed = modelManager.removeModel(item.id)
            if (removed) {
                checkModels()
            }
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

    private val _selectedAnswerLength = MutableStateFlow(AnswerLength.SHORT)
    val selectedAnswerLength: StateFlow<AnswerLength> = _selectedAnswerLength.asStateFlow()

    fun setAnswerLength(length: AnswerLength) {
        _selectedAnswerLength.value = length
    }

    /** Send a message in a conversation */
    fun sendMessage(
        convId: Long,
        text: String,
        docIds: List<Long>,
        answerLength: AnswerLength = _selectedAnswerLength.value
    ) {
        chatRepository.sendMessage(convId, text, docIds, answerLength)
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
