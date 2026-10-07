package com.siddu.offlinedocai

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.siddu.offlinedocai.ai.embeddings.AvailableEmbeddingModels
import com.siddu.offlinedocai.ai.embeddings.EmbeddingManager
import com.siddu.offlinedocai.ai.embeddings.EmbeddingModelStatus
import com.siddu.offlinedocai.ai.llm.AnswerLength
import com.siddu.offlinedocai.ai.llm.AvailableLlmModels
import com.siddu.offlinedocai.ai.llm.GGUFLLM
import com.siddu.offlinedocai.ai.llm.LlamaNative
import com.siddu.offlinedocai.ai.llm.LlmLoadState
import com.siddu.offlinedocai.ai.retrieval.VectorStore
import com.siddu.offlinedocai.data.AppDatabase
import com.siddu.offlinedocai.data.ai.ModelItem
import com.siddu.offlinedocai.data.ai.ModelManager
import com.siddu.offlinedocai.data.ai.ModelStatus
import com.siddu.offlinedocai.data.download.DownloadEvent
import com.siddu.offlinedocai.data.download.DownloadStatus
import com.siddu.offlinedocai.data.download.EnqueueResult
import com.siddu.offlinedocai.data.download.ModelDownloadManager
import com.siddu.offlinedocai.data.entities.ConversationEntity
import com.siddu.offlinedocai.data.entities.DocumentEntity
import com.siddu.offlinedocai.data.entities.MessageEntity
import com.siddu.offlinedocai.domain.repository.ChatRepository
import com.siddu.offlinedocai.domain.repository.DocumentRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.firstOrNull
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

/** A model that just finished downloading and may be switched to. */
data class PendingModelSwitch(val id: String, val name: String, val isEmbedding: Boolean)

/**
 * Central ViewModel for the OfflineDocAI app.
 *
 * Model state comes from process-wide singletons (ModelManager, GGUFLLM, EmbeddingManager,
 * ModelDownloadManager), so it is the same everywhere and survives the Activity being recreated.
 */
class MainViewModel(application: Application) : AndroidViewModel(application) {
    private val database: AppDatabase = AppDatabase.getInstance(application)

    private val documentDao = database.documentDao()
    private val conversationDao = database.conversationDao()
    private val messageDao = database.messageDao()

    private val _documents = MutableStateFlow<List<DocumentEntity>>(emptyList())
    val documents: StateFlow<List<DocumentEntity>> = _documents

    private val _conversations = MutableStateFlow<List<ConversationEntity>>(emptyList())
    val conversations: StateFlow<List<ConversationEntity>> = _conversations

    // --- Models -----------------------------------------------------------------------------------
    val modelManager = ModelManager.getInstance(application)
    private val llm = GGUFLLM.getInstance(application)
    val embeddingManager = EmbeddingManager.getInstance(application)
    private val downloads = ModelDownloadManager.getInstance(application)

    val installedModels: StateFlow<List<ModelItem>> = modelManager.models
    val activeLlmModelItem: StateFlow<ModelItem?> = modelManager.activeModel
    val llmLoadState: StateFlow<LlmLoadState> = llm.loadState
    val modelImportProgress = modelManager.importProgress
    val modelImportError: StateFlow<String?> = modelManager.importError

    val downloadStatuses: StateFlow<Map<String, DownloadStatus>> = downloads.statuses
    val embeddingStatuses: StateFlow<Map<String, EmbeddingModelStatus>> = embeddingManager.statuses
    val activeEmbeddingId: StateFlow<String> = embeddingManager.activeId

    /** Shown after a download completes while another model is active ("Use it now?"). */
    private val _pendingSwitch = MutableStateFlow<PendingModelSwitch?>(null)
    val pendingSwitch: StateFlow<PendingModelSwitch?> = _pendingSwitch.asStateFlow()

    /** One-off messages for a snackbar. */
    private val _userMessage = MutableStateFlow<String?>(null)
    val userMessage: StateFlow<String?> = _userMessage.asStateFlow()
    fun consumeUserMessage() { _userMessage.value = null }

    private val _embeddingSwitchInProgress = MutableStateFlow<String?>(null)
    val embeddingSwitchInProgress: StateFlow<String?> = _embeddingSwitchInProgress.asStateFlow()

    val cpuVariant: String get() = LlamaNative.cpuVariant
    val compatibilityModeActive: Boolean get() = LlamaNative.compatibilityModeActive
    val compatibilityReason: String? get() = LlamaNative.compatibilityReason
    private val _compatibilityPreferred = MutableStateFlow(LlamaNative.isCompatibilityModePreferred(application))
    val compatibilityPreferred: StateFlow<Boolean> = _compatibilityPreferred.asStateFlow()

    // Question Queue Manager for batch/sequential question execution
    val questionQueueManager = com.siddu.offlinedocai.ai.queue.QuestionQueueManager(viewModelScope)
    val questionQueueState: StateFlow<com.siddu.offlinedocai.ai.queue.QuestionQueueState> = questionQueueManager.queueState

    // Repositories
    private val documentRepository = DocumentRepository(application, database)
    private val vectorStore = VectorStore(database.chunkDao())
    private val chatRepository = ChatRepository(application, database, vectorStore, viewModelScope)

    val importState: StateFlow<DocumentRepository.ImportState> = documentRepository.importState
    val chatState: StateFlow<ChatRepository.ChatState> = chatRepository.chatState

    init {
        viewModelScope.launch {
            documentDao.getAllDocuments().collect { _documents.value = it }
        }
        viewModelScope.launch {
            conversationDao.getAllConversations().collect { _conversations.value = it }
        }
        viewModelScope.launch {
            downloads.events.collect { onDownloadEvent(it) }
        }
        // Startup work happens off the main thread: model scan (GGUF headers), partial-download cleanup,
        // search-model health check, then re-indexing and warming up the language model.
        viewModelScope.launch(Dispatchers.IO) {
            modelManager.refresh()
            llm.onActiveModelChanged()
            downloads.cleanupAndRestore()
            if (!embeddingManager.ensureActiveReady()) {
                val spec = embeddingManager.activeSpec
                _userMessage.value = "The search model \"${spec.name}\" does not work on this device. " +
                    "Open Settings → Search model and download the compatibility version."
            }
            documentRepository.reindexStaleDocuments()
            if (modelManager.getActiveModel() != null) llm.warmUp()
        }
    }

    enum class ThemeMode { SYSTEM, LIGHT, DARK }
    private val _themeMode = MutableStateFlow(ThemeMode.SYSTEM)
    val themeMode: StateFlow<ThemeMode> = _themeMode.asStateFlow()

    fun setThemeMode(mode: ThemeMode) {
        _themeMode.value = mode
    }

    // --- Language models ----------------------------------------------------------------------------

    private suspend fun onDownloadEvent(event: DownloadEvent) {
        when (event) {
            is DownloadEvent.Completed -> if (event.isEmbedding) {
                embeddingManager.refresh()
                val spec = AvailableEmbeddingModels.byId(event.id) ?: return
                _pendingSwitch.value = PendingModelSwitch(spec.id, spec.name, isEmbedding = true)
            } else {
                modelManager.refresh()
                val spec = AvailableLlmModels.list.firstOrNull { it.id == event.id } ?: return
                // The first installed model becomes active automatically; otherwise ask before switching.
                val active = modelManager.getActiveModel()
                if (active == null || active.filename.equals(spec.fileName, ignoreCase = true)) {
                    selectLlmModelByFileName(spec.fileName)
                } else {
                    _pendingSwitch.value = PendingModelSwitch(spec.id, spec.name, isEmbedding = false)
                }
            }
            is DownloadEvent.Failed -> Unit // shown on the model's own card
        }
    }

    fun dismissPendingSwitch() { _pendingSwitch.value = null }

    fun confirmPendingSwitch() {
        val pending = _pendingSwitch.value ?: return
        _pendingSwitch.value = null
        if (pending.isEmbedding) activateEmbeddingModel(pending.id)
        else AvailableLlmModels.list.firstOrNull { it.id == pending.id }?.let { selectLlmModelByFileName(it.fileName) }
    }

    private fun selectLlmModelByFileName(fileName: String) {
        val item = modelManager.models.value.firstOrNull { it.filename.equals(fileName, ignoreCase = true) } ?: return
        selectLlmModel(item)
    }

    /** Makes [item] the active language model and loads it in the background; the load result is shown in Settings. */
    fun selectLlmModel(item: ModelItem) {
        if (item.status == ModelStatus.ERROR) {
            _userMessage.value = "This model file is damaged or unsupported: ${item.errorMessage ?: "invalid GGUF"}. Delete it and download it again."
            return
        }
        if (!modelManager.setActiveModel(item.id)) {
            _userMessage.value = "Could not select ${item.displayName}."
            return
        }
        llm.onActiveModelChanged()
        viewModelScope.launch(Dispatchers.Default) { llm.warmUp() }
    }

    fun retryLoadLlm() {
        viewModelScope.launch(Dispatchers.Default) { llm.warmUp() }
    }

    fun importLlmModel(uri: Uri) {
        viewModelScope.launch {
            val result = modelManager.importModelFromUri(uri) { target ->
                if (llm.loadedModelPath() == target.absolutePath) llm.release()
            }
            result.onSuccess { item ->
                modelManager.refresh()
                modelManager.models.value.firstOrNull { it.id == item.id }?.let { selectLlmModel(it) }
            }
        }
    }

    /** Deletes a language model: refused while generating; the native model is released first so RAM and disk are freed. */
    fun deleteModelItem(item: ModelItem) {
        viewModelScope.launch {
            if (modelManager.isGenerationActive()) {
                _userMessage.value = "Wait for the current answer to finish before deleting this model."
                return@launch
            }
            if (llm.loadedModelPath() == item.file.absolutePath) llm.release()
            if (!modelManager.removeModel(item.id)) {
                _userMessage.value = "Could not delete ${item.displayName}."
                return@launch
            }
            modelManager.refresh()
            llm.onActiveModelChanged()
            if (modelManager.getActiveModel() != null) launch(Dispatchers.Default) { llm.warmUp() }
        }
    }

    // --- Downloads (language and search models) -----------------------------------------------------

    fun downloadModel(id: String) {
        when (downloads.enqueue(id)) {
            EnqueueResult.ALREADY_ACTIVE -> _userMessage.value = "This model is already downloading."
            EnqueueResult.ALREADY_INSTALLED -> _userMessage.value = "This model is already installed."
            EnqueueResult.UNKNOWN_MODEL -> _userMessage.value = "Unknown model."
            EnqueueResult.NOT_ENOUGH_SPACE, EnqueueResult.STARTED -> Unit // shown on the card
        }
    }

    fun cancelDownload(id: String) = downloads.cancel(id)

    fun dismissDownloadStatus(id: String) = downloads.dismiss(id)

    // --- Search (embedding) models ------------------------------------------------------------------

    /** Number of documents that will be re-indexed if the search model changes. */
    fun documentsToReindex(): Int = _documents.value.count { it.status != DocumentRepository.STATUS_FAILED }

    fun activateEmbeddingModel(id: String) {
        if (_embeddingSwitchInProgress.value != null) return
        _embeddingSwitchInProgress.value = id
        viewModelScope.launch {
            val result = embeddingManager.activate(id)
            _embeddingSwitchInProgress.value = null
            result.onSuccess {
                val name = AvailableEmbeddingModels.byId(id)?.name ?: id
                val count = documentsToReindex()
                _userMessage.value = if (count > 0) "Search model switched to $name. Re-indexing $count document(s)…" else "Search model switched to $name."
                documentRepository.reindexStaleDocuments()
            }.onFailure { e ->
                _userMessage.value = "This search model cannot be used on this device: ${e.message}"
            }
        }
    }

    fun deleteEmbeddingModel(id: String) {
        viewModelScope.launch {
            embeddingManager.delete(id).onFailure { _userMessage.value = it.message }
        }
    }

    fun setCompatibilityMode(enabled: Boolean) {
        LlamaNative.setCompatibilityModePreferred(getApplication(), enabled)
        _compatibilityPreferred.value = enabled
        _userMessage.value = "Restart the app to apply this setting."
    }

    // --- Chat ---------------------------------------------------------------------------------------

    /** Start sequential processing of multiple questions; each waits for the previous answer. */
    fun startQuestionQueue(convId: Long, questions: List<String>, docIds: List<Long>) {
        questionQueueManager.startQueue(convId, questions, docIds) { targetConvId, questionText, targetDocIds ->
            chatRepository.ask(targetConvId, questionText, targetDocIds, _selectedAnswerLength.value)
        }
    }

    fun pauseQuestionQueue() = questionQueueManager.pauseQueue()
    fun resumeQuestionQueue() = questionQueueManager.resumeQueue()
    fun stopQuestionQueue() {
        questionQueueManager.stopQueue()
        chatRepository.stopGeneration()
    }
    fun clearQuestionQueue() = questionQueueManager.clearQueue()

    fun parseQuestions(rawInput: String): List<String> =
        com.siddu.offlinedocai.ai.queue.QuestionQueueParser.parseQuestions(rawInput)

    fun importDocument(uri: Uri) = documentRepository.importDocument(uri)

    fun reprocessDocument(document: DocumentEntity) = documentRepository.reprocessDocument(document)

    /** Start a new conversation linked to selected document IDs */
    suspend fun startConversation(title: String = "New Chat", documentIds: List<Long> = emptyList()): Long =
        chatRepository.startConversation(title, documentIds.firstOrNull())

    private val _selectedAnswerLength = MutableStateFlow(AnswerLength.DEFAULT)
    val selectedAnswerLength: StateFlow<AnswerLength> = _selectedAnswerLength.asStateFlow()

    fun setAnswerLength(length: AnswerLength) {
        _selectedAnswerLength.value = length
    }

    fun sendMessage(
        convId: Long,
        text: String,
        docIds: List<Long>,
        answerLength: AnswerLength = _selectedAnswerLength.value
    ) {
        chatRepository.sendMessage(convId, text, docIds, answerLength)
    }

    /** Stops the answer being generated (the text so far is kept). */
    fun stopGeneration() = chatRepository.stopGeneration()

    /** Deletes the document, all of its chunks/embeddings and its private file copy. */
    fun deleteDocument(documentId: Long) {
        viewModelScope.launch { documentRepository.deleteDocument(documentId) }
    }

    fun deleteConversation(convId: Long) {
        viewModelScope.launch {
            messageDao.deleteMessagesForConversation(convId)
            conversationDao.getById(convId)?.let { conversationDao.delete(it) }
        }
    }

    fun renameConversation(convId: Long, newTitle: String) {
        viewModelScope.launch {
            conversationDao.updateTitle(convId, newTitle.trim())
        }
    }

    fun shareConversationPdf(context: android.content.Context, convId: Long) {
        viewModelScope.launch {
            val conv = conversationDao.getById(convId) ?: return@launch
            val messages = messageDao.getMessagesForConversation(convId).firstOrNull() ?: emptyList()
            kotlinx.coroutines.withContext(Dispatchers.Main) {
                com.siddu.offlinedocai.ui.PdfExporter.exportAndShareConversation(
                    context = context,
                    conversationTitle = conv.title,
                    messages = messages
                )
            }
        }
    }

    fun messagesForConversation(convId: Long): StateFlow<List<MessageEntity>> =
        messageDao.getMessagesForConversation(convId).stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())
}
