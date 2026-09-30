package com.siddu.offlinedocai.domain.repository

import android.content.Context
import android.util.Log
import com.siddu.offlinedocai.ai.embeddings.EmbeddingModel
import com.siddu.offlinedocai.ai.embeddings.ONNXEmbeddingEngine
import com.siddu.offlinedocai.ai.llm.LocalLLM
import com.siddu.offlinedocai.ai.llm.OllamaLLM
import com.siddu.offlinedocai.ai.llm.GGUFLLM
import com.siddu.offlinedocai.ai.retrieval.AssembledContext
import com.siddu.offlinedocai.ai.retrieval.ContextAssembler
import com.siddu.offlinedocai.ai.retrieval.QueryClassification
import com.siddu.offlinedocai.ai.retrieval.QueryIntent
import com.siddu.offlinedocai.ai.retrieval.QueryIntentRouter
import com.siddu.offlinedocai.ai.retrieval.RetrievalResult
import com.siddu.offlinedocai.ai.retrieval.SourceBlock
import com.siddu.offlinedocai.ai.retrieval.VectorStore
import com.siddu.offlinedocai.data.AppDatabase
import com.siddu.offlinedocai.data.dao.MessageDao
import com.siddu.offlinedocai.data.entities.MessageEntity
import com.siddu.offlinedocai.data.entities.ConversationEntity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay
import java.util.UUID

/**
 * Repository that implements the RAG chat pipeline.
 */
class ChatRepository(
    private val context: Context,
    private val db: AppDatabase,
    private val vectorStore: VectorStore,
    private val useOllama: Boolean = false // dev flag – set false for final offline mode
) {
    private val TAG = "ChatRepository"

    private val messageDao: MessageDao = db.messageDao()
    private val conversationDao = db.conversationDao()
    private val embeddingEngine: EmbeddingModel = ONNXEmbeddingEngine(context)
    internal val llm: LocalLLM = if (useOllama) OllamaLLM() else GGUFLLM(context)

    private val _chatState = MutableStateFlow<ChatState>(ChatState.Idle)
    val chatState: StateFlow<ChatState> = _chatState

    /** Safely switch the underlying GGUF LLM model (e.g. Qwen 0.5B vs Llama 3.2 1B) */
    fun selectLlmModel(model: GGUFLLM.SupportedModel): Boolean {
        return if (llm is GGUFLLM) {
            llm.selectModel(model)
        } else false
    }

    fun getActiveLlmModel(): GGUFLLM.SupportedModel? {
        return (llm as? GGUFLLM)?.getActiveModel()
    }

    /** Create a new conversation, auto-generate welcome message, and return its ID */
    suspend fun startConversation(title: String = "Chat ${System.currentTimeMillis()}", documentId: Long? = null): Long = kotlinx.coroutines.withContext(Dispatchers.IO) {
        val conversation = ConversationEntity(
            id = 0,
            title = title,
            createdAt = System.currentTimeMillis(),
            documentId = documentId
        )
        val convId = conversationDao.insert(conversation)

        // Insert initial greeting message
        if (documentId != null && documentId > 0) {
            val doc = db.documentDao().getById(documentId)
            val docName = doc?.name ?: "Selected Document"
            val pageCount = doc?.pageCount ?: 0
            
            val welcomeText = """
👋 Welcome to your document assistant! 

I have parsed **$docName** ($pageCount pages), computed semantic embeddings, and successfully indexed the text chunks into local memory database.

### Document Summary:
This document is ready for local Retrieval-Augmented Generation (RAG) queries. You can ask me to search, summarize, or retrieve specific key details from it.

How may I help you with it?
""".trimIndent()
            
            messageDao.insert(
                MessageEntity(
                    id = 0,
                    conversationId = convId,
                    role = "assistant",
                    content = welcomeText,
                    timestamp = System.currentTimeMillis()
                )
            )
        } else {
            val welcomeText = """
👋 Hello! I am your local offline AI assistant. 

How may I help you today?

*Note: This is a general LLM chat session. To query a specific document using RAG, start a new conversation and check the document checkbox to load it as reference.*
""".trimIndent()
            
            messageDao.insert(
                MessageEntity(
                    id = 0,
                    conversationId = convId,
                    role = "assistant",
                    content = welcomeText,
                    timestamp = System.currentTimeMillis()
                )
            )
        }

        convId
    }

    /** Send a user message, run RAG or LLM, and stream the assistant response */
    fun sendMessage(conversationId: Long, userText: String, relevantDocumentIds: List<Long>) {
        CoroutineScope(Dispatchers.IO).launch {
            var timerJob: kotlinx.coroutines.Job? = null
            try {
                Log.i("RAG_DEBUG", "RAG_DEBUG_01_QUERY_START: query='$userText' convId=$conversationId docIds=$relevantDocumentIds")
                _chatState.value = ChatState.Retrieving
                // Persist user message
                val userMsg = MessageEntity(
                    id = 0,
                    conversationId = conversationId,
                    role = "user",
                    content = userText,
                    timestamp = System.currentTimeMillis()
                )
                messageDao.insert(userMsg)

                // Clean and filter valid document IDs (greater than 0)
                val cleanDocIds = relevantDocumentIds.filter { it > 0 }

                // Build prompt based on whether valid documents are linked
                var assembledContext: com.siddu.offlinedocai.ai.retrieval.AssembledContext? = null
                val (prompt, sourcesData) = if (cleanDocIds.isEmpty()) {
                    Pair(buildNormalPrompt(userText), null)
                } else {
                    // Embedding of the query
                    Log.i("RAG_DEBUG", "RAG_DEBUG_02_EMBEDDING_START")
                    val queryEmbedding = embeddingEngine.getEmbedding(userText)
                    Log.i("RAG_DEBUG", "RAG_DEBUG_03_EMBEDDING_DONE: size=${queryEmbedding.size}")
                    val queryClassification = QueryIntentRouter.route(userText)

                    // Retrieve detailed candidates (top-20 for adequate candidate pool)
                    val retrieved = vectorStore.retrieveDetailed(
                        queryEmbedding = queryEmbedding,
                        relevantDocumentIds = cleanDocIds,
                        topK = 20,
                        query = userText
                    )
                    Log.i("RAG_DEBUG", "RAG_DEBUG_04_RETRIEVAL_DONE: count=${retrieved.size} pages=${retrieved.map { it.chunk.pageNumber }.distinct()}")

                    val evidenceBudget = ContextAssembler.resolveEvidenceBudget(queryClassification.intent)
                    val assembled = ContextAssembler.assemble(
                        retrievedResults = retrieved,
                        queryIntent = queryClassification.intent,
                        queryClassification = queryClassification,
                        maxEvidenceTokens = evidenceBudget
                    )
                    assembledContext = assembled
                    Log.i("RAG_DEBUG", "RAG_DEBUG_05_CONTEXT_DONE: tokens=${assembled.totalEstimatedTokens} chars=${assembled.contextText.length}")

                    // Build prompt
                    val p = buildPrompt(assembled.contextText, userText)
                    Pair(p, assembled.sourcesJson)
                }

                _chatState.value = ChatState.Generating(0)
                timerJob = CoroutineScope(Dispatchers.Default).launch {
                    var elapsed = 0
                    while (true) {
                        delay(1000)
                        elapsed++
                        if (_chatState.value is ChatState.Generating) {
                            _chatState.value = ChatState.Generating(elapsed)
                        } else {
                            break
                        }
                    }
                }

                // Generate full answer from GGUFLLM
                val answer = if (assembledContext != null && assembledContext.sourceBlocks.isEmpty() && assembledContext.contextText.startsWith("[")) {
                    // Grounded response for negative/empty retrieval
                    when {
                        assembledContext.contextText.startsWith("[PAGE_NOT_FOUND:") -> {
                            val p = assembledContext.contextText.removePrefix("[PAGE_NOT_FOUND:").removeSuffix("]").trim()
                            "I couldn't find page $p in the provided document."
                        }
                        assembledContext.contextText.startsWith("[PAGE_RANGE_NOT_FOUND:") -> {
                            val r = assembledContext.contextText.removePrefix("[PAGE_RANGE_NOT_FOUND:").removeSuffix("]").trim()
                            "I couldn't find pages $r in the provided document."
                        }
                        else -> {
                            "I searched the provided document(s), but could not find information answering \"$userText\"."
                        }
                    }
                } else {
                    val rawAnswer = try {
                        val deferred = kotlinx.coroutines.CoroutineScope(Dispatchers.Default).async {
                            llm.generate(prompt)
                        }
                        kotlinx.coroutines.withTimeout(120_000L) {
                            deferred.await()
                        }
                    } catch (timeout: kotlinx.coroutines.TimeoutCancellationException) {
                        Log.e("RAG_DEBUG", "LLM inference exceeded 120s threshold on device")
                        (llm as? GGUFLLM)?.cancel()
                        throw IllegalStateException("Model generation timed out after 120 seconds. Please try again with a shorter query.")
                    } catch (t: Throwable) {
                        Log.e("RAG_DEBUG", "LLM inference encountered error: ${t.message}", t)
                        throw t
                    }
                    Log.i("RAG_DEBUG", "RAG_DEBUG_10_REPOSITORY_RETURN: len=${rawAnswer.length}")
                    if (rawAnswer.isBlank()) {
                        throw IllegalStateException("Local neural model produced an empty answer.")
                    }
                    rawAnswer
                }

                timerJob?.cancel()

                // Insert empty assistant message to initialize streaming updates
                val assistantMsgId = messageDao.insert(
                    MessageEntity(
                        id = 0,
                        conversationId = conversationId,
                        role = "assistant",
                        content = "",
                        timestamp = System.currentTimeMillis(),
                        sources = sourcesData
                    )
                )

                // Simulate token/word streaming updates
                val words = answer.split(" ")
                var currentContent = ""
                for (i in words.indices) {
                    currentContent += (if (i > 0) " " else "") + words[i]
                    messageDao.update(
                        MessageEntity(
                            id = assistantMsgId,
                            conversationId = conversationId,
                            role = "assistant",
                            content = currentContent,
                            timestamp = System.currentTimeMillis(),
                            sources = sourcesData
                        )
                    )
                    delay(30) // speed of stream
                }

                Log.i("RAG_DEBUG", "RAG_DEBUG_11_SUCCESS_STATE: answerLen=${answer.length}")
                _chatState.value = ChatState.Idle
            } catch (t: Throwable) {
                timerJob?.cancel()
                Log.e("RAG_DEBUG", "RAG_DEBUG_12_ERROR_STATE: error=${t.message}", t)
                Log.e(TAG, "Chat generation failed", t)
                val errorMsg = t.message ?: "Unknown error"
                try {
                    messageDao.insert(
                        MessageEntity(
                            id = 0,
                            conversationId = conversationId,
                            role = "assistant",
                            content = "An error occurred while answering: $errorMsg",
                            timestamp = System.currentTimeMillis()
                        )
                    )
                } catch (insertError: Throwable) {
                    Log.e(TAG, "Failed to insert error message into conversation", insertError)
                }
                _chatState.value = ChatState.Error(errorMsg)
            } finally {
                timerJob?.cancel()
                if (_chatState.value is ChatState.Generating || _chatState.value is ChatState.Retrieving || _chatState.value is ChatState.Thinking) {
                    _chatState.value = ChatState.Idle
                }
            }
        }
    }

    private fun buildNormalPrompt(question: String): String {
        return """
You are a helpful offline AI assistant. Answer the user's question directly.

Question: $question

Answer:
""".trimIndent()
    }

    private fun buildPrompt(context: String, question: String): String {
        return """
You are an offline AI assistant. Use the following extracted document context to answer the user's question.

Context:
$context

Question: $question

Answer concisely, referencing the page numbers when appropriate.
""".trimIndent()
    }

    /** Expose context assembly for testing and direct pipeline queries */
    fun assembleContext(
        retrievedResults: List<com.siddu.offlinedocai.ai.retrieval.RetrievalResult>,
        queryIntent: com.siddu.offlinedocai.ai.retrieval.QueryIntent,
        queryClassification: QueryClassification? = null,
        maxEvidenceTokens: Int = ContextAssembler.DEFAULT_MAX_CONTEXT_TOKENS
    ): AssembledContext {
        return com.siddu.offlinedocai.ai.retrieval.ContextAssembler.assemble(
            retrievedResults = retrievedResults,
            queryIntent = queryIntent,
            queryClassification = queryClassification,
            maxEvidenceTokens = maxEvidenceTokens
        )
    }

    sealed class ChatState {
        object Idle : ChatState()
        object Thinking : ChatState()
        object Retrieving : ChatState()
        data class Generating(val elapsedSeconds: Int = 0) : ChatState()
        data class Error(val message: String) : ChatState()
    }
}
