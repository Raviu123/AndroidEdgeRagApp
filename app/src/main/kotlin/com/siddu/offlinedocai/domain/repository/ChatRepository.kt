package com.siddu.offlinedocai.domain.repository

import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.siddu.offlinedocai.ai.embeddings.EmbeddingModel
import com.siddu.offlinedocai.ai.embeddings.ONNXEmbeddingEngine
import com.siddu.offlinedocai.ai.llm.LocalLLM
import com.siddu.offlinedocai.ai.llm.OllamaLLM
import com.siddu.offlinedocai.ai.llm.GGUFLLM
import com.siddu.offlinedocai.ai.llm.AnswerLength
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
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
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

    fun selectLlmModel(file: java.io.File): Boolean {
        return if (llm is GGUFLLM) {
            llm.selectModel(file)
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

        /** Send a user message, run RAG or LLM, and persist the final assistant response */
    fun sendMessage(
        conversationId: Long,
        userText: String,
        relevantDocumentIds: List<Long>,
        answerLength: AnswerLength = AnswerLength.SHORT
    ) {
        CoroutineScope(Dispatchers.IO).launch {
            val requestStartedAt = SystemClock.elapsedRealtime()
            var timerJob: kotlinx.coroutines.Job? = null
            try {
                Log.i("CHAT_LATENCY", "request_start convId=$conversationId docCount=${relevantDocumentIds.size} mode=$answerLength")
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
                Log.i("CHAT_LATENCY", "user_message_persist_ms=${SystemClock.elapsedRealtime() - requestStartedAt}")

                // Clean and filter valid document IDs (greater than 0)
                val cleanDocIds = relevantDocumentIds.filter { it > 0 }

                _chatState.value = ChatState.Generating(0)
                timerJob = launch(Dispatchers.Default) {
                    var elapsed = 0
                    try {
                        while (isActive) {
                            delay(1000)
                            elapsed++
                            if (_chatState.value is ChatState.Generating) {
                                _chatState.value = ChatState.Generating(elapsed)
                            } else {
                                break
                            }
                        }
                    } catch (_: CancellationException) {
                        // Parent coroutine is ending; the UI busy state is cleared elsewhere.
                    }
                }

                // ─────────────────────────────────────────────────────────────
                // ROUTING: no document selected → pure conversational LLM call
                //          document(s) selected → existing RAG pipeline
                // ─────────────────────────────────────────────────────────────
                val answer: String
                val sourcesData: String?

                if (cleanDocIds.isEmpty()) {
                    // Normal chat — no document context.
                    // Call generateConversational() which bypasses:
                    //   parsePrompt → parseSourceBlocks → CitationValidator.validate
                    // None of those must ever execute on a no-document message.
                    Log.i("RAG_DEBUG", "RAG_DEBUG_NORMAL_CHAT: query='$userText'")
                    sourcesData = null
                    val generationStartedAt = SystemClock.elapsedRealtime()
                    answer = try {
                        val deferred = kotlinx.coroutines.CoroutineScope(Dispatchers.Default).async {
                            (llm as? GGUFLLM)?.generateConversational(userText, answerLength)
                                ?: llm.generate(buildNormalPrompt(userText, answerLength), answerLength)
                        }
                        kotlinx.coroutines.withTimeout(120_000L) { deferred.await() }
                    } catch (timeout: kotlinx.coroutines.TimeoutCancellationException) {
                        Log.e("RAG_DEBUG", "LLM inference exceeded 120s threshold on device")
                        (llm as? GGUFLLM)?.cancel()
                        throw IllegalStateException("Model generation timed out after 120 seconds. Please try again with a shorter query.")
                    } catch (t: Throwable) {
                        Log.e("RAG_DEBUG", "Conversational LLM error: ${t.message}", t)
                        throw t
                    }
                    Log.i("CHAT_LATENCY", "path=no_document llm_total_ms=${SystemClock.elapsedRealtime() - generationStartedAt}")
                } else {
                    // RAG path — document(s) explicitly selected. Full pipeline unchanged.
                    Log.i("RAG_DEBUG", "RAG_DEBUG_02_EMBEDDING_START")
                    val embeddingStartedAt = SystemClock.elapsedRealtime()
                    val queryEmbedding = embeddingEngine.getEmbedding(userText)
                    Log.i("CHAT_LATENCY", "rag_query_embedding_ms=${SystemClock.elapsedRealtime() - embeddingStartedAt}")
                    Log.i("RAG_DEBUG", "RAG_DEBUG_03_EMBEDDING_DONE: size=${queryEmbedding.size}")
                    val queryClassification = QueryIntentRouter.route(userText)

                    val retrievalStartedAt = SystemClock.elapsedRealtime()
                    val retrieved = vectorStore.retrieveDetailed(
                        queryEmbedding = queryEmbedding,
                        relevantDocumentIds = cleanDocIds,
                        topK = 20,
                        query = userText
                    )
                    Log.i("CHAT_LATENCY", "rag_retrieval_total_ms=${SystemClock.elapsedRealtime() - retrievalStartedAt} results=${retrieved.size}")
                    Log.i("RAG_DEBUG", "RAG_DEBUG_04_RETRIEVAL_DONE: count=${retrieved.size} pages=${retrieved.map { it.chunk.pageNumber }.distinct()}")

                    val evidenceBudget = ContextAssembler.resolveEvidenceBudget(queryClassification.intent)
                    val contextStartedAt = SystemClock.elapsedRealtime()
                    val assembled = ContextAssembler.assemble(
                        retrievedResults = retrieved,
                        queryIntent = queryClassification.intent,
                        queryClassification = queryClassification,
                        maxEvidenceTokens = evidenceBudget
                    )
                    Log.i("CHAT_LATENCY", "rag_context_assembly_ms=${SystemClock.elapsedRealtime() - contextStartedAt}")
                    Log.i("RAG_DEBUG", "RAG_DEBUG_05_CONTEXT_DONE: tokens=${assembled.totalEstimatedTokens} chars=${assembled.contextText.length}")
                    sourcesData = assembled.sourcesJson

                    answer = if (assembled.sourceBlocks.isEmpty() && assembled.contextText.startsWith("[")) {
                        when {
                            assembled.contextText.startsWith("[PAGE_NOT_FOUND:") -> {
                                val p = assembled.contextText.removePrefix("[PAGE_NOT_FOUND:").removeSuffix("]").trim()
                                "I couldn't find page $p in the provided document."
                            }
                            assembled.contextText.startsWith("[PAGE_RANGE_NOT_FOUND:") -> {
                                val r = assembled.contextText.removePrefix("[PAGE_RANGE_NOT_FOUND:").removeSuffix("]").trim()
                                "I couldn't find pages $r in the provided document."
                            }
                            else -> "I searched the provided document(s), but could not find information answering \"$userText\"."
                        }
                    } else {
                        val ragPrompt = buildPrompt(assembled.contextText, userText, answerLength)
                        val generationStartedAt = SystemClock.elapsedRealtime()
                        val rawAnswer = try {
                            val deferred = kotlinx.coroutines.CoroutineScope(Dispatchers.Default).async {
                                llm.generate(ragPrompt, answerLength)
                            }
                            kotlinx.coroutines.withTimeout(120_000L) { deferred.await() }
                        } catch (timeout: kotlinx.coroutines.TimeoutCancellationException) {
                            Log.e("RAG_DEBUG", "LLM inference exceeded 120s threshold on device")
                            (llm as? GGUFLLM)?.cancel()
                            throw IllegalStateException("Model generation timed out after 120 seconds. Please try again with a shorter query.")
                        } catch (t: Throwable) {
                            Log.e("RAG_DEBUG", "LLM inference encountered error: ${t.message}", t)
                            throw t
                        }
                        Log.i("CHAT_LATENCY", "path=document_rag llm_total_ms=${SystemClock.elapsedRealtime() - generationStartedAt}")
                        Log.i("RAG_DEBUG", "RAG_DEBUG_10_REPOSITORY_RETURN: len=${rawAnswer.length}")
                        if (rawAnswer.isBlank()) {
                            throw IllegalStateException("Local neural model produced an empty answer.")
                        }
                        rawAnswer
                    }
                }

                timerJob?.cancelAndJoin()

                // CRITICAL: Set state to Idle IMMEDIATELY after generation completes,
                // BEFORE persistence. This ensures the UI "Thinking..." indicator clears
                // even if persistence takes time or encounters issues.
                _chatState.value = ChatState.Idle

                val persistStartedAt = SystemClock.elapsedRealtime()
                messageDao.insert(
                    MessageEntity(
                        id = 0,
                        conversationId = conversationId,
                        role = "assistant",
                        content = answer,
                        timestamp = System.currentTimeMillis(),
                        sources = sourcesData
                    )
                )
                Log.i("CHAT_LATENCY", "assistant_response_persist_ms=${SystemClock.elapsedRealtime() - persistStartedAt}")

                Log.i(
                    "CHAT_LATENCY",
                    "request_complete path=${if (cleanDocIds.isEmpty()) "no_document" else "document_rag"} total_ms=${SystemClock.elapsedRealtime() - requestStartedAt} answer_chars=${answer.length}"
                )
            } catch (t: Throwable) {
                timerJob?.cancelAndJoin()
                Log.e("RAG_DEBUG", "RAG_DEBUG_12_ERROR_STATE: error=${t.message}", t)
                Log.e(TAG, "Chat generation failed", t)
                val rawMsg = t.message ?: "Unknown error"
                val errorMsg = when {
                    rawMsg.contains("is not installed", ignoreCase = true) || rawMsg.contains("model file not found", ignoreCase = true) || rawMsg.contains("Real GGUF execution required", ignoreCase = true) ->
                        "Local LLM model is not installed. Open Settings -> Model Management and import a GGUF model."
                    rawMsg.contains("cannot be read", ignoreCase = true) ->
                        "Installed model cannot be read. Re-import the model."
                    rawMsg.contains("not a valid", ignoreCase = true) || rawMsg.contains("unsupported GGUF", ignoreCase = true) ->
                        "The selected file is not a valid or supported GGUF model."
                    rawMsg.contains("memory", ignoreCase = true) ->
                        "Unable to load the model because the device does not have enough available memory."
                    rawMsg.contains("initialize the local LLM", ignoreCase = true) || rawMsg.contains("Failed to load local GGUF model", ignoreCase = true) ->
                        "Unable to initialize the local LLM. Check the model and device compatibility."
                    rawMsg.contains("generation failed", ignoreCase = true) ->
                        "Local generation failed. The model remains installed, but the current generation could not be completed."
                    else ->
                        "An error occurred while answering: $rawMsg"
                }
                try {
                    messageDao.insert(
                        MessageEntity(
                            id = 0,
                            conversationId = conversationId,
                            role = "assistant",
                            content = errorMsg,
                            timestamp = System.currentTimeMillis()
                        )
                    )
                } catch (insertError: Throwable) {
                    Log.e(TAG, "Failed to insert error message into conversation", insertError)
                }
                _chatState.value = ChatState.Error(errorMsg)
            } finally {
                timerJob?.cancelAndJoin()
                if (_chatState.value is ChatState.Generating || _chatState.value is ChatState.Retrieving || _chatState.value is ChatState.Thinking) {
                    _chatState.value = ChatState.Idle
                }
            }
        }
    }

    private fun buildNormalPrompt(question: String, answerLength: AnswerLength = AnswerLength.SHORT): String {
        val lengthInstruction = when (answerLength) {
            AnswerLength.SHORT -> """
- Give the minimum sufficient answer supported by the evidence.
- Never invent facts.
- Never use outside knowledge.
- Never repeat a fact.
- Never include irrelevant evidence.
- Do not create empty list items.
- Do not output headings as answers.
- For lists, preserve all requested items.
- Stop when the relevant evidence has been adequately covered.
""".trimIndent()
            AnswerLength.LONG -> """
- Give the most complete answer that is directly relevant to the question and supported by the evidence.
- Never invent facts.
- Never use outside knowledge.
- Never repeat a fact.
- Never include irrelevant evidence.
- Do not create empty list items.
- Do not output headings as answers.
- For step-by-step, troubleshooting, or checklist questions, provide numbered points followed by a brief grounded explanation of what that step means in context.
- Explain each point using only the retrieved evidence and keep the explanation tied to the document wording.
- For list questions, preserve every requested item with its grounded description from the source.
- Format lists with clean numbering (e.g. 1. Item - Explanation based on the source evidence).
- Include all relevant points without skipping or truncating the answer.
- Cover additional distinct relevant evidence when it helps explain the answer.
- Stop when the relevant evidence has been adequately covered.
""".trimIndent()
        }
        return """
[ANSWER_LENGTH: ${answerLength.name}]
You are a helpful offline AI assistant. $lengthInstruction

Question: $question

Answer:
""".trimIndent()
    }

    private fun buildPrompt(context: String, question: String, answerLength: AnswerLength = AnswerLength.SHORT): String {
        val lengthInstruction = when (answerLength) {
            AnswerLength.SHORT -> """
- Give the minimum sufficient answer supported by the evidence.
- Never invent facts.
- Never use outside knowledge.
- Never repeat a fact.
- Never include irrelevant evidence.
- Do not create empty list items.
- Do not output headings as answers.
- For lists, preserve all requested items.
- Stop when the relevant evidence has been adequately covered.
""".trimIndent()
            AnswerLength.LONG -> """
- Give the most complete answer that is directly relevant to the question and supported by the evidence.
- Never invent facts.
- Never use outside knowledge.
- Never repeat a fact.
- Never include irrelevant evidence.
- Do not create empty list items.
- Do not output headings as answers.
- For step-by-step, troubleshooting, or checklist questions, provide numbered points followed by a brief grounded explanation of what that step means in context.
- Explain each point using only the retrieved evidence and keep the explanation tied to the document wording.
- For list questions, preserve every requested item with its grounded description from the source.
- Format lists with clean numbering (e.g. 1. Item - Explanation based on the source evidence).
- Include all relevant points without skipping or truncating the answer.
- Cover additional distinct relevant evidence when it helps explain the answer.
- Stop when the relevant evidence has been adequately covered.
""".trimIndent()
        }
        return """
[ANSWER_LENGTH: ${answerLength.name}]
You are an offline AI assistant. Use the following extracted document context to answer the user's question.

Context:
$context

Question: $question

Answer:
$lengthInstruction
Reference the page numbers when appropriate.
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
