package com.siddu.offlinedocai.domain.repository

import android.content.Context
import android.util.Log
import com.siddu.offlinedocai.ai.embeddings.EmbeddingModel
import com.siddu.offlinedocai.ai.embeddings.GGUFEmbeddingEngine
import com.siddu.offlinedocai.ai.llm.LocalLLM
import com.siddu.offlinedocai.ai.llm.OllamaLLM
import com.siddu.offlinedocai.ai.retrieval.VectorStore
import com.siddu.offlinedocai.data.AppDatabase
import com.siddu.offlinedocai.data.dao.MessageDao
import com.siddu.offlinedocai.data.entities.MessageEntity
import com.siddu.offlinedocai.data.entities.ConversationEntity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import java.util.UUID

/**
 * Repository that implements the RAG chat pipeline.
 *   1. Persist a ConversationEntity when a new chat starts.
 *   2. For each user message, embed the text via [EmbeddingModel].
 *   3. Retrieve relevant chunks using [VectorStore] (top‑K).
 *   4. Build a prompt that includes the retrieved context and the user question.
 *   5. Generate a response via [LocalLLM] (Ollama for dev, or GGUFLLM stub).
 *   6. Persist both user and assistant messages.
 */
class ChatRepository(
    private val context: Context,
    private val db: AppDatabase,
    private val vectorStore: VectorStore,
    private val useOllama: Boolean = true // dev flag – set false for final offline mode
) {
    private val TAG = "ChatRepository"
    private val messageDao: MessageDao = db.messageDao()
    private val conversationDao = db.conversationDao()
    private val embeddingEngine: EmbeddingModel = GGUFEmbeddingEngine(context)
    private val llm: LocalLLM = if (useOllama) OllamaLLM() else GGUFLLM(context)

    private val _chatState = MutableStateFlow<ChatState>(ChatState.Idle)
    val chatState: StateFlow<ChatState> = _chatState

    /** Create a new conversation and return its generated ID */
    fun startConversation(title: String = "Chat ${System.currentTimeMillis()}"): Long {
        var convId = -1L
        CoroutineScope(Dispatchers.IO).launch {
            val conversation = ConversationEntity(
                id = 0,
                title = title,
                createdAt = System.currentTimeMillis()
            )
            convId = conversationDao.insert(conversation)
        }
        return convId
    }

    /** Send a user message, run RAG, and emit the assistant response */
    fun sendMessage(conversationId: Long, userText: String, relevantDocumentIds: List<Long>) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                _chatState.value = ChatState.Thinking
                // Persist user message
                val userMsg = MessageEntity(
                    id = 0,
                    conversationId = conversationId,
                    sender = "user",
                    content = userText,
                    timestamp = System.currentTimeMillis()
                )
                messageDao.insert(userMsg)

                // Embedding of the query
                val queryEmbedding = embeddingEngine.getEmbedding(userText)

                // Retrieve relevant chunks (top‑5)
                val chunks = vectorStore.retrieve(queryEmbedding, relevantDocumentIds, topK = 5)
                val contextText = chunks.joinToString("\n\n") { "[Page ${it.pageNumber}] ${it.text}" }

                // Build prompt
                val prompt = buildPrompt(contextText, userText)

                // Generate answer
                val answer = llm.generate(prompt)

                // Persist assistant message
                val assistantMsg = MessageEntity(
                    id = 0,
                    conversationId = conversationId,
                    sender = "assistant",
                    content = answer,
                    timestamp = System.currentTimeMillis()
                )
                messageDao.insert(assistantMsg)

                _chatState.value = ChatState.Idle
            } catch (e: Exception) {
                Log.e(TAG, "Chat generation failed", e)
                _chatState.value = ChatState.Error(e.message ?: "Unknown error")
            }
        }
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

    sealed class ChatState {
        object Idle : ChatState()
        object Thinking : ChatState()
        data class Error(val message: String) : ChatState()
    }
}
