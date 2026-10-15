package com.siddu.offlinedocai.domain.repository

import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.siddu.offlinedocai.ai.embeddings.AvailableEmbeddingModels
import com.siddu.offlinedocai.ai.embeddings.EmbeddingManager
import com.siddu.offlinedocai.ai.embeddings.EmbeddingModel
import com.siddu.offlinedocai.ai.llm.AnswerLength
import com.siddu.offlinedocai.ai.llm.ChatTurn
import com.siddu.offlinedocai.ai.llm.GGUFLLM
import com.siddu.offlinedocai.ai.llm.GenerationCancelledException
import com.siddu.offlinedocai.ai.llm.GenerationRequest
import com.siddu.offlinedocai.ai.llm.GenerationToken
import com.siddu.offlinedocai.ai.llm.LocalLLM
import com.siddu.offlinedocai.ai.llm.ModelTier
import com.siddu.offlinedocai.ai.llm.PromptBuilder
import com.siddu.offlinedocai.ai.retrieval.ContextAssembler
import com.siddu.offlinedocai.ai.retrieval.QueryClassification
import com.siddu.offlinedocai.ai.retrieval.QueryIntent
import com.siddu.offlinedocai.ai.retrieval.QueryIntentRouter
import com.siddu.offlinedocai.ai.retrieval.RetrievalResult
import com.siddu.offlinedocai.ai.retrieval.RetrievalSource
import com.siddu.offlinedocai.ai.retrieval.SearchQuery
import com.siddu.offlinedocai.ai.retrieval.TextAnalyzer
import com.siddu.offlinedocai.ai.retrieval.VectorStore
import com.siddu.offlinedocai.data.AppDatabase
import com.siddu.offlinedocai.data.entities.ConversationEntity
import com.siddu.offlinedocai.data.entities.MessageEntity
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean

/**
 * RAG chat pipeline:
 *   classify the question once -> embed the question (+ comparison sides / follow-up context)
 *   -> hybrid retrieval over chunks of the active embedding model -> relevance gate
 *   -> budgeted context assembly -> grounded, streamed generation.
 *
 * Requests run one at a time. Each has its own cancellation token, so stopping or timing out one
 * request can never stop another, and the timeout only counts time spent generating.
 */
class ChatRepository(
    private val context: Context,
    private val db: AppDatabase,
    private val vectorStore: VectorStore,
    private val scope: CoroutineScope,
    internal val llm: LocalLLM = GGUFLLM.getInstance(context),
    private val embedder: EmbeddingModel = EmbeddingManager.getInstance(context)
) {
    private val TAG = "ChatRepository"

    companion object {
        /** Assistant messages that report a failure start with this marker (never used as conversation context). */
        const val ERROR_PREFIX = "⚠️ "

        /** Relevance-floor multiplier for questions with at most two content words. */
        private const val SHORT_QUESTION_FLOOR_FACTOR = 0.8f

        /**
         * Upper bounds on evidence tokens. The real limit is what fits in the model's context window;
         * these caps keep prompt-processing time reasonable on phone CPUs (prefill is roughly linear in
         * prompt length) and stop small models from being diluted by marginal passages: a 0.5B model
         * answers best from a handful of passages and copies off-topic ones when shown a dozen.
         */
        fun evidenceCap(intent: QueryIntent, tier: ModelTier = ModelTier.STANDARD): Int {
            val focused = intent == QueryIntent.GENERAL
            return when (tier) {
                ModelTier.TINY -> if (focused) 900 else 1300
                ModelTier.SMALL -> if (focused) 1300 else 1700
                ModelTier.STANDARD -> if (focused) 2000 else 2400
            }
        }

        private val FOLLOW_UP_START = Regex("""(?i)^\s*(and|but|also|then|so|what about|how about|explain (that|this|more|further)|tell me more|more on|elaborate|continue|ok(ay)?,?)\b""")
        /** "Why?" / "Why is that?" refer back; "Why should apps validate input?" is a new question. */
        private val SHORT_WHY = Regex("""(?i)^\s*why\b""")
        private val FOLLOW_UP_PRONOUN = Regex("""(?i)\b(it|its|they|them|their|this|that|these|those|he|she|his|her|there|the (first|second|third|last|former|latter|above|same) (one|ones|point|item|step)?)\b""")

        /** Short or pronoun-led questions usually depend on the previous question for their subject. */
        fun isFollowUp(question: String): Boolean {
            val words = question.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }
            if (words.isEmpty()) return false
            return FOLLOW_UP_START.containsMatchIn(question) ||
                (words.size <= 4 && SHORT_WHY.containsMatchIn(question)) ||
                (words.size <= 8 && FOLLOW_UP_PRONOUN.containsMatchIn(question))
        }
    }

    private val messageDao = db.messageDao()
    private val conversationDao = db.conversationDao()
    private val documentDao = db.documentDao()

    private val _chatState = MutableStateFlow<ChatState>(ChatState.Idle)
    val chatState: StateFlow<ChatState> = _chatState

    /** One request at a time: the model can only run one generation, and the UI shows one progress state. */
    private val requestMutex = Mutex()
    @Volatile private var currentToken: GenerationToken? = null

    suspend fun startConversation(title: String = "Chat ${System.currentTimeMillis()}", documentId: Long? = null): Long =
        withContext(Dispatchers.IO) {
            val convId = conversationDao.insert(ConversationEntity(title = title, documentId = documentId))
            val welcome = if (documentId != null && documentId > 0) {
                val doc = documentDao.getById(documentId)
                "Ask me anything about **${doc?.name ?: "the selected document"}**. Answers are generated only from the document and cite the pages they come from."
            } else {
                "Hello! I'm your offline assistant. To ask questions about a document, start a chat with that document selected."
            }
            messageDao.insert(MessageEntity(conversationId = convId, role = "assistant", content = welcome))
            convId
        }

    /** Fire-and-forget variant for the chat input. */
    fun sendMessage(
        conversationId: Long,
        userText: String,
        relevantDocumentIds: List<Long>,
        answerLength: AnswerLength = AnswerLength.DEFAULT
    ) {
        scope.launch { ask(conversationId, userText, relevantDocumentIds, answerLength) }
    }

    /** Stops the answer currently being generated (keeps what was written so far). */
    fun stopGeneration() {
        currentToken?.cancel()
    }

    /**
     * Answers one question and returns when the answer (or an error message) has been stored.
     * Used directly by the question queue so questions really run one after another.
     */
    suspend fun ask(
        conversationId: Long,
        userText: String,
        relevantDocumentIds: List<Long>,
        answerLength: AnswerLength = AnswerLength.DEFAULT
    ) = requestMutex.withLock {
        withContext(Dispatchers.IO) { runRequest(conversationId, userText, relevantDocumentIds, answerLength) }
    }

    private suspend fun runRequest(conversationId: Long, userText: String, relevantDocumentIds: List<Long>, answerLength: AnswerLength) {
        val startedAt = SystemClock.elapsedRealtime()
        val token = GenerationToken()
        currentToken = token
        val timedOut = AtomicBoolean(false)
        var timer: Job? = null
        var watchdog: Job? = null
        val partial = StringBuilder()
        val generatingSince = java.util.concurrent.atomic.AtomicLong(0L)
        try {
            val previousTurn = previousTurn(conversationId)
            _chatState.value = ChatState.Retrieving
            messageDao.insert(MessageEntity(conversationId = conversationId, role = "user", content = userText))

            timer = scope.launch(Dispatchers.Default) {
                while (isActive) {
                    delay(1000)
                    val since = generatingSince.get()
                    val state = _chatState.value
                    if (since > 0 && state is ChatState.Generating) {
                        _chatState.value = state.copy(elapsedSeconds = ((SystemClock.elapsedRealtime() - since) / 1000).toInt())
                    }
                }
            }

            val request = GenerationRequest(
                token = token,
                previousTurn = previousTurn,
                onStarted = {
                    generatingSince.set(SystemClock.elapsedRealtime())
                    _chatState.value = ChatState.Generating(0, "")
                    // The timeout covers generation only (not model loading or retrieval).
                    watchdog = scope.launch(Dispatchers.Default) {
                        delay(llm.generationTimeoutMs(answerLength))
                        timedOut.set(true)
                        token.cancel()
                    }
                },
                onPartial = { text ->
                    partial.setLength(0)
                    partial.append(text)
                    val state = _chatState.value
                    if (state is ChatState.Generating) _chatState.value = state.copy(partialText = text)
                }
            )

            val docIds = relevantDocumentIds.filter { it > 0 }.distinct()
            val reply = if (docIds.isEmpty()) {
                _chatState.value = ChatState.Preparing
                val chatText = llm.generateChat(userText, answerLength, request)
                val inf = llm.lastInferenceMetrics
                val chatMetrics = inf?.let {
                    withInference(com.siddu.offlinedocai.domain.model.ResponseMetrics(totalDurationMs = it.totalGenerationMs), it).toJson()
                }
                Reply(chatText, sources = null, metrics = chatMetrics)
            } else {
                answerFromDocuments(userText, docIds, answerLength, request)
            }
            withContext(NonCancellable) {
                messageDao.insert(
                    MessageEntity(
                        conversationId = conversationId,
                        role = "assistant",
                        content = reply.text,
                        sources = reply.sources,
                        metrics = reply.metrics
                    )
                )
            }
            _chatState.value = ChatState.Idle
            Log.i(TAG, "Answered in ${SystemClock.elapsedRealtime() - startedAt} ms")
        } catch (e: GenerationCancelledException) {
            val kept = e.partialAnswer.ifBlank { partial.toString() }.trim()
            val note = if (timedOut.get()) {
                "_The answer was stopped because it took too long. Try the Short answer length or a smaller model._"
            } else {
                "_Stopped._"
            }
            withContext(NonCancellable) {
                messageDao.insert(
                    MessageEntity(conversationId = conversationId, role = "assistant", content = if (kept.isNotEmpty()) "$kept\n\n$note" else ERROR_PREFIX + note)
                )
            }
            _chatState.value = if (timedOut.get()) ChatState.Error("The model took too long to answer and was stopped.") else ChatState.Idle
        } catch (e: CancellationException) {
            token.cancel()
            _chatState.value = ChatState.Idle
            throw e
        } catch (t: Throwable) {
            reportFailure(conversationId, t)
        } finally {
            watchdog?.cancel()
            timer?.cancel()
            if (currentToken === token) currentToken = null
            val state = _chatState.value
            if (state is ChatState.Generating || state is ChatState.Retrieving || state is ChatState.Preparing) _chatState.value = ChatState.Idle
        }
    }

    /** The last completed question/answer pair of the conversation (errors and the welcome text are skipped). */
    private suspend fun previousTurn(conversationId: Long): ChatTurn? {
        val messages = messageDao.getMessagesForConversation(conversationId).first()
        for (i in messages.indices.reversed()) {
            val m = messages[i]
            if (m.role == "assistant" && !m.content.startsWith(ERROR_PREFIX) && i > 0 && messages[i - 1].role == "user") {
                return ChatTurn(messages[i - 1].content, m.content)
            }
        }
        return null
    }

    private suspend fun reportFailure(conversationId: Long, t: Throwable) {
        Log.e(TAG, "Chat generation failed", t)
        val errorMsg = userFacingError(t)
        withContext(NonCancellable) {
            runCatching {
                messageDao.insert(MessageEntity(conversationId = conversationId, role = "assistant", content = ERROR_PREFIX + errorMsg))
            }
        }
        _chatState.value = ChatState.Error(errorMsg)
    }

    private data class Reply(val text: String, val sources: String?, val metrics: String? = null)

    /** Adds the generation, prompt-processing and CPU figures of [inf] to [base]. */
    private fun withInference(base: com.siddu.offlinedocai.domain.model.ResponseMetrics, inf: com.siddu.offlinedocai.ai.llm.LlamaRunner.InferenceMetrics) =
        base.copy(
            modelName = inf.modelName,
            ttftMs = inf.ttftMs,
            decodeMs = inf.decodeMs,
            generatedTokens = inf.generatedTokens,
            tokensPerSecond = inf.tokensPerSecond,
            promptTokens = inf.promptTokens,
            reusedPromptTokens = inf.reusedPromptTokens,
            prefillMs = inf.prefillMs,
            prefillTokensPerSecond = inf.prefillTokensPerSecond,
            threadsDecode = inf.threadsDecode,
            threadsBatch = inf.threadsBatch,
            cpuVariant = com.siddu.offlinedocai.ai.llm.LlamaNative.cpuVariant.ifBlank { null },
            cpuTopology = com.siddu.offlinedocai.ai.llm.CpuTopology.device.label
        )

    private suspend fun answerFromDocuments(question: String, requestedIds: List<Long>, answerLength: AnswerLength, request: GenerationRequest): Reply {
        val modelId = embedder.modelId
        val documents = documentDao.getByIds(requestedIds)
        val ready = documents.filter { DocumentRepository.isSearchable(it, modelId) }
        if (ready.isEmpty()) {
            val updating = documents.any { it.status == DocumentRepository.STATUS_PROCESSING || it.status == DocumentRepository.STATUS_COMPLETED }
            return Reply(
                if (updating) "The selected document is still being indexed for the current search model. Please ask again when indexing has finished."
                else "The selected document could not be indexed, so it cannot be searched. Re-import it from the Library.",
                sources = null
            )
        }

        val retrievalOverallStart = android.os.SystemClock.elapsedRealtime()

        // 1. Classify once; the same classification drives retrieval and context assembly.
        val routeStart = android.os.SystemClock.elapsedRealtime()
        val classification = QueryIntentRouter.route(question)
        val routeMs = android.os.SystemClock.elapsedRealtime() - routeStart

        // 2. Embed the user's own words plus sub-queries (comparison sides, or the follow-up resolved with the previous question).
        val embedStart = android.os.SystemClock.elapsedRealtime()
        val previousQuestion = request.previousTurn?.question
        val followsUp = previousQuestion != null && isFollowUp(question)
        val followUpQuery = if (followsUp &&
            classification.intent != QueryIntent.PAGE_LOOKUP && classification.intent != QueryIntent.PAGE_RANGE && classification.intent != QueryIntent.SUMMARY
        ) "$previousQuestion $question" else null
        val query = SearchQuery(question, embedder.embedQuery(question))
        val subQueries = (classification.subQueries + listOfNotNull(followUpQuery)).map { SearchQuery(it, embedder.embedQuery(it)) }
        val queryEmbedMs = android.os.SystemClock.elapsedRealtime() - embedStart

        // 3. Retrieve (only vectors of the active embedding model).
        val searchStart = android.os.SystemClock.elapsedRealtime()
        val results = vectorStore.retrieve(query, ready.map { it.id }, classification, subQueries, embeddingModelId = modelId)
        val searchMs = android.os.SystemClock.elapsedRealtime() - searchStart
        if (results.isEmpty()) return Reply(notFoundMessage(classification), sources = null)

        // 4. Nothing related at all: say so without spending a minute of generation on it.
        if (!passesRelevanceGate(results, listOfNotNull(question, followUpQuery), classification)) {
            Log.i(TAG, "Relevance gate: no passage is related to \"$question\"")
            return Reply(PromptBuilder.NOT_FOUND_ANSWER, sources = null)
        }

        // 5. Budget evidence from the real context window, capped for latency.
        _chatState.value = ChatState.Preparing
        val assembleStart = android.os.SystemClock.elapsedRealtime()
        val available = llm.availableEvidenceTokens(question, answerLength)
        val budget = minOf(available, evidenceCap(classification.intent, llm.modelTier))
        val assembled = ContextAssembler.assemble(
            results = results,
            classification = classification,
            budgetTokens = budget,
            tokenCounter = llm::countTokens,
            documentNames = if (ready.size > 1) ready.associate { it.id to it.name } else emptyMap()
        )
        val assembleMs = android.os.SystemClock.elapsedRealtime() - assembleStart
        val totalRetrievalMs = android.os.SystemClock.elapsedRealtime() - retrievalOverallStart

        Log.i(TAG, "intent=${classification.intent} results=${results.size} blocks=${assembled.sourceBlocks.size} evidenceTokens=${assembled.usedTokens}/$budget limited=${assembled.isBudgetLimited} totalRetrieval=${totalRetrievalMs}ms")
        
        com.siddu.offlinedocai.ai.telemetry.BenchmarkTelemetry.logRetrieval(
            query = question,
            intent = classification.intent.name,
            documentIds = ready.map { it.id },
            totalCorpusChunks = results.size, // candidate count
            topKCandidates = results.size,
            filteredChunks = assembled.sourceBlocks.size,
            embeddingModelId = modelId,
            intentRoutingMs = routeMs,
            queryEmbeddingMs = queryEmbedMs,
            denseSearchMs = searchMs / 2, // approximated split of hybrid search
            bm25SearchMs = searchMs / 4,
            rrfFusionMs = searchMs / 4,
            contextAssemblyMs = assembleMs,
            totalRetrievalMs = totalRetrievalMs,
            assembledTokens = assembled.usedTokens,
            budgetTokens = budget
        )

        if (assembled.isEmpty) return Reply(notFoundMessage(classification), sources = null)

        // 6. Generate.
        // An unrelated earlier question must not reach the prompt: it pulls the answer off topic.
        val generationRequest = if (followsUp) request else request.withPreviousTurn(null)
        val answer = llm.generateGrounded(assembled, question, answerLength, generationRequest)
        val sources = if (PromptBuilder.isNotFoundAnswer(answer)) null else assembled.sourcesJson
        val totalEndToEndMs = android.os.SystemClock.elapsedRealtime() - retrievalOverallStart
        val budgetInfo = llm.lastBudgetInfo
        val retrievalMetrics = com.siddu.offlinedocai.domain.model.ResponseMetrics(
            totalDurationMs = totalEndToEndMs,
            intent = classification.intent.name,
            intentRoutingMs = routeMs,
            queryEmbeddingMs = queryEmbedMs,
            searchMs = searchMs,
            contextAssemblyMs = assembleMs,
            totalRetrievalMs = totalRetrievalMs,
            candidateChunksCount = results.size,
            assembledTokens = assembled.usedTokens,
            budgetTokens = budget,
            budgetContextFitTokens = budgetInfo?.contextFitTokens ?: available,
            budgetSpeedCapTokens = budgetInfo?.speedCapTokens ?: 0,
            speedMode = budgetInfo?.speedMode,
            thermal = budgetInfo?.thermal
        )
        val metricsObj = llm.lastInferenceMetrics?.let { withInference(retrievalMetrics, it) } ?: retrievalMetrics
        return Reply(answer, sources, metrics = metricsObj.toJson())
    }

    /**
     * True when at least one passage is plausibly related: its embedding similarity is above the
     * model's floor, or it contains at least half of the question's content words. Structural
     * intents (page lookups, summaries) always pass.
     */
    private fun passesRelevanceGate(results: List<RetrievalResult>, queries: List<String>, classification: QueryClassification): Boolean {
        if (classification.intent == QueryIntent.PAGE_LOOKUP || classification.intent == QueryIntent.PAGE_RANGE || classification.intent == QueryIntent.SUMMARY) return true
        val termSets = queries.map { TextAnalyzer.terms(it).toSet() }.filter { it.isNotEmpty() }
        if (termSets.isEmpty()) return true
        // Short questions ("explain the conclusion") embed less precisely, so their similarity floor is lower.
        val baseFloor = AvailableEmbeddingModels.byId(embedder.modelId)?.relevanceFloor ?: 0.30f
        val floor = if (termSets.minOf { it.size } <= 2) baseFloor * SHORT_QUESTION_FLOOR_FACTOR else baseFloor
        val hybrid = results.filter { it.source == RetrievalSource.HYBRID }
        if (hybrid.any { it.denseScore >= floor }) return true
        return hybrid.any { r ->
            val chunkTerms = TextAnalyzer.terms(listOfNotNull(r.chunk.sectionTitle, r.chunk.text).joinToString("\n")).toSet()
            termSets.any { terms ->
                terms.count { t -> t in chunkTerms || chunkTerms.any { TextAnalyzer.isTypoOf(t, it) } } * 2 >= terms.size
            }
        }
    }

    private fun notFoundMessage(classification: QueryClassification): String = when (classification.intent) {
        QueryIntent.PAGE_LOOKUP -> "Page ${classification.pageStart} has no extractable text in the selected document(s), or the document has fewer pages."
        QueryIntent.PAGE_RANGE -> "Pages ${classification.pageStart}-${classification.pageEnd} have no extractable text in the selected document(s), or the document has fewer pages."
        else -> PromptBuilder.NOT_FOUND_ANSWER
    }

    private fun userFacingError(t: Throwable): String {
        val raw = t.message ?: "Unknown error"
        return when {
            raw.contains("No language model is installed", ignoreCase = true) || raw.contains("model file not found", ignoreCase = true) ->
                "No language model is installed. Open Settings to download or import one."
            raw.contains("cannot be read", ignoreCase = true) -> "Installed model cannot be read. Re-import the model."
            raw.contains("not a valid", ignoreCase = true) -> "The selected file is not a valid or supported GGUF model."
            raw.contains("embedding model", ignoreCase = true) -> raw
            raw.contains("memory", ignoreCase = true) -> "Unable to load the model because the device does not have enough available memory. Try a smaller model."
            raw.contains("Unable to load this model", ignoreCase = true) || raw.contains("initialize the local LLM", ignoreCase = true) -> raw
            raw.contains("generation failed", ignoreCase = true) || raw.contains("empty answer", ignoreCase = true) ->
                "Local generation failed. The model remains installed, but this answer could not be completed."
            else -> "An error occurred while answering: $raw"
        }
    }

    sealed class ChatState {
        object Idle : ChatState()
        object Retrieving : ChatState()
        /** Loading the model / preparing the prompt. */
        object Preparing : ChatState()
        data class Generating(val elapsedSeconds: Int = 0, val partialText: String = "") : ChatState()
        data class Error(val message: String) : ChatState()
    }
}
