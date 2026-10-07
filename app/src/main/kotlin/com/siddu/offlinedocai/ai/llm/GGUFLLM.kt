package com.siddu.offlinedocai.ai.llm

import android.app.ActivityManager
import android.content.Context
import android.util.Log
import com.siddu.offlinedocai.ai.retrieval.AssembledContext
import com.siddu.offlinedocai.ai.retrieval.ContextAssembler
import com.siddu.offlinedocai.data.ai.ModelManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File

/** What the UI shows about the language model: always derived from the real native load result. */
sealed class LlmLoadState {
    object NoModel : LlmLoadState()
    data class NotLoaded(val modelName: String) : LlmLoadState()
    data class Loading(val modelName: String) : LlmLoadState()
    data class Ready(val modelName: String, val contextSize: Int, val cpuVariant: String) : LlmLoadState()
    data class Failed(val modelName: String, val reason: String) : LlmLoadState()
}

/**
 * On-device GGUF model (llama.cpp) for grounded answering and plain chat. One instance per process
 * ([getInstance]), so a model is never loaded twice.
 *
 * - The model to use always comes from [ModelManager] (single source of truth for the active model).
 * - Prompts are formatted with the model's own chat template (falling back to a family template
 *   only if the GGUF has none), from structured messages built by [PromptBuilder].
 * - The context window is sized from device memory once per model and never exceeds the model's
 *   training context.
 * - Reasoning models (DeepSeek-R1 style) get a separate thinking budget; if they are still thinking
 *   when it runs out, the reasoning is closed and the model is asked for the answer.
 * - There is no non-LLM fallback: if the model is missing or fails, an exception is thrown and the
 *   UI shows a real error instead of a fabricated answer.
 */
class GGUFLLM private constructor(private val context: Context) : LocalLLM {

    private val TAG = "GGUFLLM"

    companion object {
        const val PREFERRED_CONTEXT_SIZE = 4096
        const val MIN_CONTEXT_SIZE = 2048
        /**
         * Devices with less total RAM than this use the smaller window. A 4096-token KV cache costs
         * ~110-140 MB for the 0.5-1.5B catalog models, affordable on any "4 GB" phone (reports ~3.6 GB).
         */
        private const val LARGE_CONTEXT_MIN_RAM_BYTES = 3_400L * 1024 * 1024
        private const val SAFETY_MARGIN_TOKENS = 64
        private const val THINKING_BUDGET_TOKENS = 768
        private const val CHAT_TEMPERATURE = 0.6f
        private const val TOP_P = 0.9f
        private const val REPEAT_PENALTY = 1.1f

        const val MODEL_NOT_INSTALLED = "No language model is installed. Open Settings and download or import a model."

        @Volatile private var instance: GGUFLLM? = null

        fun getInstance(context: Context): GGUFLLM =
            instance ?: synchronized(this) {
                instance ?: GGUFLLM(context.applicationContext).also { instance = it }
            }
    }

    val llamaRunner: LlamaRunner = LlamaRunner(context)
    private val modelManager: ModelManager get() = ModelManager.getInstance(context)

    /** Serialises load / prompt preparation / generation / release, so a model switch can never interleave with a request. */
    private val modelMutex = Mutex()
    private val contextSizeByModel = HashMap<String, Int>()

    @Volatile private var templateFamily: TemplateFamily = TemplateFamily.CHATML
    @Volatile private var hasNativeTemplate = false
    @Volatile private var isReasoningModel = false
    @Volatile private var promptOpensThinking = false

    private val _loadState = MutableStateFlow<LlmLoadState>(LlmLoadState.NoModel)
    val loadState: StateFlow<LlmLoadState> = _loadState.asStateFlow()

    private fun activeModelFile(): File? = runCatching { modelManager.getActiveModelFile() }.getOrNull()

    /** Loads the active model so the first question does not pay the load time. Returns true when ready. */
    suspend fun warmUp(): Boolean = withContext(Dispatchers.Default) {
        modelMutex.withLock { runCatching { ensureLoaded() }.isSuccess }
    }

    /** Frees the native model (e.g. before its file is deleted or replaced). Waits for a running generation. */
    suspend fun release() = withContext(Dispatchers.Default) {
        modelMutex.withLock {
            llamaRunner.freeModel()
            _loadState.value = activeModelFile()?.let { LlmLoadState.NotLoaded(displayName(it)) } ?: LlmLoadState.NoModel
        }
    }

    /** Path of the model file currently held in memory, if any. */
    fun loadedModelPath(): String? = llamaRunner.getLoadedModelPath()

    /** Refreshes the idle state after the active model changed (no loading). */
    fun onActiveModelChanged() {
        val file = activeModelFile()
        val loaded = llamaRunner.getLoadedModelPath()
        if (file == null) {
            if (loaded == null) _loadState.value = LlmLoadState.NoModel
        } else if (loaded != file.absolutePath && _loadState.value !is LlmLoadState.Loading) {
            _loadState.value = LlmLoadState.NotLoaded(displayName(file))
        }
    }

    private fun displayName(file: File): String =
        modelManager.models.value.firstOrNull { it.file.absolutePath == file.absolutePath }?.displayName ?: file.nameWithoutExtension

    /** Must be called with [modelMutex] held. */
    private fun ensureLoaded() {
        val file = activeModelFile()
        if (file == null) {
            _loadState.value = LlmLoadState.NoModel
            throw IllegalStateException(MODEL_NOT_INSTALLED)
        }
        val name = displayName(file)
        if (!llamaRunner.isNativeLoaded) {
            val reason = llamaRunner.getLastErrorMessage() ?: "Unable to initialize the local LLM."
            _loadState.value = LlmLoadState.Failed(name, reason)
            throw IllegalStateException(reason)
        }
        val ctxSize = contextSizeByModel.getOrPut(file.absolutePath) { chooseContextSize() }
        if (llamaRunner.isModelLoaded() && llamaRunner.getLoadedModelPath() == file.absolutePath) {
            if (_loadState.value !is LlmLoadState.Ready) {
                _loadState.value = LlmLoadState.Ready(name, llamaRunner.contextSize, LlamaNative.cpuVariant)
            }
            return
        }

        _loadState.value = LlmLoadState.Loading(name)
        modelManager.setGenerating(true) // the file must not be deleted while it is being mapped
        try {
            if (!llamaRunner.loadModel(file.absolutePath, ctxSize, CpuInfo.performanceCores)) {
                val reason = llamaRunner.getLastErrorMessage() ?: "Unable to initialize the local LLM. Check the model and device compatibility."
                _loadState.value = LlmLoadState.Failed(name, reason)
                throw IllegalStateException(reason)
            }
        } finally {
            modelManager.setGenerating(false)
        }
        val template = llamaRunner.metadata("tokenizer.chat_template")
        val architecture = llamaRunner.metadata("general.architecture")
        val modelName = llamaRunner.metadata("general.name").orEmpty()
        templateFamily = PromptBuilder.familyFor(template, architecture)
        hasNativeTemplate = llamaRunner.applyChatTemplate(listOf(ChatMessage("user", "ping"))) != null
        isReasoningModel = PromptBuilder.isReasoningModel(template, modelName + " " + file.name)
        promptOpensThinking = isReasoningModel && render(listOf(ChatMessage("user", "ping"))).trimEnd().endsWith("<think>")
        _loadState.value = LlmLoadState.Ready(name, llamaRunner.contextSize, LlamaNative.cpuVariant)
        Log.i(TAG, "Model ready: ${file.name} arch=$architecture n_ctx=${llamaRunner.contextSize} nativeTemplate=$hasNativeTemplate family=$templateFamily reasoning=$isReasoningModel")
    }

    private fun chooseContextSize(): Int {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager ?: return MIN_CONTEXT_SIZE
        val info = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        return if (info.totalMem >= LARGE_CONTEXT_MIN_RAM_BYTES) PREFERRED_CONTEXT_SIZE else MIN_CONTEXT_SIZE
    }

    private fun contextWindow(): Int = llamaRunner.contextSize.takeIf { it > 0 } ?: MIN_CONTEXT_SIZE

    /** Tokens of the visible answer. */
    fun maxNewTokens(length: AnswerLength): Int {
        val wanted = when (length) {
            AnswerLength.SHORT -> 192
            AnswerLength.DEFAULT -> 448
            AnswerLength.LONG -> 900
        }
        return minOf(wanted, contextWindow() / 4)
    }

    /** Thinking budget for reasoning models (0 for normal instruct models). */
    private fun thinkingTokens(): Int = if (isReasoningModel) minOf(THINKING_BUDGET_TOKENS, contextWindow() / 4) else 0

    override fun generationTimeoutMs(answerLength: AnswerLength): Long {
        // Generous for slow phones: ~3 tokens/s worst case plus prompt processing.
        val tokens = maxNewTokens(answerLength) + thinkingTokens()
        return 90_000L + tokens * 330L
    }

    private fun render(messages: List<ChatMessage>): String {
        val prepared = if (templateFamily == TemplateFamily.DEEPSEEK) PromptBuilder.foldSystemIntoUser(messages) else messages
        return (if (hasNativeTemplate) llamaRunner.applyChatTemplate(prepared) else null)
            ?: PromptBuilder.formatFallback(prepared, templateFamily)
    }

    override fun countTokens(text: String): Int =
        llamaRunner.countTokens(text, isPrompt = false) ?: ContextAssembler.estimateTokens(text)

    private fun countPromptTokens(prompt: String): Int =
        llamaRunner.countTokens(prompt, isPrompt = true) ?: ContextAssembler.estimateTokens(prompt)

    override suspend fun availableEvidenceTokens(question: String, answerLength: AnswerLength): Int = withContext(Dispatchers.Default) {
        modelMutex.withLock {
            ensureLoaded()
            val skeleton = render(PromptBuilder.groundedMessages(emptyList(), question, answerLength, isBudgetLimited = true))
            val overhead = countPromptTokens(skeleton)
            (contextWindow() - overhead - maxNewTokens(answerLength) - thinkingTokens() - SAFETY_MARGIN_TOKENS).coerceAtLeast(0)
        }
    }

    override suspend fun generateGrounded(
        context: AssembledContext,
        question: String,
        answerLength: AnswerLength,
        request: GenerationRequest
    ): String = withContext(Dispatchers.Default) {
        modelMutex.withLock {
            ensureLoaded()
            val maxNew = maxNewTokens(answerLength) + thinkingTokens()
            val previous = request.previousTurn?.question
            var blocks = context.sourceBlocks
            var prompt = render(PromptBuilder.groundedMessages(blocks, question, answerLength, context.isBudgetLimited, previous))
            // Final guard: the assembler budgets conservatively, but never send a prompt that cannot fit.
            while (blocks.size > 1 && countPromptTokens(prompt) + maxNew > contextWindow()) {
                blocks = blocks.dropLast(1)
                prompt = render(PromptBuilder.groundedMessages(blocks, question, answerLength, isBudgetLimited = true, previousQuestion = previous))
            }
            // Grounded answers are decoded greedily: deterministic and least likely to drift from the sources.
            val answer = runInference(prompt, maxNewTokens(answerLength), temperature = 0f, request = request)
            PromptBuilder.sanitizeCitations(answer, blocks)
        }
    }

    override suspend fun generateChat(message: String, answerLength: AnswerLength, request: GenerationRequest): String =
        withContext(Dispatchers.Default) {
            modelMutex.withLock {
                ensureLoaded()
                runInference(render(PromptBuilder.chatMessages(message, answerLength, request.previousTurn)), maxNewTokens(answerLength), CHAT_TEMPERATURE, request)
            }
        }

    /** Must be called with [modelMutex] held. */
    private fun runInference(prompt: String, answerTokens: Int, temperature: Float, request: GenerationRequest): String {
        val token = request.token
        modelManager.setGenerating(true)
        try {
            request.onStarted?.invoke()
            val partial: ((String) -> Unit)? = request.onPartial?.let { cb ->
                { text: String -> cb(PromptBuilder.visibleAnswer(text, promptOpensThinking)) }
            }
            val budget = answerTokens + thinkingTokens()
            var raw = llamaRunner.inference(prompt, budget, temperature, TOP_P, REPEAT_PENALTY, token, onPartial = partial)
            if (token.isCancelled) throw GenerationCancelledException(PromptBuilder.cleanOutput(raw, promptOpensThinking))

            // A reasoning model that ran out of budget while thinking: close the reasoning and ask for the answer.
            if (isReasoningModel && PromptBuilder.isStillThinking(raw, promptOpensThinking)) {
                val forced = prompt + raw.trimEnd() + "\n</think>\n\n"
                if (countPromptTokens(forced) + answerTokens <= contextWindow()) {
                    val answer = llamaRunner.inference(forced, answerTokens, temperature, TOP_P, REPEAT_PENALTY, token, onPartial = request.onPartial)
                    raw = raw.trimEnd() + "\n</think>\n\n" + answer
                }
            }
            val cleaned = PromptBuilder.cleanOutput(raw, promptOpensThinking)
            if (token.isCancelled) throw GenerationCancelledException(cleaned)
            if (cleaned.isBlank()) throw IllegalStateException("Local neural model produced an empty answer.")
            return cleaned
        } finally {
            modelManager.setGenerating(false)
        }
    }

    override fun cancel() {
        llamaRunner.cancelInference()
    }
}
