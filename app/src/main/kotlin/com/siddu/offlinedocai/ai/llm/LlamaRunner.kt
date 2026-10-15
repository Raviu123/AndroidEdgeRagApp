package com.siddu.offlinedocai.ai.llm

import android.content.Context
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

enum class LlamaState {
    UNINITIALIZED,
    LOADING,
    READY,
    GENERATING,
    ERROR,
    UNLOADED
}

/** A chat turn handed to the model's chat template. */
data class ChatMessage(val role: String, val content: String)

/**
 * Cancellation handle for one generation request. Cancelling a token only ever stops *its own*
 * request: a token cancelled before its inference starts makes that inference return immediately,
 * and a token never stops another request that happens to be running.
 */
class GenerationToken {
    private val cancelled = AtomicBoolean(false)
    val isCancelled: Boolean get() = cancelled.get()
    internal var onCancel: (() -> Unit)? = null

    fun cancel() {
        if (cancelled.compareAndSet(false, true)) onCancel?.invoke()
    }
}

/**
 * Thin, thread-safe wrapper over the llama.cpp JNI bridge ([LlamaNative]). All strings cross JNI as UTF-8 bytes.
 */
class LlamaRunner(context: Context) {

    private val TAG = "LlamaRunner"

    val isNativeLoaded: Boolean = LlamaNative.ensureInitialized(context)

    @Volatile private var handle: Long = 0L
    @Volatile private var loadedModelPath: String? = null
    private var requestedContextSize: Int = 0
    /** Guards [handle] against being freed while a cancel is being delivered to it. */
    private val handleLock = Any()
    @Volatile private var activeToken: GenerationToken? = null
    /** First generation after a load exercises the CPU kernels; keep the crash sentinel armed for it. */
    @Volatile private var firstGenerationPending = false

    private val _runnerState = MutableStateFlow(if (isNativeLoaded) LlamaState.UNINITIALIZED else LlamaState.ERROR)
    val runnerState: StateFlow<LlamaState> = _runnerState.asStateFlow()

    data class InferenceMetrics(
        val modelName: String,
        /** Call start -> first token sampled (prompt processing included), measured natively. */
        val ttftMs: Long,
        val decodeMs: Long,
        val totalGenerationMs: Long,
        val generatedTokens: Int,
        val tokensPerSecond: Double,
        val promptTokens: Int = 0,
        /** Prompt prefix served from the previous request's KV cache (not re-processed). */
        val reusedPromptTokens: Int = 0,
        val prefillMs: Long = 0L,
        val prefillTokensPerSecond: Double = 0.0,
        val threadsDecode: Int = 0,
        val threadsBatch: Int = 0
    )

    @Volatile
    var lastInferenceMetrics: InferenceMetrics? = null
        private set

    /** Receives (newPromptTokens, prefillMs, generatedTokens, decodeMs) after each completed inference. */
    @Volatile var onSpeedSample: ((Long, Long, Long, Long) -> Unit)? = null

    val currentState: LlamaState get() = _runnerState.value
    val isGenerating: Boolean get() = _runnerState.value == LlamaState.GENERATING

    @Volatile private var lastErrorMessage: String? =
        if (isNativeLoaded) null else "Unable to initialize the local LLM: ${LlamaNative.loadError ?: "native library not loaded."}"
    fun getLastErrorMessage(): String? = lastErrorMessage

    /** Context window actually allocated for the loaded model (0 when nothing is loaded). */
    @Volatile var contextSize: Int = 0
        private set

    @Synchronized
    fun isModelLoaded(): Boolean = isNativeLoaded && handle != 0L

    fun getLoadedModelPath(): String? = loadedModelPath

    /** Loads (or reuses) a GGUF model, then applies [threads]. Blocking; call from a background dispatcher. */
    @Synchronized
    fun loadModel(modelPath: String, contextSize: Int = 4096, threads: ThreadPlan = ThreadPlan(4, 4)): Boolean {
        if (!loadModelInternal(modelPath, contextSize, threads.decode)) return false
        setThreads(threads)
        return true
    }

    /** Thread counts for generation / prompt processing; applies to the next run without reloading. */
    @Synchronized
    fun setThreads(threads: ThreadPlan) {
        if (handle == 0L) return
        val cores = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
        LlamaNative.nativeSetThreads(handle, threads.decode.coerceIn(1, cores), threads.batch.coerceIn(1, cores))
    }

    /**
     * Synthetic speed test (see [LlamaNative.nativeBenchmark]); null when it failed or was interrupted.
     * Runs with the crash sentinel armed: it is often the first time the CPU kernels execute.
     */
    @Synchronized
    fun benchmark(promptTokens: Int, genTokens: Int, threadsDecode: Int, threadsBatch: Int): LongArray? {
        if (handle == 0L) return null
        val ptr = handle
        _runnerState.value = LlamaState.GENERATING
        return try {
            LlamaNative.nativeResetCancel(ptr)
            benchmarkRunning = true
            LlamaNative.guarded { LlamaNative.nativeBenchmark(ptr, promptTokens, genTokens, threadsDecode, threadsBatch) }
        } catch (t: Throwable) {
            Log.w(TAG, "Benchmark failed: ${t.message}")
            null
        } finally {
            benchmarkRunning = false
            _runnerState.value = if (handle != 0L) LlamaState.READY else LlamaState.ERROR
        }
    }

    @Volatile private var benchmarkRunning = false

    /** Stops a running [benchmark] (never a real generation). Not synchronized: it must not wait for the run. */
    fun interruptBenchmark() {
        synchronized(handleLock) {
            if (benchmarkRunning && handle != 0L) LlamaNative.nativeCancel(handle)
        }
    }

    private fun loadModelInternal(modelPath: String, contextSize: Int, threadCount: Int): Boolean {
        if (!isNativeLoaded) {
            lastErrorMessage = lastErrorMessage ?: "Unable to initialize the local LLM: native library not loaded."
            return false
        }
        val file = File(modelPath)
        val headerError = validateGguf(file)
        if (headerError != null) {
            lastErrorMessage = headerError
            _runnerState.value = LlamaState.ERROR
            return false
        }
        if (loadedModelPath == modelPath && handle != 0L && requestedContextSize == contextSize) {
            return true
        }
        freeModel()
        _runnerState.value = LlamaState.LOADING
        return try {
            val cores = Runtime.getRuntime().availableProcessors()
            val threads = threadCount.coerceIn(1, maxOf(1, cores))
            val startedAt = SystemClock.elapsedRealtime()
            val ptr = LlamaNative.guarded { LlamaNative.nativeInit(modelPath.toByteArray(Charsets.UTF_8), contextSize, threads) }
            if (ptr == 0L) {
                lastErrorMessage = "Unable to load this model. The file may be damaged or the device may not have enough free memory."
                _runnerState.value = LlamaState.ERROR
                false
            } else {
                synchronized(handleLock) { handle = ptr }
                loadedModelPath = modelPath
                requestedContextSize = contextSize
                this.contextSize = LlamaNative.nativeGetContextSize(ptr)
                firstGenerationPending = true
                lastErrorMessage = null
                _runnerState.value = LlamaState.READY
                Log.i(TAG, "Loaded ${file.name} in ${SystemClock.elapsedRealtime() - startedAt} ms (n_ctx=${this.contextSize}, threads=$threads, cpu=${LlamaNative.cpuVariant})")
                true
            }
        } catch (oom: OutOfMemoryError) {
            lastErrorMessage = "Unable to load the model because the device does not have enough available memory."
            _runnerState.value = LlamaState.ERROR
            false
        } catch (e: Throwable) {
            lastErrorMessage = "Unable to initialize the local LLM. Check the model and device compatibility."
            _runnerState.value = LlamaState.ERROR
            Log.e(TAG, "nativeInit failed: ${e.message}", e)
            false
        }
    }

    /** Number of model tokens in [text]. [isPrompt] counts a fully formatted prompt (BOS + control tokens). */
    @Synchronized
    fun countTokens(text: String, isPrompt: Boolean = false): Int? {
        if (handle == 0L) return null
        return LlamaNative.nativeCountTokens(handle, text.toByteArray(Charsets.UTF_8), isPrompt).takeIf { it >= 0 }
    }

    /** GGUF metadata value, e.g. "general.architecture" or "tokenizer.chat_template". */
    @Synchronized
    fun metadata(key: String): String? {
        if (handle == 0L) return null
        return LlamaNative.nativeGetMetadata(handle, key)?.toString(Charsets.UTF_8)
    }

    /** Applies the model's own chat template; null when the model has none that llama.cpp recognises. */
    @Synchronized
    fun applyChatTemplate(messages: List<ChatMessage>, addAssistantTurn: Boolean = true): String? {
        if (handle == 0L) return null
        val roles = messages.map { it.role }.toTypedArray()
        val contents = messages.map { it.content.toByteArray(Charsets.UTF_8) }.toTypedArray()
        return LlamaNative.nativeApplyChatTemplate(handle, roles, contents, addAssistantTurn)?.toString(Charsets.UTF_8)
    }

    /**
     * Runs generation for a fully formatted prompt. Blocking; throws IllegalStateException on failure.
     * Returns whatever was generated before [token] was cancelled (possibly empty).
     */
    @Synchronized
    fun inference(
        prompt: String,
        maxTokens: Int,
        temperature: Float = 0.2f,
        topP: Float = 0.9f,
        repeatPenalty: Float = 1.1f,
        token: GenerationToken = GenerationToken(),
        seed: Int = 42,
        onPartial: ((String) -> Unit)? = null
    ): String {
        check(isNativeLoaded) { "Native llama_jni library is not loaded." }
        check(handle != 0L) { "No GGUF model is loaded." }
        if (prompt.isBlank() || token.isCancelled) return ""
        val ptr = handle
        val startedAt = SystemClock.elapsedRealtime()
        _runnerState.value = LlamaState.GENERATING
        // Take ownership of the native cancel flag for this request, then re-check: a cancel that
        // raced with the registration is delivered instead of being lost.
        LlamaNative.nativeResetCancel(ptr)
        activeToken = token
        token.onCancel = { cancelNative(token) }
        if (token.isCancelled) LlamaNative.nativeCancel(ptr)
        try {
            val listener = NativeTextListener { bytes -> onPartial?.invoke(bytes.toString(Charsets.UTF_8)) }
            val run = {
                LlamaNative.nativeInference(ptr, prompt.toByteArray(Charsets.UTF_8), maxTokens, temperature, topP, repeatPenalty, seed, listener)
            }
            // The sentinel is armed for the first run after a load, and for every run on crash-prone chips.
            val bytes = if (firstGenerationPending || LlamaNative.alwaysGuarded) LlamaNative.guarded(run).also { firstGenerationPending = false } else run()
            val totalMs = SystemClock.elapsedRealtime() - startedAt
            val resultText = bytes.toString(Charsets.UTF_8)

            // [promptTokens, reusedTokens, prefillMs, firstTokenMs, decodeMs, generatedTokens, threadsDecode, threadsBatch]
            val s = runCatching { LlamaNative.nativeLastStats(ptr) }.getOrNull()?.takeIf { it.size >= 8 }
            val promptTokens = s?.get(0)?.toInt() ?: 0
            val reused = s?.get(1)?.toInt() ?: 0
            val prefillMs = s?.get(2) ?: 0L
            val ttftMs = s?.get(3) ?: totalMs
            val decodeMs = s?.get(4) ?: 0L
            val generatedTokens = s?.get(5)?.toInt() ?: (countTokens(resultText) ?: 0)
            // The decode window starts when the first token is sampled and holds one model step per generated token.
            val tps = if (decodeMs > 0 && generatedTokens > 0) (generatedTokens * 1000.0) / decodeMs else 0.0
            val prefillTps = if (prefillMs > 0) ((promptTokens - reused) * 1000.0) / prefillMs else 0.0
            val modelFilename = loadedModelPath?.let { java.io.File(it).name } ?: "GGUF Model"

            lastInferenceMetrics = InferenceMetrics(
                modelName = modelFilename,
                ttftMs = ttftMs,
                decodeMs = decodeMs,
                totalGenerationMs = totalMs,
                generatedTokens = generatedTokens,
                tokensPerSecond = tps,
                promptTokens = promptTokens,
                reusedPromptTokens = reused,
                prefillMs = prefillMs,
                prefillTokensPerSecond = prefillTps,
                threadsDecode = s?.get(6)?.toInt() ?: 0,
                threadsBatch = s?.get(7)?.toInt() ?: 0
            )
            if (!token.isCancelled) {
                onSpeedSample?.invoke((promptTokens - reused).toLong(), prefillMs, generatedTokens.toLong(), decodeMs)
            }

            Log.i(TAG, "Inference finished in $totalMs ms (prompt $promptTokens tok, reused $reused, prefill ${prefillMs}ms = ${"%.1f".format(java.util.Locale.US, prefillTps)} tok/s, TTFT ${ttftMs}ms, decode ${decodeMs}ms for $generatedTokens tok = ${"%.2f".format(java.util.Locale.US, tps)} tok/s)")
            com.siddu.offlinedocai.ai.telemetry.BenchmarkTelemetry.logGeneration(
                modelName = modelFilename,
                contextSize = this.contextSize,
                promptTokens = promptTokens,
                generatedTokens = generatedTokens,
                ttftMs = ttftMs,
                decodeMs = decodeMs,
                totalGenerationMs = totalMs,
                tokensPerSecond = tps,
                isCancelled = token.isCancelled,
                reusedPromptTokens = reused,
                prefillMs = prefillMs,
                prefillTokensPerSecond = prefillTps,
                threadsDecode = s?.get(6)?.toInt() ?: 0,
                threadsBatch = s?.get(7)?.toInt() ?: 0,
                cpuVariant = LlamaNative.cpuVariant,
                cpuTopology = CpuTopology.device.label
            )
            return resultText
        } catch (e: Throwable) {
            lastErrorMessage = "Local generation failed. The model remains installed, but the current generation could not be completed."
            throw IllegalStateException("Local generation failed: ${e.message}", e)
        } finally {
            token.onCancel = null
            activeToken = null
            _runnerState.value = if (handle != 0L) LlamaState.READY else LlamaState.ERROR
        }
    }

    @Synchronized
    fun freeModel() {
        val ptr: Long
        synchronized(handleLock) {
            ptr = handle
            handle = 0L
        }
        loadedModelPath = null
        contextSize = 0
        requestedContextSize = 0
        if (ptr != 0L) runCatching { LlamaNative.nativeFree(ptr) }.onFailure { Log.e(TAG, "nativeFree failed", it) }
        _runnerState.value = LlamaState.UNLOADED
    }

    /** Stops the generation that is currently running, whichever request it belongs to. */
    fun cancelInference() {
        activeToken?.cancel() ?: synchronized(handleLock) {
            if (isNativeLoaded && handle != 0L) LlamaNative.nativeCancel(handle)
        }
    }

    private fun cancelNative(token: GenerationToken) {
        synchronized(handleLock) {
            if (activeToken === token && handle != 0L) LlamaNative.nativeCancel(handle)
        }
    }

    private fun validateGguf(file: File): String? {
        if (!file.exists()) return "Model file not found at ${file.absolutePath}"
        if (!file.canRead()) return "Installed model cannot be read. Re-import the model."
        if (file.length() < 1024) return "The selected file is not a valid or supported GGUF model."
        return try {
            val magic = ByteArray(4)
            file.inputStream().use { if (it.read(magic) < 4) return "The selected file is not a valid or supported GGUF model." }
            if (String(magic, Charsets.US_ASCII) != "GGUF") "The selected file is not a valid or supported GGUF model." else null
        } catch (e: Exception) {
            "Installed model cannot be read. Re-import the model."
        }
    }
}
