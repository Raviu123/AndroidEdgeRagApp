package com.siddu.offlinedocai.ai.llm

import android.content.Context
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File

/**
 * Lifecycle states for the native llama.cpp runner.
 */
enum class LlamaState {
    UNINITIALIZED,
    LOADING,
    READY,
    GENERATING,
    ERROR,
    UNLOADED
}

/**
 * Thread-safe native JNI runner for llama.cpp on Android.
 *
 * Manages model lifecycle:
 * - Safe native library loading (System.loadLibrary("llama_jni"))
 * - Model instantiation and context management via nativeInit
 * - In-memory model caching across queries to prevent repeated loading overhead
 * - Native token generation via nativeInference
 * - Idempotent resource cleanup via nativeFree
 * - Clear state transitions: UNINITIALIZED -> LOADING -> READY -> GENERATING -> UNLOADED / ERROR
 */
class LlamaRunner(private val context: Context) {

    private val TAG = "LlamaRunner"

    var isNativeLoaded: Boolean = false
        private set

    private var modelContextPointer: Long = 0L
    private var loadedModelPath: String? = null

    private val _runnerState = MutableStateFlow(LlamaState.UNINITIALIZED)
    val runnerState: StateFlow<LlamaState> = _runnerState.asStateFlow()

    val currentState: LlamaState
        get() = _runnerState.value

    val isGenerating: Boolean
        get() = _runnerState.value == LlamaState.GENERATING

    private var lastErrorMessage: String? = null
    fun getLastErrorMessage(): String? = lastErrorMessage

    init {
        try {
            System.loadLibrary("llama_jni")
            isNativeLoaded = true
            _runnerState.value = LlamaState.UNINITIALIZED
            Log.i(TAG, "Native llama_jni library loaded successfully.")
        } catch (e: UnsatisfiedLinkError) {
            isNativeLoaded = false
            _runnerState.value = LlamaState.ERROR
            lastErrorMessage = "Native library not available on platform: ${e.message}"
            Log.w(TAG, "Native llama_jni library not available on this platform: ${e.message}")
        } catch (e: Exception) {
            isNativeLoaded = false
            _runnerState.value = LlamaState.ERROR
            lastErrorMessage = "Unexpected error loading llama_jni: ${e.message}"
            Log.e(TAG, "Unexpected error loading llama_jni: ${e.message}", e)
        }
    }

    /**
     * Checks if a model is currently loaded in memory and ready for inference.
     */
    @Synchronized
    fun isModelLoaded(): Boolean {
        return isNativeLoaded && modelContextPointer != 0L && _runnerState.value == LlamaState.READY
    }

    /**
     * Gets the file path of the currently loaded model, if any.
     */
    @Synchronized
    fun getLoadedModelPath(): String? = loadedModelPath

    /**
     * Loads a GGUF model from a filesystem path into llama.cpp.
     * Reuses the existing model context if the same model file is already loaded.
     */
    @Synchronized
    fun loadModel(
        modelPath: String,
        contextSize: Int = 2048,
        threadCount: Int = 4
    ): Boolean {
        if (!isNativeLoaded) {
            lastErrorMessage = "Native llama_jni library is not loaded."
            _runnerState.value = LlamaState.ERROR
            Log.w(TAG, "Cannot load model: native llama_jni library is not loaded.")
            return false
        }

        if (modelPath.isBlank()) {
            lastErrorMessage = "modelPath is empty."
            _runnerState.value = LlamaState.ERROR
            Log.e(TAG, "Cannot load model: modelPath is empty.")
            return false
        }

        val file = File(modelPath)
        if (!file.exists()) {
            lastErrorMessage = "Model file does not exist at $modelPath"
            _runnerState.value = LlamaState.ERROR
            Log.e(TAG, "Cannot load model: file does not exist at $modelPath")
            return false
        }
        if (!file.canRead()) {
            lastErrorMessage = "Model file exists but cannot be read at $modelPath"
            _runnerState.value = LlamaState.ERROR
            Log.e(TAG, "Cannot load model: file cannot be read at $modelPath")
            return false
        }
        if (file.length() < 1024) {
            lastErrorMessage = "Model file is invalid or empty (< 1024 bytes) at $modelPath"
            _runnerState.value = LlamaState.ERROR
            Log.e(TAG, "Cannot load model: file size too small at $modelPath")
            return false
        }

        // Validate GGUF Magic Header
        try {
            file.inputStream().use { input ->
                val magic = ByteArray(4)
                val read = input.read(magic)
                if (read < 4 || String(magic) != "GGUF") {
                    lastErrorMessage = "The selected file is not a valid or supported GGUF model."
                    _runnerState.value = LlamaState.ERROR
                    Log.e(TAG, "Invalid GGUF file header at $modelPath! Header: ${String(magic)}")
                    return false
                }
            }
        } catch (e: Exception) {
            lastErrorMessage = "Failed reading GGUF header from $modelPath: ${e.message}"
            _runnerState.value = LlamaState.ERROR
            Log.e(TAG, "Failed reading GGUF header from $modelPath: ${e.message}")
            return false
        }

        // Reuse model context if already loaded for this path
        if (loadedModelPath == modelPath && modelContextPointer != 0L && _runnerState.value == LlamaState.READY) {
            Log.d(TAG, "Model already loaded from $modelPath. Reusing context.")
            return true
        }

        // Free previous model context if path differs
        freeModel()

        _runnerState.value = LlamaState.LOADING

        Log.i("MODEL_VERIFICATION", "==================================================")
        Log.i("MODEL_VERIFICATION", "MODEL_PATH=${file.absolutePath}")
        Log.i("MODEL_VERIFICATION", "MODEL_FILENAME=${file.name}")
        Log.i("MODEL_VERIFICATION", "MODEL_FILE_SIZE=${file.length()}")
        Log.i("MODEL_VERIFICATION", "GGUF_MAGIC_VERIFIED=true")
        Log.i("MODEL_VERIFICATION", "==================================================")

        return try {
            val cores = Runtime.getRuntime().availableProcessors()
            val effectiveThreads = if (threadCount in 1..cores) threadCount else maxOf(1, minOf(cores, 4))
            val startTime = System.currentTimeMillis()

            val ptr = nativeInit(modelPath, contextSize, effectiveThreads)
            if (ptr != 0L) {
                modelContextPointer = ptr
                loadedModelPath = modelPath
                _runnerState.value = LlamaState.READY
                lastErrorMessage = null
                val elapsed = System.currentTimeMillis() - startTime
                Log.i(TAG, "GGUF model initialized successfully in $elapsed ms from: $modelPath (threads=$effectiveThreads, ctx=$contextSize)")
                true
            } else {
                lastErrorMessage = "Unable to initialize the local LLM. Check the model and device compatibility."
                _runnerState.value = LlamaState.ERROR
                Log.e(TAG, "nativeInit returned 0 (null context) for model: $modelPath")
                false
            }
        } catch (oom: OutOfMemoryError) {
            lastErrorMessage = "Unable to load the model because the device does not have enough available memory."
            _runnerState.value = LlamaState.ERROR
            Log.e(TAG, "OOM during nativeInit: ${oom.message}", oom)
            freeModel()
            false
        } catch (e: Throwable) {
            val msg = e.message ?: "Unknown native error"
            lastErrorMessage = if (msg.contains("memory", ignoreCase = true) || msg.contains("alloc", ignoreCase = true)) {
                "Unable to load the model because the device does not have enough available memory."
            } else {
                "Unable to initialize the local LLM. Check the model and device compatibility."
            }
            _runnerState.value = LlamaState.ERROR
            Log.e(TAG, "Exception during nativeInit: ${e.message}", e)
            freeModel()
            false
        }
    }

    /**
     * Executes neural inference on the loaded model.
     * Returns the raw generated string, or throws an IllegalStateException on failure.
     */
    @Synchronized
    fun inference(
        prompt: String,
        maxTokens: Int = 384,
        temperature: Float = 0.1f
    ): String {
        if (!isNativeLoaded) {
            throw IllegalStateException("Native llama_jni library is not loaded.")
        }
        if (modelContextPointer == 0L) {
            throw IllegalStateException("No GGUF model is loaded. Call loadModel() first.")
        }
        if (prompt.isBlank()) {
            return ""
        }

        val inferenceStartedAt = SystemClock.elapsedRealtime()
        _runnerState.value = LlamaState.GENERATING
        return try {
            val result = nativeInference(modelContextPointer, prompt, maxTokens, temperature)
            if (result.startsWith("Error:")) {
                Log.e(TAG, "nativeInference error: $result")
                _runnerState.value = LlamaState.ERROR
                lastErrorMessage = "Local generation failed. The model remains installed, but the current generation could not be completed."
                throw IllegalStateException("Local generation failed: $result")
            }
            _runnerState.value = LlamaState.READY
            Log.i(
                "CHAT_LATENCY",
                "gguf_native_inference_ms=${SystemClock.elapsedRealtime() - inferenceStartedAt} prompt_chars=${prompt.length} output_chars=${result.length} max_tokens=$maxTokens"
            )
            result
        } catch (e: Throwable) {
            Log.e(TAG, "Exception during nativeInference: ${e.message}", e)
            _runnerState.value = if (modelContextPointer != 0L) LlamaState.READY else LlamaState.ERROR
            lastErrorMessage = "Local generation failed. The model remains installed, but the current generation could not be completed."
            throw e
        }
    }

    /**
     * Frees the native model context and releases all llama.cpp memory.
     * Safe to call multiple times (idempotent).
     */
    @Synchronized
    fun freeModel() {
        val ptr = modelContextPointer
        if (ptr != 0L) {
            try {
                nativeFree(ptr)
                Log.i(TAG, "Native model context ($ptr) freed successfully.")
            } catch (e: Throwable) {
                Log.e(TAG, "Error freeing native model context: ${e.message}", e)
            } finally {
                modelContextPointer = 0L
                loadedModelPath = null
                _runnerState.value = LlamaState.UNLOADED
            }
        } else {
            _runnerState.value = LlamaState.UNLOADED
        }
    }

    // JNI Native methods matching llama_jni.cpp
    private external fun nativeInit(modelPath: String, contextSize: Int, threadCount: Int): Long
    private external fun nativeFree(modelContext: Long)
    private external fun nativeInference(modelContext: Long, prompt: String, maxTokens: Int, temperature: Float): String
    private external fun nativeCancel()

    fun cancelInference() {
        if (isNativeLoaded) {
            nativeCancel()
            if (_runnerState.value == LlamaState.GENERATING) {
                _runnerState.value = LlamaState.READY
            }
        }
    }
}
