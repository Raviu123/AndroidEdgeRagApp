package com.siddu.offlinedocai.ai.llm

import android.content.Context
import android.util.Log
import java.io.File

/**
 * Thread-safe native JNI runner for llama.cpp on Android.
 *
 * Manages model lifecycle:
 * - Safe native library loading (System.loadLibrary("llama_jni"))
 * - Model instantiation and context management via nativeInit
 * - In-memory model caching across queries to prevent repeated loading overhead
 * - Native token generation via nativeInference
 * - Idempotent resource cleanup via nativeFree
 */
class LlamaRunner(private val context: Context) {

    private val TAG = "LlamaRunner"

    var isNativeLoaded: Boolean = false
        private set

    private var modelContextPointer: Long = 0L
    private var loadedModelPath: String? = null

    init {
        try {
            System.loadLibrary("llama_jni")
            isNativeLoaded = true
            Log.i(TAG, "Native llama_jni library loaded successfully.")
        } catch (e: UnsatisfiedLinkError) {
            isNativeLoaded = false
            Log.w(TAG, "Native llama_jni library not available on this platform: ${e.message}")
        } catch (e: Exception) {
            isNativeLoaded = false
            Log.e(TAG, "Unexpected error loading llama_jni: ${e.message}", e)
        }
    }

    /**
     * Checks if a model is currently loaded in memory and ready for inference.
     */
    @Synchronized
    fun isModelLoaded(): Boolean {
        return isNativeLoaded && modelContextPointer != 0L
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
            Log.w(TAG, "Cannot load model: native llama_jni library is not loaded.")
            return false
        }

        if (modelPath.isBlank()) {
            Log.e(TAG, "Cannot load model: modelPath is empty.")
            return false
        }

        val file = File(modelPath)
        if (!file.exists() || !file.canRead() || file.length() < 1024) {
            Log.e(TAG, "Cannot load model: file does not exist or is invalid at $modelPath")
            return false
        }

        // Validate GGUF Magic Header
        try {
            file.inputStream().use { input ->
                val magic = ByteArray(4)
                val read = input.read(magic)
                if (read < 4 || String(magic) != "GGUF") {
                    Log.e(TAG, "Invalid GGUF file header at $modelPath! Header: ${String(magic)}")
                    return false
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed reading GGUF header from $modelPath: ${e.message}")
            return false
        }

        Log.i("MODEL_VERIFICATION", "==================================================")
        Log.i("MODEL_VERIFICATION", "MODEL_PATH=${file.absolutePath}")
        Log.i("MODEL_VERIFICATION", "MODEL_FILENAME=${file.name}")
        Log.i("MODEL_VERIFICATION", "MODEL_FILE_SIZE=${file.length()}")
        Log.i("MODEL_VERIFICATION", "GGUF_MAGIC_VERIFIED=true")
        Log.i("MODEL_VERIFICATION", "==================================================")

        // Reuse model context if already loaded for this path
        if (loadedModelPath == modelPath && modelContextPointer != 0L) {
            Log.d(TAG, "Model already loaded from $modelPath. Reusing context.")
            return true
        }

        // Free previous model context if path differs
        freeModel()

        return try {
            val cores = Runtime.getRuntime().availableProcessors()
            val effectiveThreads = if (threadCount in 1..cores) threadCount else maxOf(1, minOf(cores, 4))
            val startTime = System.currentTimeMillis()

            val ptr = nativeInit(modelPath, contextSize, effectiveThreads)
            if (ptr != 0L) {
                modelContextPointer = ptr
                loadedModelPath = modelPath
                val elapsed = System.currentTimeMillis() - startTime
                Log.i(TAG, "GGUF model initialized successfully in $elapsed ms from: $modelPath (threads=$effectiveThreads, ctx=$contextSize)")
                true
            } else {
                Log.e(TAG, "nativeInit returned 0 (null context) for model: $modelPath")
                false
            }
        } catch (e: Throwable) {
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

        return try {
            val result = nativeInference(modelContextPointer, prompt, maxTokens, temperature)
            if (result.startsWith("Error:")) {
                Log.e(TAG, "nativeInference error: $result")
                throw IllegalStateException("Llama native inference failed: $result")
            }
            result
        } catch (e: Throwable) {
            Log.e(TAG, "Exception during nativeInference: ${e.message}", e)
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
            }
        }
    }

    // JNI Native methods matching llama_jni.cpp
    private external fun nativeInit(modelPath: String, contextSize: Int, threadCount: Int): Long
    private external fun nativeFree(modelContext: Long)
    private external fun nativeInference(modelContext: Long, prompt: String, maxTokens: Int, temperature: Float): String
}
