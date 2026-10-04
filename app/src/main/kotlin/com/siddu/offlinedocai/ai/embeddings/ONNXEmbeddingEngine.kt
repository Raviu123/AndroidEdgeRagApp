package com.siddu.offlinedocai.ai.embeddings

import android.content.Context
import android.os.SystemClock
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.InputStream
import java.nio.LongBuffer
import kotlin.math.sqrt

/**
 * Production On-Device Semantic Embedding Engine powered by ONNX Runtime.
 *
 * Model: all-MiniLM-L6-v2 (Sentence-Transformers)
 * Output Dimension: 384-dimensional dense semantic vector
 * Pooling: Mean Pooling across attention mask
 * Normalization: L2 unit normalization
 *
 * Thread-safe, reuses session, throws real exceptions if inference fails,
 * and NEVER returns an all-zero vector in production.
 */
class ONNXEmbeddingEngine(
    private val context: Context,
    private val allowMockInJvmTests: Boolean = false
) : EmbeddingModel {

    private val TAG = "ONNXEmbeddingEngine"

    companion object {
        const val MODEL_IDENTIFIER = "all-MiniLM-L6-v2-v1"
        const val EMBEDDING_DIM = 384
        const val MAX_SEQUENCE_LENGTH = 256
    }

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var tokenizer: WordPieceTokenizer? = null
    private var isInitialized = false

    @Synchronized
    fun initialize(): Boolean {
        if (isInitialized && ortSession != null && tokenizer != null) {
            return true
        }

        val initializationStartedAt = SystemClock.elapsedRealtime()
        try {
            val modelDir = File(context.filesDir, "models/embedding")
            if (!modelDir.exists()) modelDir.mkdirs()

            val onnxFile = File(modelDir, "all-MiniLM-L6-v2.onnx")
            val vocabFile = File(modelDir, "vocab.txt")

            // 1. Extract bundled model and vocab from assets if needed
            if (!onnxFile.exists() || onnxFile.length() < 1024) {
                Log.i(TAG, "Extracting bundled all-MiniLM-L6-v2.onnx from assets...")
                context.assets.open("models/embedding/all-MiniLM-L6-v2.onnx").use { input ->
                    onnxFile.outputStream().use { output -> input.copyTo(output) }
                }
            }

            if (!vocabFile.exists() || vocabFile.length() < 100) {
                Log.i(TAG, "Extracting bundled vocab.txt from assets...")
                context.assets.open("models/embedding/vocab.txt").use { input ->
                    vocabFile.outputStream().use { output -> input.copyTo(output) }
                }
            }

            if (!onnxFile.exists() || !vocabFile.exists()) {
                Log.e(TAG, "Failed to resolve ONNX embedding model or vocab files on disk")
                return false
            }

            // 2. Initialize OrtEnvironment, OrtSession and WordPieceTokenizer
            val env = OrtEnvironment.getEnvironment()
            val sessionOptions = OrtSession.SessionOptions()
            sessionOptions.setIntraOpNumThreads(2)
            val session = env.createSession(onnxFile.absolutePath, sessionOptions)

            tokenizer = WordPieceTokenizer(vocabFile)
            ortEnv = env
            ortSession = session
            isInitialized = true

            Log.i(
                TAG,
                "ONNX embedding engine initialized successfully (Model: ${onnxFile.name}, ${onnxFile.length()} bytes)"
            )
            Log.i("CHAT_LATENCY", "onnx_cold_initialize_ms=${SystemClock.elapsedRealtime() - initializationStartedAt}")
            return true
        } catch (e: Throwable) {
            Log.e(TAG, "Failed to initialize ONNX embedding session: ${e.message}", e)
            close()
            return false
        }
    }

    fun isLoaded(): Boolean = isInitialized && ortSession != null && tokenizer != null

    override suspend fun getEmbedding(text: String): FloatArray = withContext(Dispatchers.Default) {
        val embeddingStartedAt = SystemClock.elapsedRealtime()
        val cleanText = text.trim()
        if (cleanText.isEmpty()) {
            val emptyVec = FloatArray(EMBEDDING_DIM) { 0.0001f }
            return@withContext normalize(emptyVec)
        }

        // Ensure session is initialized
        if (!isLoaded()) {
            val success = initialize()
            if (!success) {
                if (allowMockInJvmTests) {
                    Log.w(TAG, "ONNX unavailable in JVM test mode; using deterministic semantic mock.")
                    return@withContext computeDeterministicMockEmbedding(cleanText)
                }
                throw IllegalStateException("ONNX embedding model failed to load. Check assets and storage.")
            }
        }

        val session = ortSession ?: throw IllegalStateException("ONNX session is null")
        val tok = tokenizer ?: throw IllegalStateException("Tokenizer is null")
        val env = ortEnv ?: OrtEnvironment.getEnvironment()

        val tokens = tok.tokenize(cleanText)
        val (inputIds, attentionMask) = tok.padOrTruncate(tokens, MAX_SEQUENCE_LENGTH)
        val tensorStartedAt = SystemClock.elapsedRealtime()

        val shape = longArrayOf(1, MAX_SEQUENCE_LENGTH.toLong())
        val inputs = HashMap<String, OnnxTensor>()

        try {
            val inputTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds), shape)
            inputs["input_ids"] = inputTensor

            val maskTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(attentionMask), shape)
            inputs["attention_mask"] = maskTensor

            if (session.inputNames.contains("token_type_ids")) {
                val tokenTypeIds = LongArray(MAX_SEQUENCE_LENGTH) { 0L }
                val typeTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(tokenTypeIds), shape)
                inputs["token_type_ids"] = typeTensor
            }

            Log.i("CHAT_LATENCY", "onnx_query_tensor_prepare_ms=${SystemClock.elapsedRealtime() - tensorStartedAt}")
            val inferenceStartedAt = SystemClock.elapsedRealtime()
            session.run(inputs).use { results ->
                val output = results[0].value
                val pooledVector = when (output) {
                    is Array<*> -> {
                        val firstElem = output[0]
                        if (firstElem is FloatArray) {
                            // Direct sentence embedding [1, 384]
                            firstElem.copyOf()
                        } else if (firstElem is Array<*>) {
                            // 3D token embeddings: [1, seq_len, 384]
                            @Suppress("UNCHECKED_CAST")
                            val tokenEmbeddings = firstElem as Array<FloatArray>
                            val hiddenSize = tokenEmbeddings[0].size
                            val meanVector = FloatArray(hiddenSize) { 0f }
                            var validTokens = 0

                            for (i in 0 until minOf(tokenEmbeddings.size, attentionMask.size)) {
                                if (attentionMask[i] == 1L) {
                                    validTokens++
                                    val tokenVec = tokenEmbeddings[i]
                                    for (j in 0 until hiddenSize) {
                                        meanVector[j] += tokenVec[j]
                                    }
                                }
                            }
                            if (validTokens > 0) {
                                for (j in 0 until hiddenSize) {
                                    meanVector[j] /= validTokens.toFloat()
                                }
                            }
                            meanVector
                        } else {
                            throw IllegalStateException("Unexpected tensor inner element type: ${firstElem?.javaClass}")
                        }
                    }
                    else -> throw IllegalStateException("Unexpected output tensor type: ${output?.javaClass}")
                }

                // Verify non-zero and L2 normalize
                val normalized = normalize(pooledVector)
                if (isAllZero(normalized)) {
                    throw IllegalStateException("Embedding inference produced an all-zero vector for text: '$cleanText'")
                }
                Log.i(
                    "CHAT_LATENCY",
                    "onnx_query_inference_ms=${SystemClock.elapsedRealtime() - inferenceStartedAt} total_embedding_ms=${SystemClock.elapsedRealtime() - embeddingStartedAt}"
                )
                normalized
            }
        } catch (e: Throwable) {
            Log.e(TAG, "Embedding inference failed: ${e.message}", e)
            if (allowMockInJvmTests) {
                return@withContext computeDeterministicMockEmbedding(cleanText)
            }
            throw IllegalStateException("ONNX inference failed: ${e.message}", e)
        } finally {
            inputs.values.forEach { tensor ->
                try {
                    tensor.close()
                } catch (closeEx: Throwable) {
                    Log.w(TAG, "Failed closing tensor: ${closeEx.message}")
                }
            }
        }
    }

    private fun normalize(vector: FloatArray): FloatArray {
        var sumSq = 0.0
        for (v in vector) {
            sumSq += (v * v).toDouble()
        }
        val norm = sqrt(sumSq).toFloat()
        if (norm > 1e-12f) {
            for (i in vector.indices) {
                vector[i] /= norm
            }
        }
        return vector
    }

    private fun isAllZero(vector: FloatArray): Boolean {
        for (v in vector) {
            if (v != 0f) return false
        }
        return true
    }

    /**
     * Deterministic pseudo-semantic embedding for JVM unit tests where native ONNX runtime is absent.
     * Uses dispersion hashing across character n-grams and words to produce consistent similarity scores.
     */
    private fun computeDeterministicMockEmbedding(text: String): FloatArray {
        val vector = FloatArray(EMBEDDING_DIM) { 0f }
        val words = text.lowercase()
            .split(Regex("[^a-zA-Z0-9]+"))
            .filter { it.isNotBlank() }

        if (words.isEmpty()) {
            vector[0] = 1f
            return vector
        }

        for (w in words) {
            val h1 = Math.abs(w.hashCode()) % EMBEDDING_DIM
            val h2 = Math.abs((w + "_salt").hashCode()) % EMBEDDING_DIM
            vector[h1] += 1.0f
            vector[h2] += 0.5f
        }

        return normalize(vector)
    }

    @Synchronized
    fun close() {
        try {
            ortSession?.close()
            ortSession = null
            ortEnv?.close()
            ortEnv = null
            tokenizer = null
            isInitialized = false
            Log.i(TAG, "ONNX embedding engine resources closed.")
        } catch (e: Throwable) {
            Log.w(TAG, "Error closing ONNX resources: ${e.message}")
        }
    }
}
