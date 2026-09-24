package com.example.offlinedocai.ai.embeddings

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.LongBuffer
import kotlin.math.sqrt

class EmbeddingModel(private val context: Context) {
    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var tokenizer: WordPieceTokenizer? = null
    private var modelPath: String? = null

    companion object {
        private const val TAG = "EmbeddingModel"
        private const val DEFAULT_EMBEDDING_DIM = 384
    }

    /**
     * Load the ONNX model from the specified file path.
     * Expects a vocab.txt file to exist in the same directory.
     */
    fun loadModel(onnxPath: String): Boolean {
        try {
            close() // Clean up existing session

            val modelFile = File(onnxPath)
            if (!modelFile.exists()) {
                Log.e(TAG, "ONNX model file not found at: $onnxPath")
                return false
            }

            // Look for vocab.txt in the same folder
            val parentDir = modelFile.parentFile
            val vocabFile = File(parentDir, "vocab.txt")
            if (!vocabFile.exists()) {
                Log.e(TAG, "vocab.txt not found in same directory as ONNX model: ${vocabFile.absolutePath}")
                return false
            }

            env = OrtEnvironment.getEnvironment()
            session = env?.createSession(onnxPath, OrtSession.SessionOptions())
            tokenizer = WordPieceTokenizer(vocabFile)
            modelPath = onnxPath
            Log.d(TAG, "Successfully loaded ONNX embedding model from $onnxPath")
            return true
        } catch (e: Exception) {
            Log.e(TAG, "Error loading ONNX embedding model: ${e.message}", e)
            close()
            return false
        }
    }

    fun isModelLoaded(): Boolean {
        return session != null && tokenizer != null
    }

    fun getLoadedModelPath(): String? = modelPath

    /**
     * Generate embedding vector (FloatArray) for the given text.
     * Throws an IllegalStateException if the model is not loaded.
     */
    fun getEmbedding(text: String): FloatArray {
        val currentSession = session
        val currentTokenizer = tokenizer

        if (currentSession != null && currentTokenizer != null) {
            val inputs = mutableMapOf<String, OnnxTensor>()
            try {
                val tokens = currentTokenizer.tokenize(text)
                val maxLength = 256
                val (inputIds, attentionMask) = currentTokenizer.padOrTruncate(tokens, maxLength)

                val ortEnv = env ?: OrtEnvironment.getEnvironment().also { env = it }

                // Create input tensors
                val shape = longArrayOf(1, maxLength.toLong())
                val inputTensor = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(inputIds), shape)
                inputs["input_ids"] = inputTensor
                val maskTensor = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(attentionMask), shape)
                inputs["attention_mask"] = maskTensor

                // MiniLM / BERT models might also need token_type_ids, add if present in session inputs
                if (currentSession.inputNames.contains("token_type_ids")) {
                    val tokenTypeIds = LongArray(maxLength) { 0L }
                    val typeTensor = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(tokenTypeIds), shape)
                    inputs["token_type_ids"] = typeTensor
                }

                currentSession.run(inputs).use { results ->
                    val outputValue = results[0]
                    // Standard SentenceTransformers return token embeddings [1, sequence_length, hidden_size]
                    // or sentence embedding directly [1, hidden_size]
                    val floatArray = (outputValue.value as Array<*>)[0]

                    return if (floatArray is FloatArray) {
                        // Directly returned 1D sentence embedding
                        normalize(floatArray)
                    } else if (floatArray is Array<*>) {
                        // 2D token embeddings [sequence_length, hidden_size], perform mean pooling
                        val tokenEmbeddings = floatArray as Array<FloatArray>
                        val meanEmbedding = FloatArray(tokenEmbeddings[0].size)

                        var validTokens = 0
                        for (i in 0 until minOf(tokenEmbeddings.size, attentionMask.size)) {
                            if (attentionMask[i] == 1L) {
                                validTokens++
                                for (j in meanEmbedding.indices) {
                                    meanEmbedding[j] += tokenEmbeddings[i][j]
                                }
                            }
                        }
                        if (validTokens > 0) {
                            for (j in meanEmbedding.indices) {
                                meanEmbedding[j] /= validTokens.toFloat()
                            }
                        }
                        normalize(meanEmbedding)
                    } else {
                        throw IllegalStateException("Unexpected tensor shape: ${outputValue.info}")
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "ONNX inference failed: ${e.message}", e)
                throw IllegalStateException("Embedding inference failed", e)
            } finally {
                inputs.values.forEach {
                    try {
                        it.close()
                    } catch (ex: Exception) {
                        Log.e(TAG, "Error closing tensor: ${ex.message}")
                    }
                }
            }
        }
        // No model loaded – signal caller
        throw IllegalStateException("Embedding model not loaded. Please import a compatible ONNX model via Settings.")
    }

    /**
     * Generates a normalized, 384-dimensional vector based on the words in the text.
     * Guaranteed to produce similar vectors for texts with overlapping vocabulary.
     */
    private fun getFallbackEmbedding(text: String): FloatArray {
        val vector = FloatArray(DEFAULT_EMBEDDING_DIM) { 0f }
        val words = text.lowercase()
            .split(Regex("[^a-zA-Z0-9]+"))
            .filter { it.length > 1 }

        if (words.isEmpty()) {
            vector[0] = 1f
            return vector
        }

        for (word in words) {
            // Hash the word deterministically to map to vector dimensions
            val hash1 = word.hashCode().coerceAtLeast(0)
            val index1 = hash1 % DEFAULT_EMBEDDING_DIM
            vector[index1] += 1.0f

            // Double hash for dispersion
            val hash2 = (word + "_disp").hashCode().coerceAtLeast(0)
            val index2 = hash2 % DEFAULT_EMBEDDING_DIM
            vector[index2] += 0.5f
        }

        return normalize(vector)
    }

    private fun normalize(vector: FloatArray): FloatArray {
        var sumSq = 0f
        for (v in vector) {
            sumSq += v * v
        }
        val norm = sqrt(sumSq.toDouble()).toFloat()
        if (norm > 0f) {
            for (i in vector.indices) {
                vector[i] /= norm
            }
        }
        return vector
    }

    fun close() {
        try {
            session?.close()
            session = null
            env?.close()
            env = null
            tokenizer = null
            modelPath = null
        } catch (e: Exception) {
            Log.e(TAG, "Error closing embedding model resources: ${e.message}")
        }
    }
}
