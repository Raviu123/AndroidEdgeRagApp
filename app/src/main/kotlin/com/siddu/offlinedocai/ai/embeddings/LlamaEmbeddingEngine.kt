package com.siddu.offlinedocai.ai.embeddings

import android.content.Context
import android.util.Log
import com.siddu.offlinedocai.ai.llm.CpuInfo
import com.siddu.offlinedocai.ai.llm.LlamaNative
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.sqrt

/**
 * Sentence embeddings from a BERT-family GGUF model through llama.cpp (the same native engine as the
 * language model, with its own context). Used on phones where ONNX Runtime does not work.
 */
class LlamaEmbeddingEngine(
    context: Context,
    val spec: EmbeddingModelSpec,
    private val modelFile: File
) : EmbeddingModel {

    private val TAG = "LlamaEmbeddingEngine"
    private val appContext = context.applicationContext
    private var handle = 0L
    @Volatile var lastError: String? = null
        private set

    override val modelId: String get() = spec.id

    @Synchronized
    fun initialize(): Boolean {
        if (handle != 0L) return true
        if (!LlamaNative.ensureInitialized(appContext)) {
            lastError = LlamaNative.loadError ?: "The native inference library is unavailable."
            return false
        }
        if (!modelFile.exists()) {
            lastError = "Model file is missing. Download it again."
            return false
        }
        return try {
            val ptr = LlamaNative.guarded {
                LlamaNative.nativeEmbedInit(modelFile.absolutePath.toByteArray(Charsets.UTF_8), spec.maxSequenceLength, minOf(4, CpuInfo.performanceCores))
            }
            if (ptr == 0L) {
                lastError = "llama.cpp could not load this embedding model."
                false
            } else {
                val dim = LlamaNative.nativeEmbedDim(ptr)
                if (dim != spec.dimension) {
                    LlamaNative.nativeEmbedFree(ptr)
                    lastError = "Unexpected embedding size $dim (expected ${spec.dimension})."
                    false
                } else {
                    handle = ptr
                    lastError = null
                    true
                }
            }
        } catch (t: Throwable) {
            lastError = t.message ?: t.javaClass.simpleName
            Log.e(TAG, "Failed to load GGUF embedding model", t)
            false
        }
    }

    override suspend fun getEmbedding(text: String): FloatArray = withContext(Dispatchers.Default) { embedOne(text) }

    override suspend fun embedQuery(text: String): FloatArray = withContext(Dispatchers.Default) { embedOne(spec.queryPrefix + text) }

    override suspend fun embedDocuments(texts: List<String>): List<FloatArray> = withContext(Dispatchers.Default) { texts.map { embedOne(it) } }

    @Synchronized
    private fun embedOne(text: String): FloatArray {
        if (!initialize()) throw IllegalStateException("The on-device embedding model failed to load: $lastError")
        val raw = LlamaNative.nativeEmbed(handle, text.trim().ifEmpty { " " }.toByteArray(Charsets.UTF_8))
            ?: throw IllegalStateException("Embedding inference failed for the GGUF model.")
        var sumSq = 0.0
        for (v in raw) sumSq += (v * v).toDouble()
        val norm = sqrt(sumSq).toFloat()
        check(norm > 1e-12f && raw.none { it.isNaN() }) { "Embedding inference produced an invalid vector" }
        for (i in raw.indices) raw[i] /= norm
        return raw
    }

    @Synchronized
    fun close() {
        if (handle != 0L) runCatching { LlamaNative.nativeEmbedFree(handle) }
        handle = 0L
    }
}
