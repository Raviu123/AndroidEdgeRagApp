package com.siddu.offlinedocai.ai.embeddings

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Primary embedding engine for OfflineDocAI.
 * Delegates to [ONNXEmbeddingEngine] to run actual local on-device neural embeddings
 * via all-MiniLM-L6-v2.onnx (384-dimensional).
 */
class GGUFEmbeddingEngine(
    context: Context,
    allowMockInJvmTests: Boolean = true
) : EmbeddingModel {

    private val delegate = ONNXEmbeddingEngine(context, allowMockInJvmTests = allowMockInJvmTests)

    override suspend fun getEmbedding(text: String): FloatArray = withContext(Dispatchers.Default) {
        delegate.getEmbedding(text)
    }

    fun isLoaded(): Boolean = delegate.isLoaded()

    fun close() {
        delegate.close()
    }
}
