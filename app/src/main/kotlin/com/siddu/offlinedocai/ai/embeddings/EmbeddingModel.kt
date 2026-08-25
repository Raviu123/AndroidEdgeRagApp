package com.siddu.offlinedocai.ai.embeddings

/**
 * Interface for an embedding model that can transform a piece of text into a vector of floats.
 * Implementations should load a GGUF model file from the app's private storage and run inference
 * locally on the device. For the prototype a deterministic fallback is provided.
 */
interface EmbeddingModel {
    /**
     * Compute an embedding for the given [text]. The returned FloatArray length is defined by the
     * underlying model (e.g., 384, 512, 768). The function runs on a background dispatcher.
     */
    suspend fun getEmbedding(text: String): FloatArray
}
