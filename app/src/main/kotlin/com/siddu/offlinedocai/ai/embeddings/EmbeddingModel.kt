package com.siddu.offlinedocai.ai.embeddings

/**
 * A sentence-embedding model. Documents and queries are embedded separately because some models
 * (e.g. BGE) expect an instruction prefix on queries only. Implementations never return placeholder
 * vectors: a failure is thrown so it can be shown to the user.
 */
interface EmbeddingModel {
    /** Embedding of a document passage (chunk). */
    suspend fun getEmbedding(text: String): FloatArray

    /** Embedding of a search query. */
    suspend fun embedQuery(text: String): FloatArray = getEmbedding(text)

    /** Embeddings of several passages; implementations may batch them. */
    suspend fun embedDocuments(texts: List<String>): List<FloatArray> = texts.map { getEmbedding(it) }

    /** Identifier stored with every vector; vectors with different ids are never compared. */
    val modelId: String
}
