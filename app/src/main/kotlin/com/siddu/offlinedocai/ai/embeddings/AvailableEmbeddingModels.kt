package com.siddu.offlinedocai.ai.embeddings

import com.siddu.offlinedocai.ai.llm.formatDownloadSize

enum class EmbeddingBackend { ONNX, LLAMA_CPP }

enum class EmbeddingPooling { MEAN, CLS }

/**
 * One selectable embedding model. Sizes and SHA-256 values are the exact ones published on Hugging
 * Face (verified 2026-10-07); a download that differs is rejected.
 */
data class EmbeddingModelSpec(
    /** Stored with every chunk vector. Changing it forces documents to be re-indexed. */
    val id: String,
    val name: String,
    val backend: EmbeddingBackend,
    val fileName: String,
    val downloadUrl: String,
    val sizeBytes: Long,
    val sha256: String,
    val dimension: Int,
    val pooling: EmbeddingPooling,
    /** Model input limit in tokens, including [CLS]/[SEP]. */
    val maxSequenceLength: Int,
    /** Prepended to queries only (BGE retrieval instruction). */
    val queryPrefix: String = "",
    /**
     * Cosine similarity below which a passage is considered unrelated to the question (model specific:
     * BGE scores everything higher than MiniLM). Used by the "not in the document" gate.
     */
    val relevanceFloor: Float = 0.30f,
    val badge: String,
    val description: String,
    /** Asset path when the model ships inside the APK (no download needed, cannot be deleted). */
    val bundledAsset: String? = null
) {
    val isBundled: Boolean get() = bundledAsset != null
    val downloadSize: String get() = formatDownloadSize(sizeBytes)
    val backendLabel: String get() = if (backend == EmbeddingBackend.ONNX) "ONNX Runtime" else "llama.cpp"
}

object AvailableEmbeddingModels {
    const val MINILM_ONNX_ID = "all-MiniLM-L6-v2-v1"
    const val BGE_SMALL_ONNX_ID = "bge-small-en-v1.5-int8-v1"
    const val MINILM_GGUF_ID = "all-MiniLM-L6-v2-gguf-q4km-v1"

    /** Shared by the two ONNX models (identical uncased BERT vocabulary). */
    const val VOCAB_ASSET = "models/embedding/vocab.txt"

    val list: List<EmbeddingModelSpec> = listOf(
        EmbeddingModelSpec(
            id = MINILM_ONNX_ID,
            name = "all-MiniLM-L6-v2 (ONNX INT8)",
            backend = EmbeddingBackend.ONNX,
            fileName = "all-MiniLM-L6-v2.onnx",
            downloadUrl = "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx",
            sizeBytes = 22_972_370L,
            sha256 = "afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1",
            dimension = 384,
            pooling = EmbeddingPooling.MEAN,
            maxSequenceLength = 256,
            badge = "Default · Fast",
            description = "Small and fast general-purpose search model. Built into the app.",
            bundledAsset = "models/embedding/all-MiniLM-L6-v2.onnx"
        ),
        EmbeddingModelSpec(
            id = BGE_SMALL_ONNX_ID,
            name = "BGE Small EN v1.5 (ONNX INT8)",
            backend = EmbeddingBackend.ONNX,
            fileName = "bge-small-en-v1.5-int8.onnx",
            downloadUrl = "https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/onnx/model_quantized.onnx",
            sizeBytes = 34_014_426L,
            sha256 = "6c9c6101a956d62dfb5e7190c538226c0c5bb9cb27b651234b6df063ee7dbfe4",
            dimension = 384,
            pooling = EmbeddingPooling.CLS,
            maxSequenceLength = 512,
            queryPrefix = "Represent this sentence for searching relevant passages: ",
            relevanceFloor = 0.50f,
            badge = "Best search quality",
            description = "More accurate retrieval for question answering. About 2× slower to index documents."
        ),
        EmbeddingModelSpec(
            id = MINILM_GGUF_ID,
            name = "all-MiniLM-L6-v2 (GGUF Q4_K_M)",
            backend = EmbeddingBackend.LLAMA_CPP,
            fileName = "all-MiniLM-L6-v2.Q4_K_M.gguf",
            downloadUrl = "https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.Q4_K_M.gguf",
            sizeBytes = 20_999_104L,
            sha256 = "53533e550397f2ba4e627bd5833d2c791097372d861a52022a8586282c2178cc",
            dimension = 384,
            pooling = EmbeddingPooling.MEAN,
            maxSequenceLength = 512,
            badge = "Compatibility",
            description = "Runs on the same engine as the language model. Use it if the default search model does not work on your phone."
        )
    )

    val default: EmbeddingModelSpec get() = list.first()

    fun byId(id: String?): EmbeddingModelSpec? = list.firstOrNull { it.id == id }
}
