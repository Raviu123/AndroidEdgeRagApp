package com.siddu.offlinedocai.ai.telemetry

import android.os.SystemClock
import android.util.Log
import org.json.JSONObject

/**
 * Unified Telemetry and Benchmarking logger for OfflineDocAI research evaluations.
 * Emits structured JSON events tagged with [TAG] to logcat, which are captured by
 * the USB/ADB recording harness for publication performance figures.
 */
object BenchmarkTelemetry {
    const val TAG = "OFFLINE_DOC_AI_BENCHMARK"

    /**
     * Master toggle for on-device benchmarking telemetry.
     * Set to true when collecting research evaluation data.
     * Set to false for standard production usage (zero logging overhead).
     */
    var ENABLED: Boolean = false

    /**
     * Records end-to-end ingestion pipeline metrics.
     */
    fun logIngestion(
        documentId: Long,
        documentName: String,
        fileSizeBytes: Long,
        pageCount: Int,
        chunkCount: Int,
        embeddingModelId: String,
        extractionMs: Long,
        chunkingMs: Long,
        embeddingMs: Long,
        dbInsertMs: Long,
        totalIngestMs: Long
    ) {
        if (!ENABLED) return
        val json = JSONObject().apply {
            put("event_type", "DOCUMENT_INGESTION")
            put("timestamp_epoch_ms", System.currentTimeMillis())
            put("document_id", documentId)
            put("document_name", documentName)
            put("file_size_bytes", fileSizeBytes)
            put("page_count", pageCount)
            put("chunk_count", chunkCount)
            put("embedding_model_id", embeddingModelId)
            put("extraction_ms", extractionMs)
            put("chunking_ms", chunkingMs)
            put("embedding_ms", embeddingMs)
            put("db_insert_ms", dbInsertMs)
            put("total_ingest_ms", totalIngestMs)
            put("throughput_pages_per_sec", if (totalIngestMs > 0) (pageCount * 1000.0) / totalIngestMs else 0.0)
            put("throughput_chunks_per_sec", if (embeddingMs > 0) (chunkCount * 1000.0) / embeddingMs else 0.0)
        }
        Log.i(TAG, json.toString())
    }

    /**
     * Records end-to-end retrieval metrics.
     */
    fun logRetrieval(
        query: String,
        intent: String,
        documentIds: List<Long>,
        totalCorpusChunks: Int,
        topKCandidates: Int,
        filteredChunks: Int,
        embeddingModelId: String,
        intentRoutingMs: Long,
        queryEmbeddingMs: Long,
        denseSearchMs: Long,
        bm25SearchMs: Long,
        rrfFusionMs: Long,
        contextAssemblyMs: Long,
        totalRetrievalMs: Long,
        assembledTokens: Int,
        budgetTokens: Int
    ) {
        if (!ENABLED) return
        val json = JSONObject().apply {
            put("event_type", "RAG_RETRIEVAL")
            put("timestamp_epoch_ms", System.currentTimeMillis())
            put("query", query)
            put("intent", intent)
            put("document_ids", documentIds.joinToString(","))
            put("total_corpus_chunks", totalCorpusChunks)
            put("top_k_candidates", topKCandidates)
            put("filtered_chunks", filteredChunks)
            put("embedding_model_id", embeddingModelId)
            put("intent_routing_ms", intentRoutingMs)
            put("query_embedding_ms", queryEmbeddingMs)
            put("dense_search_ms", denseSearchMs)
            put("bm25_search_ms", bm25SearchMs)
            put("rrf_fusion_ms", rrfFusionMs)
            put("context_assembly_ms", contextAssemblyMs)
            put("total_retrieval_ms", totalRetrievalMs)
            put("assembled_tokens", assembledTokens)
            put("budget_tokens", budgetTokens)
        }
        Log.i(TAG, json.toString())
    }

    /**
     * Records LLM Generation and Inference metrics.
     */
    fun logGeneration(
        modelName: String,
        contextSize: Int,
        promptTokens: Int,
        generatedTokens: Int,
        ttftMs: Long,
        decodeMs: Long,
        totalGenerationMs: Long,
        tokensPerSecond: Double,
        isCancelled: Boolean = false
    ) {
        if (!ENABLED) return
        val json = JSONObject().apply {
            put("event_type", "LLM_GENERATION")
            put("timestamp_epoch_ms", System.currentTimeMillis())
            put("model_name", modelName)
            put("context_size", contextSize)
            put("prompt_tokens", promptTokens)
            put("generated_tokens", generatedTokens)
            put("ttft_ms", ttftMs)
            put("decode_ms", decodeMs)
            put("total_generation_ms", totalGenerationMs)
            put("tokens_per_second", tokensPerSecond)
            put("is_cancelled", isCancelled)
        }
        Log.i(TAG, json.toString())
    }
}
