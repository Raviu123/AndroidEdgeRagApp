package com.siddu.offlinedocai.domain.model

import org.json.JSONObject
import java.util.Locale

data class ResponseMetrics(
    val totalDurationMs: Long = 0L,
    val intent: String? = null,
    val intentRoutingMs: Long = 0L,
    val queryEmbeddingMs: Long = 0L,
    val searchMs: Long = 0L,
    val contextAssemblyMs: Long = 0L,
    val totalRetrievalMs: Long = 0L,
    val candidateChunksCount: Int = 0,
    val assembledTokens: Int = 0,
    val modelName: String? = null,
    val ttftMs: Long = 0L,
    val decodeMs: Long = 0L,
    val generatedTokens: Int = 0,
    val tokensPerSecond: Double = 0.0
) {
    fun toJson(): String {
        return JSONObject().apply {
            put("totalDurationMs", totalDurationMs)
            put("intent", intent)
            put("intentRoutingMs", intentRoutingMs)
            put("queryEmbeddingMs", queryEmbeddingMs)
            put("searchMs", searchMs)
            put("contextAssemblyMs", contextAssemblyMs)
            put("totalRetrievalMs", totalRetrievalMs)
            put("candidateChunksCount", candidateChunksCount)
            put("assembledTokens", assembledTokens)
            put("modelName", modelName)
            put("ttftMs", ttftMs)
            put("decodeMs", decodeMs)
            put("generatedTokens", generatedTokens)
            put("tokensPerSecond", tokensPerSecond)
        }.toString()
    }

    fun toMarkdownSummary(question: String, answer: String): String {
        val totalSec = String.format(Locale.US, "%.2fs", totalDurationMs / 1000.0)
        val tpsStr = String.format(Locale.US, "%.1f tok/s", tokensPerSecond)
        val sb = StringBuilder()
        sb.append("**User Query:**\n").append(question.trim()).append("\n\n")
        sb.append("**Response:**\n").append(answer.trim()).append("\n\n")
        sb.append("---\n")
        sb.append("### ⏱️ Performance & Latency Breakdown\n")
        sb.append("• **Total End-to-End Latency:** ").append(totalSec).append("\n")
        if (tokensPerSecond > 0.0) {
            sb.append("• **Generation Speed:** ").append(tpsStr)
                .append(" (").append(generatedTokens).append(" tokens in ").append(decodeMs).append(" ms)\n")
        }
        if (ttftMs > 0L) {
            sb.append("• **Time to First Token (TTFT):** ").append(ttftMs).append(" ms\n")
        }
        if (intent != null) {
            sb.append("• **Query Intent Routing:** ").append(intentRoutingMs).append(" ms (Intent: ").append(intent).append(")\n")
        }
        if (queryEmbeddingMs > 0L) {
            sb.append("• **Query Vector Embedding:** ").append(queryEmbeddingMs).append(" ms\n")
        }
        if (searchMs > 0L) {
            sb.append("• **Hybrid Search (Dense + BM25):** ").append(searchMs).append(" ms (Candidates: ").append(candidateChunksCount).append(")\n")
        }
        if (contextAssemblyMs > 0L || assembledTokens > 0) {
            sb.append("• **Context Assembly & Budget:** ").append(contextAssemblyMs).append(" ms (").append(assembledTokens).append(" tokens)\n")
        }
        if (!modelName.isNullOrBlank()) {
            sb.append("• **Active Model:** ").append(modelName).append("\n")
        }
        return sb.toString()
    }

    companion object {
        fun fromJson(raw: String?): ResponseMetrics? {
            if (raw.isNullOrBlank()) return null
            return runCatching {
                val json = JSONObject(raw)
                ResponseMetrics(
                    totalDurationMs = json.optLong("totalDurationMs", 0L),
                    intent = json.optString("intent").takeIf { it.isNotBlank() },
                    intentRoutingMs = json.optLong("intentRoutingMs", 0L),
                    queryEmbeddingMs = json.optLong("queryEmbeddingMs", 0L),
                    searchMs = json.optLong("searchMs", 0L),
                    contextAssemblyMs = json.optLong("contextAssemblyMs", 0L),
                    totalRetrievalMs = json.optLong("totalRetrievalMs", 0L),
                    candidateChunksCount = json.optInt("candidateChunksCount", 0),
                    assembledTokens = json.optInt("assembledTokens", 0),
                    modelName = json.optString("modelName").takeIf { it.isNotBlank() },
                    ttftMs = json.optLong("ttftMs", 0L),
                    decodeMs = json.optLong("decodeMs", 0L),
                    generatedTokens = json.optInt("generatedTokens", 0),
                    tokensPerSecond = json.optDouble("tokensPerSecond", 0.0)
                )
            }.getOrNull()
        }
    }
}
