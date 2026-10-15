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
    /** Generation call start -> first token (prompt processing included). */
    val ttftMs: Long = 0L,
    val decodeMs: Long = 0L,
    val generatedTokens: Int = 0,
    val tokensPerSecond: Double = 0.0,
    // Prompt processing and device (0 / null = not recorded, e.g. older messages)
    val promptTokens: Int = 0,
    val reusedPromptTokens: Int = 0,
    val prefillMs: Long = 0L,
    val prefillTokensPerSecond: Double = 0.0,
    val threadsDecode: Int = 0,
    val threadsBatch: Int = 0,
    val cpuVariant: String? = null,
    val cpuTopology: String? = null,
    // Evidence budget
    val budgetTokens: Int = 0,
    val budgetContextFitTokens: Int = 0,
    val budgetSpeedCapTokens: Int = 0,
    val speedMode: String? = null,
    val thermal: String? = null
) {
    /** Label of the retrieval step: summaries and page questions do not search. */
    private val retrievalLabel: String
        get() = when (intent) {
            "SUMMARY" -> "Summary Sampling"
            "PAGE_LOOKUP", "PAGE_RANGE" -> "Page Fetch"
            else -> "Hybrid Search (Dense+BM25)"
        }

    /** Ordered (label, value) rows shared by the UI and the copied Markdown. */
    fun rows(): List<Pair<String, String>> = buildList {
        add("Total Turnaround" to String.format(Locale.US, "%.2fs", totalDurationMs / 1000.0))
        if (ttftMs > 0) add("Time-to-First-Token (TTFT)" to "$ttftMs ms")
        if (promptTokens > 0) {
            val newTokens = promptTokens - reusedPromptTokens
            val speed = if (prefillTokensPerSecond > 0) String.format(Locale.US, ", %.1f tok/s", prefillTokensPerSecond) else ""
            add("Prompt Processing" to "$prefillMs ms ($newTokens new tokens$speed)")
            add("Prompt Size" to "$promptTokens tokens ($reusedPromptTokens reused from cache)")
        }
        if (tokensPerSecond > 0) {
            add("Generation Speed" to String.format(Locale.US, "%.1f tok/s (%d tokens in %d ms)", tokensPerSecond, generatedTokens, decodeMs))
        }
        if (intent != null) add("Intent Routing" to "$intentRoutingMs ms ($intent)")
        if (queryEmbeddingMs > 0) add("Query Vector Embedding" to "$queryEmbeddingMs ms")
        if (searchMs > 0) add(retrievalLabel to "$searchMs ms ($candidateChunksCount chunks)")
        if (contextAssemblyMs > 0 || assembledTokens > 0) add("Context Assembly" to "$contextAssemblyMs ms ($assembledTokens tokens)")
        if (budgetTokens > 0) {
            val speedCap = if (budgetSpeedCapTokens > 0) "$budgetSpeedCapTokens" else "not measured yet"
            add("Evidence Budget" to "$budgetTokens tokens (fits $budgetContextFitTokens, speed cap $speedCap)")
        }
        if (threadsBatch > 0) add("CPU Threads (gen/prompt)" to "$threadsDecode / $threadsBatch")
        if (!cpuTopology.isNullOrBlank() || !cpuVariant.isNullOrBlank()) {
            add("CPU" to listOfNotNull(cpuTopology?.let { "cores $it" }, cpuVariant).joinToString(" • "))
        }
        if (!speedMode.isNullOrBlank()) add("Speed Mode" to listOfNotNull(speedMode, thermal?.takeIf { it != "NORMAL" }).joinToString(" • "))
        if (!modelName.isNullOrBlank()) add("Active Model" to modelName)
    }

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
            put("promptTokens", promptTokens)
            put("reusedPromptTokens", reusedPromptTokens)
            put("prefillMs", prefillMs)
            put("prefillTokensPerSecond", prefillTokensPerSecond)
            put("threadsDecode", threadsDecode)
            put("threadsBatch", threadsBatch)
            put("cpuVariant", cpuVariant)
            put("cpuTopology", cpuTopology)
            put("budgetTokens", budgetTokens)
            put("budgetContextFitTokens", budgetContextFitTokens)
            put("budgetSpeedCapTokens", budgetSpeedCapTokens)
            put("speedMode", speedMode)
            put("thermal", thermal)
        }.toString()
    }

    fun toMarkdownSummary(question: String, answer: String): String {
        val sb = StringBuilder()
        sb.append("**User Query:**\n").append(question.trim()).append("\n\n")
        sb.append("**Response:**\n").append(answer.trim()).append("\n\n")
        sb.append("---\n")
        sb.append("### ⏱️ Performance & Latency Breakdown\n")
        for ((label, value) in rows()) sb.append("• **").append(label).append(":** ").append(value).append("\n")
        return sb.toString()
    }

    companion object {
        fun fromJson(raw: String?): ResponseMetrics? {
            if (raw.isNullOrBlank()) return null
            return runCatching {
                val json = JSONObject(raw)
                fun str(key: String) = json.optString(key).takeIf { it.isNotBlank() && it != "null" }
                ResponseMetrics(
                    totalDurationMs = json.optLong("totalDurationMs", 0L),
                    intent = str("intent"),
                    intentRoutingMs = json.optLong("intentRoutingMs", 0L),
                    queryEmbeddingMs = json.optLong("queryEmbeddingMs", 0L),
                    searchMs = json.optLong("searchMs", 0L),
                    contextAssemblyMs = json.optLong("contextAssemblyMs", 0L),
                    totalRetrievalMs = json.optLong("totalRetrievalMs", 0L),
                    candidateChunksCount = json.optInt("candidateChunksCount", 0),
                    assembledTokens = json.optInt("assembledTokens", 0),
                    modelName = str("modelName"),
                    ttftMs = json.optLong("ttftMs", 0L),
                    decodeMs = json.optLong("decodeMs", 0L),
                    generatedTokens = json.optInt("generatedTokens", 0),
                    tokensPerSecond = json.optDouble("tokensPerSecond", 0.0),
                    promptTokens = json.optInt("promptTokens", 0),
                    reusedPromptTokens = json.optInt("reusedPromptTokens", 0),
                    prefillMs = json.optLong("prefillMs", 0L),
                    prefillTokensPerSecond = json.optDouble("prefillTokensPerSecond", 0.0),
                    threadsDecode = json.optInt("threadsDecode", 0),
                    threadsBatch = json.optInt("threadsBatch", 0),
                    cpuVariant = str("cpuVariant"),
                    cpuTopology = str("cpuTopology"),
                    budgetTokens = json.optInt("budgetTokens", 0),
                    budgetContextFitTokens = json.optInt("budgetContextFitTokens", 0),
                    budgetSpeedCapTokens = json.optInt("budgetSpeedCapTokens", 0),
                    speedMode = str("speedMode"),
                    thermal = str("thermal")
                )
            }.getOrNull()
        }
    }
}
