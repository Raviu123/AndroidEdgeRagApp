package com.offlinedocai.ai.llm

/**
 * Answer length control mode for final answer synthesis in OfflineDocAI.
 *
 * IMPORTANT DESIGN PRINCIPLE:
 * Answer length MUST NOT affect retrieval accuracy.
 * Query preprocessing, intent detection, embeddings, lexical retrieval,
 * dense retrieval, RRF ranking, evidence selection, and citation validation
 * remain identical for both SHORT and LONG.
 */
enum class AnswerLength(val label: String) {
    DEFAULT("Default"),
    SHORT("Short"),
    LONG("Long");

    companion object {
        val DEFAULT_MODE = DEFAULT

        fun fromString(value: String?): AnswerLength {
            return when (value?.trim()?.uppercase()) {
                "LONG" -> LONG
                "SHORT" -> SHORT
                else -> DEFAULT
            }
        }
    }
}
