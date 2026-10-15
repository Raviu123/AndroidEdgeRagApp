package com.siddu.offlinedocai.ai.llm

import com.siddu.offlinedocai.ai.retrieval.AssembledContext

/** The request's token was cancelled (user stop or timeout); [partialAnswer] is what was generated so far. */
class GenerationCancelledException(val partialAnswer: String) : Exception("Generation cancelled")

/** One earlier exchange of the conversation, used to resolve follow-up questions. */
data class ChatTurn(val question: String, val answer: String)

/**
 * Per-request options. [token] cancels only this request; [onStarted] fires when the model starts
 * generating (after loading and prompt preparation); [onPartial] receives the answer text so far.
 */
class GenerationRequest(
    val token: GenerationToken = GenerationToken(),
    val onStarted: (() -> Unit)? = null,
    val onPartial: ((String) -> Unit)? = null,
    val previousTurn: ChatTurn? = null
)

/**
 * On-device language model used by the chat pipeline. Inputs are structured (evidence blocks and
 * question are separate values); implementations must never fabricate an answer when the model is
 * unavailable — they throw instead.
 */
interface LocalLLM {
    /** Answer [question] from the assembled evidence only. */
    suspend fun generateGrounded(
        context: AssembledContext,
        question: String,
        answerLength: AnswerLength,
        request: GenerationRequest = GenerationRequest()
    ): String

    /** Plain conversation, no documents. */
    suspend fun generateChat(message: String, answerLength: AnswerLength, request: GenerationRequest = GenerationRequest()): String

    /**
     * Tokens available for evidence once the system prompt, question, template overhead and the
     * answer allowance are reserved within the model's context window.
     */
    suspend fun availableEvidenceTokens(question: String, answerLength: AnswerLength): Int

    /** Model-token count of [text] (exact when the model is loaded, conservative estimate otherwise). */
    fun countTokens(text: String): Int

    /** How long generation may run (measured from [GenerationRequest.onStarted]) before it is stopped. */
    fun generationTimeoutMs(answerLength: AnswerLength): Long = 240_000L

    /** Stops whatever generation is currently running. Prefer cancelling a request's own token. */
    fun cancel() {}

    /** Metrics from the most recent inference run. */
    val lastInferenceMetrics: LlamaRunner.InferenceMetrics? get() = null

    /** How the most recent [availableEvidenceTokens] result was derived. */
    val lastBudgetInfo: EvidenceBudgetInfo? get() = null
}

/**
 * Evidence budget explanation: [contextFitTokens] is what fits in the context window; [speedCapTokens]
 * is what this device can process within [targetSeconds] at [prefillTokensPerSecond] (null = speed not
 * measured yet, so only the context window applied).
 */
data class EvidenceBudgetInfo(
    val contextFitTokens: Int,
    val speedCapTokens: Int?,
    val prefillTokensPerSecond: Double?,
    val targetSeconds: Double,
    val thermal: String,
    val speedMode: String
)
