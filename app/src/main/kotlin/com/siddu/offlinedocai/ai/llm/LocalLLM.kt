package com.siddu.offlinedocai.ai.llm

/**
 * Simple interface representing a language model capable of generating a response for a given prompt.
 */
interface LocalLLM {
    /**
     * Generate a response for [prompt]. The function runs on a background dispatcher and returns the
     * generated text, or throws an exception if generation fails.
     */
    suspend fun generate(prompt: String): String

    /**
     * Generate a response for [prompt] respecting the desired [answerLength].
     * Defaults to calling [generate(prompt)].
     */
    suspend fun generate(prompt: String, answerLength: AnswerLength): String = generate(prompt)
}

