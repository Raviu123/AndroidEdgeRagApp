package com.siddu.offlinedocai.ai.llm

import com.siddu.offlinedocai.ai.retrieval.SourceBlock

/** Prompt formats used when a model ships no chat template llama.cpp can apply. */
enum class TemplateFamily { CHATML, LLAMA3, GEMMA, PHI3, MISTRAL, DEEPSEEK }

/**
 * Builds model prompts from structured data and post-processes model output.
 *
 * The evidence and the question are passed as separate values and are never re-parsed from a
 * string, so text inside a document ("Question: ...", "Answer:", "Instruction:") cannot change what
 * the model is asked. Untrusted text is also stripped of chat-control markers before templating.
 */
object PromptBuilder {

    const val NOT_FOUND_ANSWER = "I couldn't find this in the selected document(s)."

    private fun lengthInstruction(length: AnswerLength): String = when (length) {
        AnswerLength.SHORT -> "Be brief: answer in one to three sentences."
        AnswerLength.DEFAULT -> "Answer thoroughly: explain the relevant details in one or two well-developed paragraphs, or use a list with a sentence of explanation per item when the question asks for several items."
        AnswerLength.LONG -> "Give a detailed, comprehensive answer that covers every relevant point and explains each one fully. Organise it with bullet points, numbered steps or short headings where they help."
    }

    fun groundedSystemPrompt(length: AnswerLength, isBudgetLimited: Boolean, isSubOneBillion: Boolean = false): String = buildString {
        if (isSubOneBillion) {
            append("You answer questions using only the sources provided by the user.\n")
            append("Instructions:\n")
            append("- Use facts from the sources. Do not guess.\n")
            append("- If the sources do not contain the answer, reply: \"$NOT_FOUND_ANSWER\"\n")
            append("- ").append(lengthInstruction(length)).append("\n")
            append("- Explain the answer clearly using complete thoughts. Do not reply with single-line or tautological summaries.")
        } else {
            append("You answer questions using only the numbered sources provided by the user.\n")
            append("Rules:\n")
            append("- Use only information stated in the sources. Do not use outside knowledge and do not guess.\n")
            append("- After each statement, cite the source it came from by copying its tag exactly, for example [Source 2, Page 7].\n")
            append("- If the sources do not contain the answer, reply exactly: \"$NOT_FOUND_ANSWER\"\n")
            append("- If the sources answer only part of the question, answer that part and say what is not covered.\n")
            append("- Copy numbers, names, dates, units and quotations exactly as they appear in the sources.\n")
            if (isBudgetLimited) append("- Not every relevant passage could be shown, so do not claim that a list or answer is complete.\n")
            append("- ").append(lengthInstruction(length))
        }
    }

    fun groundedMessages(
        blocks: List<SourceBlock>,
        question: String,
        length: AnswerLength,
        isBudgetLimited: Boolean = false,
        previousQuestion: String? = null,
        isSubOneBillion: Boolean = false
    ): List<ChatMessage> {
        val sources = blocks.joinToString("\n\n") { sanitizeUntrusted(it.format()) }
        val user = buildString {
            append("Sources:\n\n")
            append(sources.ifBlank { "(no sources)" })
            if (!previousQuestion.isNullOrBlank()) {
                append("\n\n(This question follows up on the earlier question: \"")
                append(sanitizeUntrusted(previousQuestion.trim().take(300)))
                append("\")")
            }
            append("\n\nQuestion: ")
            append(sanitizeUntrusted(question.trim()))
        }
        return listOf(
            ChatMessage("system", groundedSystemPrompt(length, isBudgetLimited, isSubOneBillion)),
            ChatMessage("user", user)
        )
    }

    fun chatMessages(userMessage: String, length: AnswerLength, previousTurn: ChatTurn? = null): List<ChatMessage> = buildList {
        add(
            ChatMessage(
                "system",
                "You are a helpful, honest assistant running fully offline on the user's device. " +
                    "If you are not sure about something, say so instead of guessing. ${lengthInstruction(length)}"
            )
        )
        if (previousTurn != null) {
            add(ChatMessage("user", sanitizeUntrusted(previousTurn.question.trim().take(1000))))
            add(ChatMessage("assistant", sanitizeUntrusted(previousTurn.answer.trim().take(1500))))
        }
        add(ChatMessage("user", sanitizeUntrusted(userMessage.trim())))
    }

    /**
     * DeepSeek-R1 distills follow instructions poorly from a system turn (their reference template has
     * none), so the system text is moved into the first user turn for that family.
     */
    fun foldSystemIntoUser(messages: List<ChatMessage>): List<ChatMessage> {
        val system = messages.filter { it.role == "system" }.joinToString("\n\n") { it.content }
        if (system.isEmpty()) return messages
        var folded = false
        return messages.filter { it.role != "system" }.map { m ->
            if (!folded && m.role == "user") {
                folded = true
                m.copy(content = "$system\n\n${m.content}")
            } else m
        }
    }

    /** Models that write a `<think>…</think>` reasoning block before answering. */
    fun isReasoningModel(chatTemplate: String?, nameHints: String): Boolean {
        val t = chatTemplate.orEmpty()
        val n = nameHints.lowercase()
        return "<think>" in t || "</think>" in t || Regex("""\b(r1|qwq|reason(ing|er)?|thinking)\b""").containsMatchIn(n.replace('-', ' ').replace('_', ' '))
    }

    /** True when [raw] ends inside an unfinished reasoning block. */
    fun isStillThinking(raw: String, promptOpensThinking: Boolean): Boolean {
        val open = raw.lastIndexOf("<think>", ignoreCase = true)
        val close = raw.lastIndexOf("</think>", ignoreCase = true)
        return if (open >= 0) close < open else promptOpensThinking && close < 0
    }

    /** Text to show while streaming: reasoning is hidden behind a short status line. */
    fun visibleAnswer(partial: String, promptOpensThinking: Boolean = false): String {
        if (isStillThinking(partial, promptOpensThinking)) return "_Thinking…_"
        return cleanOutput(partial, promptOpensThinking)
    }

    // ---------------------------------------------------------------------------------------------
    // Untrusted text
    // ---------------------------------------------------------------------------------------------

    private val CONTROL_TAGS = Regex("""<(/?)(start_of_turn|end_of_turn|s|bos|eos|think|tool_call|tool_response)>""", RegexOption.IGNORE_CASE)
    private val INST_TAGS = Regex("""\[(/?)(INST|SYS)]""", RegexOption.IGNORE_CASE)

    /**
     * Neutralises chat-control markers ("<|im_end|>", "<｜User｜>", "<start_of_turn>", "[INST]", ...)
     * inside document or user text so they are tokenized as plain text, not as control tokens.
     */
    fun sanitizeUntrusted(text: String): String = text
        .replace("<|", "‹|").replace("|>", "|›")
        .replace("<｜", "‹｜").replace("｜>", "｜›")
        .replace(CONTROL_TAGS) { m -> "‹${m.groupValues[1]}${m.groupValues[2]}›" }
        .replace(INST_TAGS) { m -> "⟦${m.groupValues[1]}${m.groupValues[2]}⟧" }

    // ---------------------------------------------------------------------------------------------
    // Fallback templates (only used when the GGUF has no template llama.cpp can apply)
    // ---------------------------------------------------------------------------------------------

    fun familyFor(chatTemplate: String?, architecture: String?): TemplateFamily {
        val t = chatTemplate.orEmpty()
        return when {
            "<|im_start|>" in t -> TemplateFamily.CHATML
            "<|start_header_id|>" in t -> TemplateFamily.LLAMA3
            "<start_of_turn>" in t -> TemplateFamily.GEMMA
            "<|assistant|>" in t || "<|end|>" in t -> TemplateFamily.PHI3
            "<｜Assistant｜>" in t -> TemplateFamily.DEEPSEEK
            "[INST]" in t -> TemplateFamily.MISTRAL
            else -> when (architecture?.lowercase()) {
                null -> TemplateFamily.CHATML
                "llama" -> TemplateFamily.LLAMA3
                "gemma", "gemma2", "gemma3" -> TemplateFamily.GEMMA
                "phi3" -> TemplateFamily.PHI3
                "mistral" -> TemplateFamily.MISTRAL
                else -> TemplateFamily.CHATML
            }
        }
    }

    /** BOS is never written here: the tokenizer adds it when the model expects one. */
    fun formatFallback(messages: List<ChatMessage>, family: TemplateFamily): String = buildString {
        val system = messages.filter { it.role == "system" }.joinToString("\n\n") { it.content }
        val turns = messages.filter { it.role != "system" }
        when (family) {
            TemplateFamily.CHATML -> {
                if (system.isNotEmpty()) append("<|im_start|>system\n$system<|im_end|>\n")
                for (m in turns) append("<|im_start|>${m.role}\n${m.content}<|im_end|>\n")
                append("<|im_start|>assistant\n")
            }
            TemplateFamily.LLAMA3 -> {
                if (system.isNotEmpty()) append("<|start_header_id|>system<|end_header_id|>\n\n$system<|eot_id|>")
                for (m in turns) append("<|start_header_id|>${m.role}<|end_header_id|>\n\n${m.content}<|eot_id|>")
                append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            }
            TemplateFamily.GEMMA -> {
                turns.forEachIndexed { i, m ->
                    val role = if (m.role == "assistant") "model" else "user"
                    val content = if (i == 0 && system.isNotEmpty()) "$system\n\n${m.content}" else m.content
                    append("<start_of_turn>$role\n$content<end_of_turn>\n")
                }
                append("<start_of_turn>model\n")
            }
            TemplateFamily.PHI3 -> {
                if (system.isNotEmpty()) append("<|system|>\n$system<|end|>\n")
                for (m in turns) append("<|${m.role}|>\n${m.content}<|end|>\n")
                append("<|assistant|>\n")
            }
            TemplateFamily.MISTRAL -> {
                turns.forEachIndexed { i, m ->
                    if (m.role == "assistant") append(" ${m.content}</s>")
                    else append("[INST] ${if (i == 0 && system.isNotEmpty()) "$system\n\n" else ""}${m.content} [/INST]")
                }
            }
            TemplateFamily.DEEPSEEK -> {
                append(system)
                for (m in turns) append(if (m.role == "assistant") "<｜Assistant｜>${m.content}<｜end▁of▁sentence｜>" else "<｜User｜>${m.content}")
                append("<｜Assistant｜>")
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Output post-processing
    // ---------------------------------------------------------------------------------------------

    private val THINK_BLOCK = Regex("""<think>.*?</think>""", setOf(RegexOption.DOT_MATCHES_ALL, RegexOption.IGNORE_CASE))
    private val LEFTOVER_TOKENS = Regex("""<\|[^|>\s]{1,40}\|>|<｜[^｜]{1,40}｜>|</?(?:start_of_turn|end_of_turn|s)>|\[/?INST]""")
    private val ANSWER_PREFIX = Regex("""^(?:assistant|answer)\s*:\s*""", RegexOption.IGNORE_CASE)

    /**
     * Removes reasoning blocks (DeepSeek-R1 style), stray template tokens and a leading "Answer:".
     * Returns an empty string if the model produced only (unfinished) reasoning.
     */
    fun cleanOutput(raw: String, promptOpensThinking: Boolean = false): String {
        // The template already opened "<think>": everything up to "</think>" is reasoning; with no "</think>" at all, nothing is an answer yet.
        if (promptOpensThinking && raw.indexOf("<think>", ignoreCase = true) < 0 && raw.indexOf("</think>", ignoreCase = true) < 0) return ""
        var text = raw.replace(THINK_BLOCK, "")
        val closeIndex = text.indexOf("</think>", ignoreCase = true)
        if (closeIndex >= 0) text = text.substring(closeIndex + "</think>".length) // template opened <think> in the prompt
        val openIndex = text.indexOf("<think>", ignoreCase = true)
        if (openIndex >= 0) text = text.substring(0, openIndex) // reasoning never finished
        text = text.replace(LEFTOVER_TOKENS, "").trim()
        return text.replace(ANSWER_PREFIX, "").trim()
    }

    private val CITATION = Regex("""\[\s*Sources?\s*:?\s*([^\[\]]{1,80})]""", RegexOption.IGNORE_CASE)
    private val PAGE_SPLIT = Regex("""(?i)\b(?:page|pages|pg|p)\b\.?""")
    private val NUMBER = Regex("""\d+""")

    /**
     * Rewrites every "[Source N, Page P]"-style citation to the canonical tag of source N (fixing
     * wrong page numbers) and removes citations to sources that were never provided.
     */
    fun sanitizeCitations(answer: String, blocks: List<SourceBlock>): String {
        val byNumber = blocks.associateBy { it.number }
        val rewritten = CITATION.replace(answer) { match ->
            val sourcePart = match.groupValues[1].split(PAGE_SPLIT, limit = 2)[0]
            val tags = NUMBER.findAll(sourcePart)
                .mapNotNull { it.value.toIntOrNull()?.let(byNumber::get)?.citationTag }
                .distinct()
                .toList()
            tags.joinToString(" ")
        }
        return rewritten
            .replace(Regex("""[ \t]+([.,;:!?])"""), "$1")
            .replace(Regex("""[ \t]{2,}"""), " ")
            .trim()
    }

    private val NOT_FOUND_PARAPHRASE = Regex(
        """(?i)^(?:i\s+(?:couldn'?t|could\s+not|can'?t|cannot|was\s+unable\s+to)\s+find|(?:the\s+)?(?:provided\s+)?(?:sources?|documents?|context|passages?)\s+(?:do(?:es)?\s+not|don'?t)\s+(?:contain|mention|include|provide|say|state|specify|cover)|there\s+is\s+no\s+(?:information|mention))"""
    )

    /** True for the exact not-found reply and for short paraphrases of it (no sources are attached then). */
    fun isNotFoundAnswer(answer: String): Boolean {
        val text = answer.replace(CITATION, "").trim().trim('"').trimEnd('.')
        if (text.equals(NOT_FOUND_ANSWER.trimEnd('.'), ignoreCase = true)) return true
        // A cited answer is a (partial) answer even if it starts by saying what is missing.
        return !CITATION.containsMatchIn(answer) && text.length <= 220 && !text.contains('\n') && NOT_FOUND_PARAPHRASE.containsMatchIn(text)
    }
}
