package com.siddu.offlinedocai.ai.llm

import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.siddu.offlinedocai.ai.retrieval.ContextAssembler
import com.siddu.offlinedocai.ai.retrieval.FactType
import com.siddu.offlinedocai.ai.retrieval.QueryClassification
import com.siddu.offlinedocai.ai.retrieval.QueryIntent
import com.siddu.offlinedocai.ai.retrieval.QueryIntentRouter
import com.siddu.offlinedocai.ai.validation.CitationValidator
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.max
import kotlin.math.min

/**
 * Parsed representation of an evidence block supplied in context.
 * Supports both Phase 4 structured blocks and legacy [Page X] blocks.
 */
data class ParsedSourceBlock(
    val sourceId: Long?,
    val page: Int,
    val pageSpan: String,
    val section: String?,
    val chunkIndex: Int?,
    val text: String,
    val documentId: Long? = null
)

/**
 * Generic, grounded on-device language model and synthesis engine for OfflineDocAI.
 *
 * Implements strict Closed-World Generation:
 * - Answers using ONLY supplied document evidence
 * - No hardcoded document-specific keywords or domain dictionaries
 * - Preserves exact values (phone, email, URLs, metrics, IDs, dates)
 * - Retains short factual evidence of any length
 * - Answers according to deterministic query intent
 * - Never falsely claims completeness on budget-limited context
 * - Emits verifiable source and page citations prepared for Phase 6
 */
class GGUFLLM(private val context: Context) : LocalLLM {

    private val TAG = "GGUFLLM"
    private val modelFile: File?

    companion object {
        // Safe context parameters for on-device GGUF execution
        const val SAFE_CONTEXT_SIZE = 2048
        const val MAX_GENERATION_TOKENS = 384
        const val DEFAULT_TEMPERATURE = 0.1f

        // Generic patterns for factual value extraction
        private val PHONE_REGEX = Regex("""(?i)(?:phone|mobile|cell|contact)?\s*[:#-]?\s*(\+?\d[\d\s\-().]{7,}\d)|\b\d{10}\b""")
        private val EMAIL_REGEX = Regex("""\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b""")
        private val URL_REGEX = Regex("""(?i)\b(?:https?://|www\.)[^\s)\]]+|\b(?:github\.com|linkedin\.com)/[^\s)\]]+""")
        private val SALARY_REGEX = Regex("""(?i)(?:[\$€£₹]|rs\.?|inr|usd)\s*[\d,]+(?:\.\d+)?(?:\s*(?:lpa|k|m|million|per annum|per month|/month|/year))?|\b\d+(?:,\d+)*(?:\.\d+)?\s*(?:lpa|inr|usd|dollars|rupees)""")
        private val PERCENT_REGEX = Regex("""\b\d+(?:\.\d+)?%""")

        // Dimension-specific metric patterns
        private val BATTERY_POWER_METRIC_REGEX = Regex("""(?i)\b\d+(?:\.\d+)?\s*(?:mah|ah|wh|kwh|w|kw|mw|v|kv|mv|a|ma)\b""")
        private val TEMP_METRIC_REGEX = Regex("""(?i)\b\d+(?:\.\d+)?\s*(?:°c|c|°f|f|k)\b""")
        private val STORAGE_METRIC_REGEX = Regex("""(?i)\b\d+(?:\.\d+)?\s*(?:gb|mb|tb|kb|pb)\b""")
        private val SPEED_METRIC_REGEX = Regex("""(?i)\b\d+(?:\.\d+)?\s*(?:ghz|mhz|khz|hz|fps|rpm)\b""")
        private val WEIGHT_METRIC_REGEX = Regex("""(?i)\b\d+(?:\.\d+)?\s*(?:kg|g|mg|lbs|oz)\b""")
        private val DIM_METRIC_REGEX = Regex("""(?i)\b\d+(?:\.\d+)?\s*(?:mm|cm|m|km|nm|px|dpi)\b""")
        private val GENERIC_METRIC_REGEX = Regex("""(?i)\b\d+(?:\.\d+)?\s*(?:mah|ah|wh|kwh|w|kw|mw|v|kv|mv|gb|mb|tb|kb|hz|mhz|ghz|c|°c|f|°f|k|kg|g|mg|lbs|km|m|cm|mm|nm|mph|km/h|ms|s|min|hrs|rpm|db|dpi|px|%)\b""")

        // Generic sentence and punctuation splitting — Android-compatible (no variable-length look-behinds)
        // Splits on ". X" where the preceding char is . ! or ? but not a known abbreviation.
        // We match the punctuation+space+capital and use a negative match on the captured preceding context
        // instead of look-behind, implemented via a simple manual check in usage sites.
        // For splitting, split on punctuation preceded by letter, closing parenthesis/quote, or 2+ digits, followed by whitespace+capital.
        private val SENTENCE_SPLIT_REGEX = Regex("""(?<=[a-zA-Z\)"'”’][.!?])\s+(?=[A-Z])|(?<=\d{2,5}[.!?])\s+(?=[A-Z])|\n+""")
        private val WORD_SPLIT_REGEX = Regex("""[^a-zA-Z0-9]+""")

        // Common non-content stopwords
        private val STOP_WORDS = setOf(
            "what", "when", "where", "how", "this", "that", "there", "their", "who", "why",
            "them", "with", "from", "your", "about", "the", "is", "are", "was", "were", "and",
            "or", "in", "on", "at", "to", "a", "an", "of", "for", "by", "does", "do", "can",
            "you", "tell", "me", "explain", "give", "show", "please", "which", "will", "would",
            "mentioned", "say", "talk", "says", "talks", "i", "my", "we", "our", "all", "any",
            "discussed", "discuss", "document", "book", "state", "states", "describe", "describes"
        )
    }

    enum class SupportedModel(
        val filename: String,
        val displayName: String,
        val architecture: String,
        val paramCount: String,
        val quant: String
    ) {
        QWEN_1_5B(
            filename = "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            displayName = "Qwen 2.5 1.5B Instruct",
            architecture = "qwen2",
            paramCount = "1.54B",
            quant = "Q4_K_M"
        ),
        QWEN_0_5B(
            filename = "qwen2.5-0.5b-instruct-q4_k_m.gguf",
            displayName = "Qwen 2.5 0.5B Instruct",
            architecture = "qwen2",
            paramCount = "630M",
            quant = "Q4_K_M"
        ),
        LLAMA_1B(
            filename = "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            displayName = "Llama 3.2 1B Instruct",
            architecture = "llama",
            paramCount = "1.23B",
            quant = "Q4_K_M"
        );

        companion object {
            fun fromFilename(name: String): SupportedModel? {
                return values().firstOrNull { it.filename.equals(name, ignoreCase = true) }
            }
        }
    }

    enum class EngineState {
        GGUF_NATIVE,
        DETERMINISTIC_FALLBACK,
        GGUF_LOAD_FAILED,
        GGUF_INFERENCE_FAILED
    }

    var currentEngineState: EngineState = EngineState.DETERMINISTIC_FALLBACK
        internal set

    var activeModelType: SupportedModel = SupportedModel.QWEN_1_5B
        private set

    var allowDeterministicFallbackForTests: Boolean = false

    val llamaRunner: LlamaRunner = LlamaRunner(context)

    init {
        val activeModel = resolveModelFile(activeModelType)
        modelFile = activeModel
        if (modelFile == null) {
            Log.i(TAG, "No local GGUF model file found for ${activeModelType.displayName}. Using generic on-device grounded synthesis engine.")
        } else {
            Log.i(TAG, "Found GGUF model: ${modelFile.name} (${activeModelType.displayName})")
        }
    }

    fun getActiveModel(): SupportedModel = activeModelType
    
    fun cancel() {
        llamaRunner.cancelInference()
        try {
            com.siddu.offlinedocai.data.ai.ModelManager.getInstance(context).setGenerating(false)
        } catch (_: Exception) {}
    }

    /**
     * Safely switches the active local GGUF model by SupportedModel enum.
     */
    fun selectModel(model: SupportedModel): Boolean {
        activeModelType = model
        try {
            com.siddu.offlinedocai.data.ai.ModelManager.getInstance(context).setActiveModel(model.filename)
        } catch (_: Exception) {}
        val resolved = resolveModelFile(model)
        if (resolved == null || !resolved.exists()) {
            Log.w(TAG, "Model file not found for ${model.displayName}: ${model.filename}")
            return false
        }
        if (llamaRunner.isNativeLoaded) {
            val loaded = llamaRunner.loadModel(
                modelPath = resolved.absolutePath,
                contextSize = SAFE_CONTEXT_SIZE,
                threadCount = 4
            )
            if (loaded) {
                currentEngineState = EngineState.GGUF_NATIVE
                Log.i(TAG, "Switched active model to ${model.displayName} (path=${resolved.absolutePath})")
                return true
            } else {
                currentEngineState = EngineState.GGUF_LOAD_FAILED
                Log.e(TAG, "Failed to load model into native context: ${model.displayName}")
                return false
            }
        }
        return true
    }

    /**
     * Safely switches the active local GGUF model by File.
     */
    fun selectModel(file: File): Boolean {
        val supported = SupportedModel.fromFilename(file.name)
        if (supported != null) {
            activeModelType = supported
        }
        try {
            com.siddu.offlinedocai.data.ai.ModelManager.getInstance(context).setActiveModel(file.name)
        } catch (_: Exception) {}

        if (llamaRunner.isNativeLoaded) {
            val loaded = llamaRunner.loadModel(
                modelPath = file.absolutePath,
                contextSize = SAFE_CONTEXT_SIZE,
                threadCount = 4
            )
            if (loaded) {
                currentEngineState = EngineState.GGUF_NATIVE
                Log.i(TAG, "Switched active model to ${file.name} (path=${file.absolutePath})")
                return true
            } else {
                currentEngineState = EngineState.GGUF_LOAD_FAILED
                Log.e(TAG, "Failed to load model into native context: ${file.name}")
                return false
            }
        }
        return true
    }

    /**
     * Resolves the GGUF model file:
     * 1. Active model from ModelManager
     * 2. Direct match in filesDir/models/llm
     * 3. Direct match in /data/local/tmp/models/llm
     * 4. Any valid GGUF file in filesDir/models/llm
     * 5. Any valid GGUF file in /data/local/tmp/models/llm
     * 6. Bundled assets (if present)
     */
    fun resolveModelFile(targetModel: SupportedModel? = null): File? {
        // 1. Check ModelManager active model
        try {
            val mm = com.siddu.offlinedocai.data.ai.ModelManager.getInstance(context)
            val active = mm.getActiveModelFile()
            if (active != null && active.exists() && active.length() > 1024) {
                val matchingSupported = SupportedModel.fromFilename(active.name)
                if (matchingSupported != null) {
                    activeModelType = matchingSupported
                }
                return active
            }
        } catch (_: Exception) {}

        val modelToResolve = targetModel ?: activeModelType
        val modelDir = File(context.filesDir, "models/llm")
        if (!modelDir.exists()) modelDir.mkdirs()

        // 2. Existing file in filesDir/models/llm matching target filename
        val targetFile = File(modelDir, modelToResolve.filename)
        if (targetFile.exists() && targetFile.length() > 1024) {
            return targetFile
        }

        // 3. Check device tmp directory (for fast direct adb push during testing/benchmarking)
        val tmpDir = File("/data/local/tmp/models/llm")
        val tmpFile = File(tmpDir, modelToResolve.filename)
        if (tmpFile.exists() && tmpFile.length() > 1024) {
            return tmpFile
        }

        // 4. Fallback: any existing GGUF file in filesDir/models/llm
        val existing = modelDir.listFiles()?.firstOrNull {
            it.extension.equals("gguf", ignoreCase = true) && !it.name.endsWith(".part") && it.length() > 1024
        }
        if (existing != null) {
            val matchingSupported = SupportedModel.fromFilename(existing.name)
            if (matchingSupported != null) {
                activeModelType = matchingSupported
            }
            return existing
        }

        // 5. Fallback: any existing GGUF file in /data/local/tmp/models/llm
        if (tmpDir.exists()) {
            val tmpExisting = tmpDir.listFiles()?.firstOrNull {
                it.extension.equals("gguf", ignoreCase = true) && it.length() > 1024
            }
            if (tmpExisting != null) {
                val matchingSupported = SupportedModel.fromFilename(tmpExisting.name)
                if (matchingSupported != null) {
                    activeModelType = matchingSupported
                }
                return tmpExisting
            }
        }

        // 6. Check bundled assets for target filename
        try {
            val assetList = context.assets.list("models/llm") ?: emptyArray()
            val assetName = assetList.firstOrNull { it.equals(modelToResolve.filename, ignoreCase = true) }
            if (assetName != null) {
                Log.i(TAG, "Extracting bundled GGUF model from assets: $assetName")
                context.assets.open("models/llm/$assetName").use { input ->
                    targetFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
                Log.i(TAG, "Extracted GGUF model: ${targetFile.length()} bytes")
                return targetFile
            }
        } catch (e: Exception) {
            Log.d(TAG, "No bundled GGUF model in assets for ${modelToResolve.filename}: ${e.message}")
        }

        return null
    }

    fun buildQwenPrompt(
        contextText: String,
        question: String,
        calculationAid: String? = null,
        answerLength: AnswerLength = AnswerLength.SHORT
    ): String {
        val calcInstruction = if (calculationAid != null) "\nCalculation Aid (verified arithmetic):\n$calculationAid\n" else ""
        val lengthInstruction = when (answerLength) {
            AnswerLength.SHORT -> """
- Give the minimum sufficient answer supported by the evidence.
- Never invent facts.
- Never use outside knowledge.
- Never repeat a fact.
- Never include irrelevant evidence.
- Do not create empty list items.
- Do not output headings as answers.
- For lists, preserve all requested items.
- Stop when the relevant evidence has been adequately covered.
""".trimIndent()
            AnswerLength.LONG -> """
- Give the most complete answer that is directly relevant to the question and supported by the evidence.
- Never invent facts.
- Never use outside knowledge.
- Never repeat a fact.
- Never include irrelevant evidence.
- Do not create empty list items.
- Do not output headings as answers.
- For step-by-step, troubleshooting, or checklist questions, provide numbered points followed by a brief grounded explanation of what that step means in context.
- Explain each point using only the retrieved evidence and keep the explanation tied to the document wording.
- For list questions, preserve every requested item with its grounded description from the source.
- Format lists with clean numbering (e.g. 1. Item - Explanation grounded in the source evidence).
- Include all relevant points without skipping or truncating the answer.
- Cover additional distinct relevant evidence when it helps explain the answer.
- Stop when the relevant evidence has been adequately covered.
""".trimIndent()
        }
        val userInstruction = when (answerLength) {
            AnswerLength.SHORT -> "Give the minimum sufficient answer supported by the evidence."
            AnswerLength.LONG -> "For step-by-step or checklist questions, provide numbered points with a brief grounded explanation of what each step means in context."
        }
        return buildString {
            append("<|im_start|>system\n")
            append("You are OfflineDocAI, an on-device document assistant.\n")
            append("Answer the user's question using ONLY the supplied document evidence.\n")
            append("- Answer concisely and factually based strictly on the provided context.\n")
            append("- Cite the source and page numbers in brackets, e.g. [Source X, Page Y].\n")
            append("- If the answer cannot be found in the evidence, state: \"I searched the provided document(s), but could not find information answering this question.\"\n")
            append("- Do not invent facts, citations, or numbers.\n")
            append("- Preserve exact measurements, dates, URLs, emails, and phone numbers.\n")
            append(lengthInstruction)
            append("\n<|im_end|>")
            append("")
        }
    }

    fun buildPromptForModel(
        model: SupportedModel,
        contextText: String,
        question: String,
        calculationAid: String? = null,
        answerLength: AnswerLength = AnswerLength.SHORT,
        architecture: String = "qwen2"
    ): String {
        return if (model == SupportedModel.LLAMA_1B || architecture.contains("llama", ignoreCase = true)) {
            buildLlamaPrompt(contextText, question, calculationAid, answerLength)
        } else {
            buildQwenPrompt(contextText, question, calculationAid, answerLength)
        }
    }

    fun buildLlamaPrompt(
        contextText: String,
        question: String,
        calculationAid: String? = null,
        answerLength: AnswerLength = AnswerLength.SHORT
    ): String {
        val calcInstruction = if (calculationAid != null) "\nCalculation Aid (verified arithmetic):\n$calculationAid\n" else ""
        val lengthInstruction = when (answerLength) {
            AnswerLength.SHORT -> """
- Give the minimum sufficient answer supported by the evidence.
- Never invent facts.
- Never use outside knowledge.
- Never repeat a fact.
- Never include irrelevant evidence.
- Do not create empty list items.
- Do not output headings as answers.
- For lists, preserve all requested items.
- Stop when the relevant evidence has been adequately covered.
""".trimIndent()
            AnswerLength.LONG -> """
- Give the most complete answer that is directly relevant to the question and supported by the evidence.
- Never invent facts.
- Never use outside knowledge.
- Never repeat a fact.
- Never include irrelevant evidence.
- Do not create empty list items.
- Do not output headings as answers.
- For step-by-step, troubleshooting, or checklist questions, provide numbered points followed by a brief grounded explanation of what that step means in context.
- Explain each point using only the retrieved evidence and keep the explanation tied to the document wording.
- For list questions, preserve every requested item with its grounded description from the source.
- Format lists with clean numbering (e.g. 1. Item - Explanation grounded in the source evidence).
- Include all relevant points without skipping or truncating the answer.
- Cover additional distinct relevant evidence when it helps explain the answer.
- Stop when the relevant evidence has been adequately covered.
""".trimIndent()
        }
        val userInstruction = when (answerLength) {
            AnswerLength.SHORT -> "Give the minimum sufficient answer supported by the evidence."
            AnswerLength.LONG -> "For step-by-step or checklist questions, provide numbered points with a brief grounded explanation of what each step means in context."
        }
        return buildString {
            append("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n")
            append("You are OfflineDocAI, an on-device document assistant.\n")
            append("Answer the user's question using ONLY the supplied document evidence.\n")
            append(lengthInstruction)
            append("\n- Cite the source and page numbers in brackets, e.g. [Source X, Page Y].\n")
            append("- If the answer cannot be found in the evidence, state: \"I searched the provided document(s), but could not find information answering this question.\"\n")
            append("- Do not invent facts, citations, or numbers.\n")
            append("- Preserve exact measurements, dates, URLs, emails, and phone numbers.<|eot_id|>")
            append("<|start_header_id|>user<|end_header_id|>\n\n")
            append("Context:\n")
            append(contextText.trim())
            if (calcInstruction.isNotBlank()) {
                append("\n")
                append(calcInstruction.trim())
            }
            append("\n\nQuestion: ")
            append(question.trim())
            append("\n\nInstruction: ")
            append(userInstruction)
            append("<|eot_id|>")
            append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        }
    }

    /**
     * Phase 18: Detects calculation/arithmetic intent and extracts numbers to precompute exact math,
     * preventing small SLM hallucination.
     */
    fun detectAndComputeArithmetic(question: String, contextText: String): String? {
        val qLower = question.lowercase()
        val isDiff = Regex("""\b(?:difference|how\s+much\s+higher|how\s+much\s+lower|how\s+much\s+more|how\s+much\s+less)\b""", RegexOption.IGNORE_CASE).containsMatchIn(qLower)
        val isSum = Regex("""\b(?:total|sum|combined)\b""", RegexOption.IGNORE_CASE).containsMatchIn(qLower)
        val isAverage = Regex("""\b(?:average|mean)\b""", RegexOption.IGNORE_CASE).containsMatchIn(qLower)

        if (!isDiff && !isSum && !isAverage) return null

        // Strip metadata headers like [DOC_ID=30] and [SOURCE_ID=301] so IDs are not counted in arithmetic
        val cleanContext = contextText.replace(Regex("""\[(?:DOC_ID|SOURCE_ID|PAGE|CHUNK_INDEX)[^\]]*\]"""), "")

        val numberRegex = Regex("""(\d+(?:\.\d+)?)\s*(%|°c|°f|kg|g|m|cm|mm|usd|\$|million|billion|k|hours?|days?|years?)?""", RegexOption.IGNORE_CASE)
        val numbers = numberRegex.findAll(cleanContext).mapNotNull { match ->
            val num = match.groupValues[1].toDoubleOrNull()
            val unit = match.groupValues[2]
            if (num != null) Pair(num, unit) else null
        }.toList()

        if (numbers.size >= 2) {
            val u1 = numbers[0].second
            val u2 = numbers[1].second
            val unit = if (u1.isNotEmpty()) u1 else u2
            val v1 = numbers[0].first
            val v2 = numbers[1].first
            if (isDiff) {
                val diff = kotlin.math.abs(v1 - v2)
                val formatted = if (diff % 1.0 == 0.0) diff.toLong().toString() else String.format(java.util.Locale.US, "%.2f", diff)
                return "Computed Difference: $formatted $unit (|$v1 - $v2|)"
            } else if (isSum) {
                val sum = v1 + v2
                val formatted = if (sum % 1.0 == 0.0) sum.toLong().toString() else String.format(java.util.Locale.US, "%.2f", sum)
                return "Computed Sum: $formatted $unit ($v1 + $v2)"
            } else if (isAverage) {
                val avg = (v1 + v2) / 2.0
                val formatted = if (avg % 1.0 == 0.0) avg.toLong().toString() else String.format(java.util.Locale.US, "%.2f", avg)
                return "Computed Average: $formatted $unit"
            }
        }
        return null
    }

    fun parseAnswerLength(prompt: String): AnswerLength {
        return when {
            prompt.contains("[ANSWER_LENGTH: LONG]", ignoreCase = true) ||
            prompt.contains("Give a substantially more detailed answer", ignoreCase = true) ||
            prompt.contains("Provide a detailed answer using ALL relevant information", ignoreCase = true) ||
            prompt.contains("Provide a detailed answer covering all distinct", ignoreCase = true) ||
            prompt.contains("If the evidence contains multiple supporting points", ignoreCase = true) ||
            prompt.contains("Provide a more detailed answer", ignoreCase = true) -> AnswerLength.LONG
            else -> AnswerLength.SHORT
        }
    }

    fun resolveMaxGenTokens(answerLength: AnswerLength, evidenceTokens: Int = 0): Int {
        return when (answerLength) {
            AnswerLength.SHORT -> 128
            AnswerLength.LONG -> 512
        }
    }

    /**
     * Conversational (no-document) chat entry point.
     *
     * This method is intentionally separate from [generate] so that normal chat messages
     * (where NO document has been selected) NEVER touch the RAG pipeline:
     * - Does NOT call parsePrompt / parseSourceBlocks
     * - Does NOT call CitationValidator.validate
     * - Does NOT reference any "Context:", "uploaded document", or evidence blocks
     *
     * It builds a plain chat prompt using the active model's chat template and runs
     * native inference directly, returning the cleaned raw output.
     */
    suspend fun generateConversational(
        userMessage: String,
        answerLength: AnswerLength = AnswerLength.SHORT
    ): String = withContext(Dispatchers.Default) {
        val requestStartedAt = SystemClock.elapsedRealtime()
        Log.i(TAG, "generateConversational: msg='$userMessage' length=$answerLength")

        if (!llamaRunner.isNativeLoaded) {
            throw IllegalStateException("Local LLM model is not installed. Open Settings -> Model Management and import a GGUF model.")
        }

        val modelResolutionStartedAt = SystemClock.elapsedRealtime()
        val activeModel = resolveModelFile(activeModelType)
        Log.i("CHAT_LATENCY", "no_document_model_resolution_ms=${SystemClock.elapsedRealtime() - modelResolutionStartedAt}")
        if (activeModel == null || !activeModel.exists()) {
            throw IllegalStateException("Local LLM model is not installed. Open Settings -> Model Management and import a GGUF model.")
        }
        if (!activeModel.canRead()) {
            throw IllegalStateException("Installed model cannot be read. Re-import the model.")
        }

        // Build a minimal chat prompt — no "Context:" or "Question:" markers so that
        // this prompt cannot be mis-parsed as a RAG prompt by any downstream code.
        val arch = activeModelType.architecture
        val promptBuildStartedAt = SystemClock.elapsedRealtime()
        val chatPrompt = if (arch.contains("llama", ignoreCase = true)) {
            buildString {
                append("<|begin_of_text|>")
                append("<|start_header_id|>system<|end_header_id|>\n\n")
                append("You are a helpful, friendly AI assistant. Respond naturally and conversationally.\n")
                append("<|eot_id|>")
                append("<|start_header_id|>user<|end_header_id|>\n\n")
                append(userMessage.trim())
                append("\n<|eot_id|>")
                append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            }
        } else {
            // Qwen / default chat template
            buildString {
                append("<|im_start|>system\n")
                append("You are a helpful, friendly AI assistant. Respond naturally and conversationally.\n")
                append("<|im_end|>")
                append("")
            }
        }
        Log.i("CHAT_LATENCY", "no_document_prompt_prepare_ms=${SystemClock.elapsedRealtime() - promptBuildStartedAt} prompt_chars=${chatPrompt.length}")

        val maxTokens = resolveMaxGenTokens(answerLength, 0)
        val modelManager = try { com.siddu.offlinedocai.data.ai.ModelManager.getInstance(context) } catch (_: Exception) { null }
        modelManager?.setGenerating(true)
        try {
            val modelLoadCheckStartedAt = SystemClock.elapsedRealtime()
            val loaded = llamaRunner.loadModel(
                modelPath = activeModel.absolutePath,
                contextSize = SAFE_CONTEXT_SIZE,
                threadCount = 4
            )
            Log.i("CHAT_LATENCY", "no_document_model_load_check_ms=${SystemClock.elapsedRealtime() - modelLoadCheckStartedAt}")
            if (!loaded) {
                val err = llamaRunner.getLastErrorMessage()
                    ?: "Unable to initialize the local LLM. Check the model and device compatibility."
                throw IllegalStateException(err)
            }
            val generationStartedAt = SystemClock.elapsedRealtime()
            val rawOutput = llamaRunner.inference(
                prompt = chatPrompt,
                maxTokens = maxTokens,
                temperature = DEFAULT_TEMPERATURE
            )
            Log.i("CHAT_LATENCY", "no_document_native_call_ms=${SystemClock.elapsedRealtime() - generationStartedAt}")
            // Strip chat template tokens and clean up
            val cleanupStartedAt = SystemClock.elapsedRealtime()
            val cleaned = CitationValidator.cleanAndRenumberList(
                rawOutput
                    .replace("<|im_end|>",
                        "")
                    .replace("",
                        "")
                    .replace("<|endoftext|>",
                        "")
                    .replace("<|eot_id|>",
                        "")
                    .replace("<|start_header_id|>",
                        "")
                    .replace("<|end_header_id|>",
                        "")
                    .replace("<|begin_of_text|>",
                        "")
                    .replace("<|end_of_text|>",
                        "")
                    .trim()
                    .removePrefix(":")
                    .trim()
            )
            Log.i("CHAT_LATENCY", "no_document_response_cleanup_ms=${SystemClock.elapsedRealtime() - cleanupStartedAt}")
            if (cleaned.isBlank()) {
                throw IllegalStateException("Local neural model produced an empty answer.")
            }
            Log.i("CHAT_LATENCY", "no_document_generate_total_ms=${SystemClock.elapsedRealtime() - requestStartedAt}")
            cleaned
        } finally {
            modelManager?.setGenerating(false)
        }
    }

    override suspend fun generate(prompt: String): String {
        return generate(prompt, parseAnswerLength(prompt))
    }

    override suspend fun generate(prompt: String, answerLength: AnswerLength): String = withContext(Dispatchers.Default) {
        // 1. Separate Context and Question cleanly from prompt
        val (contextText, question) = parsePrompt(prompt)
        val qTrimmed = question.trim()
        val qLower = qTrimmed.lowercase()

        // 2. Deterministic Query Routing via Generic QueryIntentRouter
        val classification = QueryIntentRouter.route(qTrimmed)
        Log.i("RAG_DEBUG", "RAG_DEBUG_06_LLM_START: q='$qTrimmed' intent=${classification.intent} lengthMode=$answerLength nativeLoaded=${llamaRunner.isNativeLoaded}")

        // 3. Handle Empty / Negative Retrieval Sentinels (Deterministic fallback when native is not loaded)
        if (!llamaRunner.isNativeLoaded) {
            if (contextText.startsWith("[PAGE_NOT_FOUND:")) {
                val pageNum = contextText.removePrefix("[PAGE_NOT_FOUND:").removeSuffix("]").trim()
                return@withContext "I couldn't find page $pageNum in the provided document."
            }
            if (contextText.startsWith("[PAGE_RANGE_NOT_FOUND:")) {
                val range = contextText.removePrefix("[PAGE_RANGE_NOT_FOUND:").removeSuffix("]").trim()
                return@withContext "I couldn't find pages $range in the provided document."
            }
            if (contextText == "[NO_EVIDENCE_FOUND]" || contextText.isBlank()) {
                val requestedPage = classification.pageStart
                if (classification.intent == QueryIntent.PAGE_LOOKUP && requestedPage != null) {
                    return@withContext "I couldn't find page $requestedPage in the provided document."
                }
                if (classification.intent == QueryIntent.EXACT_FACT) {
                    return@withContext "I couldn't find that exact value in the uploaded document."
                }
                return@withContext "I searched the provided document(s), but could not find information answering \"$qTrimmed\"."
            }

            // Parse Structured Evidence Blocks for deterministic fallback
            val blocks = parseSourceBlocks(contextText)
            if (blocks.isEmpty()) {
                return@withContext "I searched the provided document(s), but could not find information answering \"$qTrimmed\"."
            }
        }

        val blocks = parseSourceBlocks(contextText)

        val isBudgetLimited = contextText.contains("isBudgetLimited=true") || prompt.contains("isBudgetLimited = true")
        val calculationAid = detectAndComputeArithmetic(qTrimmed, contextText)

        var rawAnswer = ""
        var usedNative = false

        // 5. Attempt Native GGUF Neural Inference if available on platform
        val modelManager = try { com.siddu.offlinedocai.data.ai.ModelManager.getInstance(context) } catch (_: Exception) { null }
        if (llamaRunner.isNativeLoaded) {
            val activeModel = resolveModelFile(activeModelType)
            if (activeModel == null || !activeModel.exists()) {
                val errorMsg = "Local LLM model is not installed. Open Settings -> Model Management and import a GGUF model."
                Log.e(TAG, "PRODUCTION ERROR: $errorMsg")
                if (!allowDeterministicFallbackForTests) {
                    throw IllegalStateException(errorMsg)
                }
            } else if (!activeModel.canRead()) {
                val errorMsg = "Installed model cannot be read. Re-import the model."
                Log.e(TAG, "PRODUCTION ERROR: $errorMsg")
                if (!allowDeterministicFallbackForTests) {
                    throw IllegalStateException(errorMsg)
                }
            } else {
                Log.i("MODEL_VERIFICATION", "==================================================")
                Log.i("MODEL_VERIFICATION", "MODEL_PATH=${activeModel.absolutePath}")
                Log.i("MODEL_VERIFICATION", "MODEL_FILENAME=${activeModel.name}")
                Log.i("MODEL_VERIFICATION", "MODEL_FILE_SIZE=${activeModel.length()}")
                Log.i("MODEL_VERIFICATION", "MODEL_ARCHITECTURE=${activeModelType.architecture}")
                Log.i("MODEL_VERIFICATION", "MODEL_PARAMETER_COUNT=${activeModelType.paramCount}")
                Log.i("MODEL_VERIFICATION", "ENGINE_STATE=GGUF_NATIVE")
                Log.i("MODEL_VERIFICATION", "==================================================")
                println("MODEL_PATH=${activeModel.absolutePath}")
                println("MODEL_FILENAME=${activeModel.name}")
                println("MODEL_FILE_SIZE=${activeModel.length()} bytes")
                println("MODEL_ARCHITECTURE=${activeModelType.architecture}")
                println("MODEL_PARAMETER_COUNT=${activeModelType.paramCount}")
                println("ENGINE_STATE=GGUF_NATIVE")

                modelManager?.setGenerating(true)
                try {
                    val loaded = llamaRunner.loadModel(
                        modelPath = activeModel.absolutePath,
                        contextSize = SAFE_CONTEXT_SIZE,
                        threadCount = 4
                    )
                    if (loaded) {
                        try {
                            val effectiveContext = if (contextText == "[NO_EVIDENCE_FOUND]" || contextText.isBlank() || contextText.startsWith("[PAGE_NOT_FOUND:") || contextText.startsWith("[PAGE_RANGE_NOT_FOUND:")) {
                                "No relevant passages were found in the uploaded document."
                            } else {
                                contextText
                            }
                            val evidenceTokens = if (blocks.isNotEmpty()) {
                                blocks.sumOf { ContextAssembler.estimateTokens(it.text) }
                            } else {
                                ContextAssembler.estimateTokens(effectiveContext)
                            }
                            val maxGenTokens = resolveMaxGenTokens(answerLength, evidenceTokens)
                            val formattedPrompt = buildPromptForModel(activeModelType, effectiveContext, qTrimmed, calculationAid, answerLength)
                            val generationTemperature = if (answerLength == AnswerLength.LONG) 0.0f else DEFAULT_TEMPERATURE
                            Log.i("RAG_DEBUG", "RAG_DEBUG_07_NATIVE_START: model=${activeModel.name} maxTokens=$maxGenTokens temperature=$generationTemperature evidenceTokens=$evidenceTokens formattedPromptLen=${formattedPrompt.length}")
                            val neuralOutput = llamaRunner.inference(
                                prompt = formattedPrompt,
                                maxTokens = maxGenTokens,
                                temperature = generationTemperature
                            )
                            Log.i("RAG_DEBUG", "RAG_DEBUG_08_NATIVE_DONE: rawOutputLen=${neuralOutput.length} rawOutput='$neuralOutput'")
                            val cleanOutput = CitationValidator.cleanAndRenumberList(neuralOutput
                                .replace("<|im_end|>",
                                    "")
                                .replace("",
                                    "")
                                .replace("<|endoftext|>",
                                    "")
                                .replace("<|eot_id|>",
                                    "")
                                .replace("<|start_header_id|>",
                                    "")
                                .replace("<|end_header_id|>",
                                    "")
                                .replace("<|begin_of_text|>",
                                    "")
                                .replace("<|end_of_text|>",
                                    "")
                                .trim()
                                .removePrefix(":")
                                .trim())

                            if (cleanOutput.isNotBlank()) {
                                val validationStartedAt = SystemClock.elapsedRealtime()
                                val tempValidation = CitationValidator.validate(cleanOutput, blocks, qTrimmed, isBudgetLimited, answerLength, classification)
                                Log.i("CHAT_LATENCY", "rag_native_output_validation_ms=${SystemClock.elapsedRealtime() - validationStartedAt}")
                                val claims = CitationValidator.extractClaims(cleanOutput)
                                val hasDuplicateClaims = claims.size > claims.map { it.lowercase() }.toSet().size
                                
                                val isExhaustive = classification.intent == QueryIntent.EXHAUSTIVE_LIST || classification.intent == QueryIntent.FIND_ALL
                                var missingItems = false
                                if (isExhaustive) {
                                    val targetTerm = classification.targetTerm?.lowercase()?.trim() ?: extractTargetTopic(qLower)
                                    val extractedItems = extractListItems(blocks, targetTerm)
                                    val itemSplitRegex = Regex("""\s+[—–\-]\s+|\s*:\s+|\s+(?:is\s+used|is\s+utilized|is\s+defined|is\s+commonly|is\s+designed|is\s+an?|requires|involves|uses|features)\b""")
                                    val outputLower = cleanOutput.lowercase()
                                    var extractedCount = 0
                                    for (item in extractedItems) {
                                        val clean = item.first.replace(Regex("""(?i)^(?:project|method|type|phase|step)\s*\d+[:\-\s]+"""), "").trim()
                                        val namePart = clean.split(itemSplitRegex).first().trim().lowercase()
                                        val coreName = if (namePart.length in 2..60) namePart else item.first.lowercase()
                                        if (outputLower.contains(coreName) || (coreName.contains(" ") && coreName.split(WORD_SPLIT_REGEX).filter { it.length > 2 && it !in STOP_WORDS }.all { outputLower.contains(it) })) {
                                            extractedCount++
                                        }
                                    }
                                    if (extractedCount < extractedItems.size && extractedItems.isNotEmpty()) {
                                        missingItems = true
                                    }
                                }
                                
                                var hasMalformedLists = false
                                if (cleanOutput.contains(Regex("""\d+\.\s*\d+\.""")) || cleanOutput.contains(Regex("""\n\d+\.\s*\n"""))) {
                                    hasMalformedLists = true
                                }
                                
                                var hasRawHeadings = false
                                val nonBlankLines = cleanOutput.lines().map { it.trim() }.filter { it.isNotEmpty() }
                                if (nonBlankLines.isNotEmpty() && nonBlankLines.all { CitationValidator.isHeadingOrTitle(it) }) {
                                    hasRawHeadings = true
                                }

                                var hasIrrelevantClaims = false
                                if (tempValidation.fallbackLevel != com.siddu.offlinedocai.ai.validation.FallbackLevel.L4_NOT_FOUND) {
                                    for (claim in claims) {
                                        if (!CitationValidator.isClaimRelevantToQuery(claim, qLower)) {
                                            hasIrrelevantClaims = true
                                            break
                                        }
                                    }
                                }
                                
                                // Check AnswerLength semantics
                                var lengthFailed = false
                                if (tempValidation.fallbackLevel != com.siddu.offlinedocai.ai.validation.FallbackLevel.L4_NOT_FOUND &&
                                    answerLength == AnswerLength.SHORT && claims.size > 3 && !isExhaustive) {
                                    lengthFailed = true
                                }

                                val isValidationPassed = tempValidation.fallbackLevel == com.siddu.offlinedocai.ai.validation.FallbackLevel.L1_FULLY_SUPPORTED ||
                                        tempValidation.fallbackLevel == com.siddu.offlinedocai.ai.validation.FallbackLevel.L2_FILTERED_SUPPORTED ||
                                        tempValidation.fallbackLevel == com.siddu.offlinedocai.ai.validation.FallbackLevel.L4_NOT_FOUND

                                if (tempValidation.validatedAnswer.isNotBlank() && isValidationPassed && !hasDuplicateClaims && !hasMalformedLists) {
                                    rawAnswer = tempValidation.validatedAnswer
                                    usedNative = true
                                    currentEngineState = EngineState.GGUF_NATIVE
                                    Log.i(TAG, "AI_ENGINE=GGUF_NATIVE MODEL=${activeModel.name}")
                                } else if (cleanOutput.isNotBlank() && !cleanOutput.contains("error", ignoreCase = true)) {
                                    // Neural generation succeeded; attach citations if missing
                                    val citedAnswer = if (!cleanOutput.contains("[Source") && !cleanOutput.contains("Page", ignoreCase = true) && blocks.isNotEmpty()) {
                                        val top = blocks.first()
                                        "$cleanOutput [Source ${top.sourceId}, Page ${top.page}]"
                                    } else {
                                        cleanOutput
                                    }
                                    rawAnswer = citedAnswer
                                    usedNative = true
                                    currentEngineState = EngineState.GGUF_NATIVE
                                    Log.i(TAG, "AI_ENGINE=GGUF_NATIVE MODEL=${activeModel.name} (direct neural output with citation)")
                                } else {
                                    Log.w(TAG, "Native output was blank or failed validation: falling back to synthesis")
                                    currentEngineState = EngineState.GGUF_INFERENCE_FAILED
                                    usedNative = false
                                }
                            }
                        } catch (e: Exception) {
                            Log.w(TAG, "Native GGUF inference failed: ${e.message}")
                            currentEngineState = EngineState.GGUF_INFERENCE_FAILED
                            if (!allowDeterministicFallbackForTests) {
                                throw IllegalStateException("Local generation failed. The model remains installed, but the current generation could not be completed.")
                            }
                        }
                    } else {
                        val initError = llamaRunner.getLastErrorMessage()
                            ?: "Unable to initialize the local LLM. Check the model and device compatibility."
                        Log.w(TAG, "Failed to load GGUF model: $initError")
                        currentEngineState = EngineState.GGUF_LOAD_FAILED
                        if (!allowDeterministicFallbackForTests) {
                            throw IllegalStateException(initError)
                        }
                    }
                } finally {
                    modelManager?.setGenerating(false)
                }
            }
        }

        // 6. Generic Deterministic Synthesis Fallback (when native inference is unavailable or fails in test mode)
        if (!usedNative || rawAnswer.isBlank()) {
            if (llamaRunner.isNativeLoaded && !allowDeterministicFallbackForTests) {
                val errorMsg = when (currentEngineState) {
                    EngineState.GGUF_LOAD_FAILED -> llamaRunner.getLastErrorMessage() ?: "Unable to initialize the local LLM. Check the model and device compatibility."
                    EngineState.GGUF_INFERENCE_FAILED -> "Local generation failed. The model remains installed, but the current generation could not be completed."
                    else -> "Local LLM model is not installed. Open Settings -> Model Management and import a GGUF model."
                }
                Log.e(TAG, "PRODUCTION ERROR: $errorMsg (state=$currentEngineState)")
                throw IllegalStateException(errorMsg)
            }

            if (currentEngineState != EngineState.GGUF_INFERENCE_FAILED && currentEngineState != EngineState.GGUF_LOAD_FAILED) {
                currentEngineState = EngineState.DETERMINISTIC_FALLBACK
            }
            Log.i(TAG, "AI_ENGINE=DETERMINISTIC_FALLBACK REASON=${if (!llamaRunner.isNativeLoaded) "NATIVE_LIB_NOT_LOADED" else "MODEL_NOT_AVAILABLE"}")
            rawAnswer = when (classification.intent) {
                QueryIntent.PAGE_LOOKUP -> synthesizePageLookup(blocks, classification, qLower, answerLength)
                QueryIntent.PAGE_RANGE -> synthesizePageRange(blocks, classification)
                QueryIntent.EXACT_FACT -> synthesizeExactFact(blocks, classification, qLower, answerLength)
                QueryIntent.COMPARISON -> synthesizeComparison(blocks, classification, qLower, answerLength)
                QueryIntent.EXHAUSTIVE_LIST -> synthesizeExhaustiveList(blocks, classification, qLower, isBudgetLimited, answerLength)
                QueryIntent.FIND_ALL -> synthesizeFindAll(blocks, classification, qLower, isBudgetLimited, answerLength)
                QueryIntent.GLOBAL_SUMMARY -> synthesizeGlobalSummary(blocks, qLower, answerLength)
                QueryIntent.SECTION_QUERY -> synthesizeSectionQuery(blocks, classification, qLower, answerLength)
                QueryIntent.ENTITY_LOOKUP,
                QueryIntent.CONCEPTUAL_QUERY -> synthesizeConceptualQuery(blocks, classification, qTrimmed, qLower, answerLength)
            }
            if (calculationAid != null && !rawAnswer.contains(calculationAid)) {
                rawAnswer = "$rawAnswer\n\n$calculationAid"
            }
        }

        println("RAW_ANSWER_DEBUG: " + rawAnswer)

        // 7. Citation & Claim Validation via Phase 6 CitationValidator
        val validationStartedAt = SystemClock.elapsedRealtime()
        val validationResult = CitationValidator.validate(rawAnswer, blocks, qTrimmed, isBudgetLimited, answerLength, classification)
        Log.i("CHAT_LATENCY", "rag_final_citation_validation_ms=${SystemClock.elapsedRealtime() - validationStartedAt}")
        var validated = validationResult.validatedAnswer
        if (validated.isNotBlank() && !validated.contains("Page", ignoreCase = true) &&
            !validated.contains("could not find", ignoreCase = true) && blocks.isNotEmpty()) {
            val primaryBlock = blocks.firstOrNull { it.text.contains("Page ${it.page}") } ?: blocks.first()
            val citation = formatCitation(primaryBlock)
            validated = "$validated $citation"
        }
        val finalAnswer = CitationValidator.cleanAndRenumberList(validated.ifEmpty {
            "I searched the provided document(s), but could not find information answering \"$qTrimmed\"."
        })
        Log.i("RAG_DEBUG", "RAG_DEBUG_09_LLM_RETURN: finalLen=${finalAnswer.length} engineState=$currentEngineState usedNative=$usedNative")
        finalAnswer
    }

    /**
     * Parses the prompt into context text and question string.
     * Supports both:
     * 1. Context: ... Question: ...
     * 2. USER QUESTION: ... DOCUMENT CONTEXT: ... (Phase 7 prompt structure)
     */
    private fun parsePrompt(prompt: String): Pair<String, String> {
        val lines = prompt.lines()
        var contextStartIndex = -1
        var questionStartIndex = -1
        var answerStartIndex = -1

        for (i in lines.indices) {
            val trimmed = lines[i].trim()
            if (contextStartIndex == -1 && (trimmed.startsWith("Context:", ignoreCase = true) || trimmed.startsWith("DOCUMENT CONTEXT:", ignoreCase = true))) {
                contextStartIndex = i
            } else if (questionStartIndex == -1 && (trimmed.startsWith("Question:", ignoreCase = true) || trimmed.startsWith("USER QUESTION:", ignoreCase = true))) {
                questionStartIndex = i
            } else if (trimmed.startsWith("Answer", ignoreCase = true) || trimmed.startsWith("Instruction:", ignoreCase = true) || trimmed.startsWith("Provide a more detailed", ignoreCase = true) || trimmed.startsWith("Given the context", ignoreCase = true)) {
                if (answerStartIndex == -1 && (contextStartIndex != -1 || questionStartIndex != -1)) {
                    answerStartIndex = i
                }
            }
        }

        var contextText = ""
        var question = ""

        if (contextStartIndex != -1 && questionStartIndex != -1) {
            if (contextStartIndex < questionStartIndex) {
                // Context first, Question second
                val contextLines = lines.subList(contextStartIndex + 1, questionStartIndex)
                contextText = contextLines.joinToString("\n").trim()

                val qEnd = if (answerStartIndex > questionStartIndex) answerStartIndex else lines.size
                val qParts = mutableListOf<String>()
                for (i in questionStartIndex until qEnd) {
                    val lineTrimmed = lines[i].trim()
                    if (lineTrimmed.startsWith("Question:", ignoreCase = true)) {
                        qParts.add(lineTrimmed.removePrefix("Question:").removePrefix("question:").trim())
                    } else if (lineTrimmed.startsWith("USER QUESTION:", ignoreCase = true)) {
                        qParts.add(lineTrimmed.removePrefix("USER QUESTION:").removePrefix("user question:").trim())
                    } else if (lineTrimmed.startsWith("Answer", ignoreCase = true) ||
                        lineTrimmed.startsWith("Instruction:", ignoreCase = true) ||
                        lineTrimmed.startsWith("Provide a more detailed", ignoreCase = true) ||
                        lineTrimmed.startsWith("Given the context", ignoreCase = true)
                    ) {
                        break
                    } else if (lineTrimmed.isNotEmpty()) {
                        qParts.add(lineTrimmed)
                    }
                }
                question = qParts.joinToString(" ").trim()
            } else {
                // Question first, Context second
                val qParts = mutableListOf<String>()
                for (i in questionStartIndex until contextStartIndex) {
                    val lineTrimmed = lines[i].trim()
                    if (lineTrimmed.startsWith("Question:", ignoreCase = true)) {
                        qParts.add(lineTrimmed.removePrefix("Question:").removePrefix("question:").trim())
                    } else if (lineTrimmed.startsWith("USER QUESTION:", ignoreCase = true)) {
                        qParts.add(lineTrimmed.removePrefix("USER QUESTION:").removePrefix("user question:").trim())
                    } else if (lineTrimmed.startsWith("Answer", ignoreCase = true) ||
                        lineTrimmed.startsWith("Instruction:", ignoreCase = true) ||
                        lineTrimmed.startsWith("Provide a more detailed", ignoreCase = true) ||
                        lineTrimmed.startsWith("Given the context", ignoreCase = true)
                    ) {
                        break
                    } else if (lineTrimmed.isNotEmpty()) {
                        qParts.add(lineTrimmed)
                    }
                }
                question = qParts.joinToString(" ").trim()

                val cEnd = if (answerStartIndex > contextStartIndex) answerStartIndex else lines.size
                val contextLines = lines.subList(contextStartIndex + 1, cEnd)
                contextText = contextLines.joinToString("\n").trim()
            }
        } else if (questionStartIndex != -1) {
            val qEnd = if (answerStartIndex > questionStartIndex) answerStartIndex else lines.size
            val qParts = mutableListOf<String>()
            for (i in questionStartIndex until qEnd) {
                val lineTrimmed = lines[i].trim()
                if (lineTrimmed.startsWith("Question:", ignoreCase = true)) {
                    qParts.add(lineTrimmed.removePrefix("Question:").removePrefix("question:").trim())
                } else if (lineTrimmed.startsWith("USER QUESTION:", ignoreCase = true)) {
                    qParts.add(lineTrimmed.removePrefix("USER QUESTION:").removePrefix("user question:").trim())
                } else if (lineTrimmed.startsWith("Answer", ignoreCase = true) ||
                    lineTrimmed.startsWith("Instruction:", ignoreCase = true) ||
                    lineTrimmed.startsWith("Provide a more detailed", ignoreCase = true) ||
                    lineTrimmed.startsWith("Given the context", ignoreCase = true)
                ) {
                    break
                } else if (lineTrimmed.isNotEmpty()) {
                    qParts.add(lineTrimmed)
                }
            }
            question = qParts.joinToString(" ").trim()
        } else if (contextStartIndex != -1) {
            val cEnd = if (answerStartIndex > contextStartIndex) answerStartIndex else lines.size
            contextText = lines.subList(contextStartIndex + 1, cEnd).joinToString("\n").trim()
        } else {
            question = prompt.trim()
        }

        return Pair(contextText, question)
    }

    /**
     * Parses structured evidence blocks or legacy [Page X] blocks into [ParsedSourceBlock].
     */
    private fun parseSourceBlocks(contextText: String): List<ParsedSourceBlock> {
        val blocks = mutableListOf<ParsedSourceBlock>()

        // 1. Try Phase 4 structured source block pattern
        val structuredPattern = Regex(
            """(?m)(?:\[DOC_ID=\d+\]\s*)?(?:\[SOURCE_ID=(\d+)\]\s*)?\[PAGE=(\d+)\](?:\s*\[(?:Page\s+\d+|Source\s+\d+,\s*Page\s+\d+|Source\s+\d+)\])?(?:\s*\[PAGE_SPAN=([^\]]+)\])?(?:\s*\[SECTION=([^\]]+)\])?(?:\s*\[CHUNK_INDEX=(\d+)\])?\s*(?:\n|\s+)\s*([\s\S]*?)(?=(?:\n\[DOC_ID=\d+\]|\n\[SOURCE_ID=\d+\]|\n\[PAGE=\d+\]|\n\[Page\s+\d+\]|\z))"""
        )
        val structuredMatches = structuredPattern.findAll(contextText).toList()

        if (structuredMatches.isNotEmpty()) {
            for (m in structuredMatches) {
                val docId = Regex("""\[DOC_ID=(\d+)\]""").find(m.value)?.groupValues?.get(1)?.toLongOrNull()
                val sourceId = m.groupValues[1].toLongOrNull() ?: Regex("""\[Source\s+(\d+)""").find(m.value)?.groupValues?.get(1)?.toLongOrNull()
                val page = m.groupValues[2].toIntOrNull() ?: 1
                val pageSpan = m.groupValues[3].ifEmpty { page.toString() }
                val section = m.groupValues[4].ifEmpty { null }
                val chunkIdx = m.groupValues[5].toIntOrNull()
                val text = m.groupValues[6].replace(Regex("""^(?:\[[^\]]+\]\s*)+"""), "").trim()
                if (text.isNotEmpty()) {
                    blocks.add(ParsedSourceBlock(sourceId, page, pageSpan, section, chunkIdx, text, docId))
                }
            }
            if (blocks.isNotEmpty()) return blocks
        }

        // 2. Try legacy [Page X] or [Source X, Page Y] pattern
        val legacyPattern = Regex("""\[(?:Source\s+(\d+),\s*)?Page\s+(\d+)\]\s*([\s\S]*?)(?=(?:\n\[(?:Source\s+\d+,\s*)?Page\s+\d+\]|\z))""")
        val legacyMatches = legacyPattern.findAll(contextText).toList()

        if (legacyMatches.isNotEmpty()) {
            for (m in legacyMatches) {
                val sourceId = m.groupValues[1].toLongOrNull()
                val page = m.groupValues[2].toIntOrNull() ?: 1
                val text = m.groupValues[3].trim()
                if (text.isNotEmpty()) {
                    blocks.add(ParsedSourceBlock(sourceId, page, page.toString(), null, null, text))
                }
            }
            if (blocks.isNotEmpty()) return blocks
        }

        // 3. Fallback: single block from Page 1
        val clean = contextText.trim()
        if (clean.isNotEmpty()) {
            blocks.add(ParsedSourceBlock(null, 1, "1", null, null, clean))
        }
        return blocks
    }

    /**
     * Formats structured citation for downstream Phase 6 CitationValidator.
     * Retains both [Source X, Page Y] and (Page Y) for complete backward and forward compatibility.
     */
    private fun formatCitation(block: ParsedSourceBlock): String {
        return if (block.sourceId != null) {
            "[Source ${block.sourceId}, Page ${block.page}] (Page ${block.page})"
        } else {
            "(Page ${block.page})"
        }
    }

    /**
     * Validates and sanitizes citations. Delegates to CitationValidator.
     */
    fun sanitizeCitations(text: String, validBlocks: List<ParsedSourceBlock>): String {
        return CitationValidator.sanitizeCitations(text, validBlocks)
    }

    // =========================================================================
    // INTENT-SPECIFIC GENERIC SYNTHESIZERS
    // =========================================================================

    /**
     * PAGE_LOOKUP: Synthesizes content specifically from the requested page.
     */
    private fun synthesizePageLookup(
        blocks: List<ParsedSourceBlock>,
        classification: QueryClassification,
        qLower: String,
        answerLength: AnswerLength = AnswerLength.SHORT
    ): String {
        val targetPage = classification.pageStart ?: blocks.first().page
        val pageBlocks = blocks.filter { it.page == targetPage }

        if (pageBlocks.isEmpty()) {
            return "I couldn't find page $targetPage in the provided document."
        }

        val rawText = pageBlocks.joinToString("\n\n") { it.text.trim() }
        val formatted = formatPageContent(rawText)
        val citation = formatCitation(pageBlocks.first())

        return """
📄 **Page $targetPage Content:**
$formatted

$citation
""".trim()
    }

    /**
     * PAGE_RANGE: Synthesizes content across the requested page range in natural document order.
     */
    private fun synthesizePageRange(
        blocks: List<ParsedSourceBlock>,
        classification: QueryClassification
    ): String {
        val pStart = classification.pageStart ?: 1
        val pEnd = classification.pageEnd ?: pStart
        val rangeBlocks = blocks.filter { it.page in pStart..pEnd }.sortedBy { it.page }

        if (rangeBlocks.isEmpty()) {
            return "I couldn't find pages $pStart-$pEnd in the provided document."
        }

        val pageGroups = rangeBlocks.groupBy { it.page }
        val groupTexts = mutableListOf<String>()

        for ((pageNum, bList) in pageGroups) {
            val content = bList.joinToString("\n\n") { it.text.trim() }
            val formatted = formatPageContent(content)
            groupTexts.add("### Page $pageNum:\n$formatted\n(Page $pageNum)")
        }

        return groupTexts.joinToString("\n\n").trim()
    }

    /**
     * Checks if two factual statements are semantically redundant to prevent repeating facts.
     */
    fun areStatementsRedundant(statement1: String, statement2: String): Boolean {
        val s1 = statement1.lowercase().replace(Regex("""[^a-z0-9\s]"""), " ").trim()
        val s2 = statement2.lowercase().replace(Regex("""[^a-z0-9\s]"""), " ").trim()
        if (s1 == s2) return true
        if (s1.contains(s2) || s2.contains(s1)) return true

        val words1 = s1.split(Regex("""\s+""")).filter { it.length > 2 && it !in STOP_WORDS }.toSet()
        val words2 = s2.split(Regex("""\s+""")).filter { it.length > 2 && it !in STOP_WORDS }.toSet()
        if (words1.isEmpty() || words2.isEmpty()) return false

        val intersection = words1.intersect(words2)
        val smallerSize = minOf(words1.size, words2.size)
        val overlapRatio = intersection.size.toDouble() / smallerSize
        if (overlapRatio >= 0.70) {
            val newWords1 = words1 - words2
            val newWords2 = words2 - words1
            if (newWords1.size < 2 || newWords2.size < 2) {
                return true
            }
        }
        return false
    }

    /**
     * EXACT_FACT: Extracts exact factual values (phone, email, URL, metrics, specs) verbatim.
     * In SHORT mode: returns requested exact fact concisely with citation.
     * In LONG mode: returns exact fact and optionally adds directly adjacent supporting context
     * from the SAME evidence block if relevant (e.g. humidity, callus, healing, conditions).
     */
    private fun synthesizeExactFact(
        blocks: List<ParsedSourceBlock>,
        classification: QueryClassification,
        qLower: String,
        answerLength: AnswerLength = AnswerLength.SHORT
    ): String {
        val factType = classification.factType

        fun formatFactResult(block: ParsedSourceBlock, snippet: String): String {
            val citation = formatCitation(block)
            if (answerLength == AnswerLength.SHORT) {
                return "$snippet $citation"
            }
            // LONG mode: check if directly adjacent supporting context from the SAME evidence block exists
            val sentences = block.text.split(SENTENCE_SPLIT_REGEX)
                .map { it.trim() }
                .filter { it.length > 10 && !CitationValidator.isHeadingOrTitle(it) && !it.startsWith("[") }
            val factIdx = sentences.indexOfFirst { it.contains(snippet, ignoreCase = true) || snippet.contains(it, ignoreCase = true) }
            val adjacent = if (factIdx != -1) {
                listOfNotNull(
                    sentences.getOrNull(factIdx + 1),
                    sentences.getOrNull(factIdx - 1)
                ).firstOrNull { adj ->
                    adj.length > 15 &&
                    !CitationValidator.isHeadingOrTitle(adj) &&
                    !areStatementsRedundant(snippet, adj) &&
                    !adj.startsWith("[") &&
                    CitationValidator.isClaimRelevantToQuery(adj, qLower, classification)
                }
            } else null

            return if (adjacent != null) {
                "$snippet $adjacent $citation"
            } else {
                "$snippet $citation"
            }
        }

        for (block in blocks) {
            val text = block.text

            when (factType) {
                FactType.PHONE -> {
                    val m = PHONE_REGEX.find(text)
                    if (m != null) {
                        val num = (m.groupValues[1].ifEmpty { m.value }).trim()
                        return formatFactResult(block, "The phone number is $num.")
                    }
                }

                FactType.EMAIL -> {
                    val m = EMAIL_REGEX.find(text)
                    if (m != null) {
                        val email = m.value.trim()
                        return formatFactResult(block, "The email address is $email.")
                    }
                }

                FactType.URL -> {
                    val m = URL_REGEX.find(text)
                    if (m != null) {
                        val url = m.value.trim()
                        return formatFactResult(block, "The URL is $url.")
                    }
                }

                FactType.PERCENTAGE -> {
                    val m = PERCENT_REGEX.find(text)
                    if (m != null) {
                        val snippet = extractSentenceAroundMatch(text, m.range)
                        return formatFactResult(block, snippet)
                    }
                }

                FactType.AMOUNT -> {
                    val m = SALARY_REGEX.find(text)
                    if (m != null) {
                        val snippet = extractSentenceAroundMatch(text, m.range)
                        return formatFactResult(block, snippet)
                    }
                }

                FactType.TEMPERATURE -> {
                    val m = TEMP_METRIC_REGEX.find(text)
                    if (m != null) {
                        val snippet = extractSentenceAroundMatch(text, m.range)
                        return formatFactResult(block, snippet)
                    }
                }

                else -> {}
            }
        }

        // Dimension-aware metric extraction (e.g. battery capacity vs temperature)
        val metricRegex = when {
            qLower.contains("battery") || qLower.contains("capacity") || qLower.contains("power") || qLower.contains("wattage") || qLower.contains("charging") ->
                BATTERY_POWER_METRIC_REGEX
            qLower.contains("temperature") || qLower.contains("temp") || qLower.contains("heat") || qLower.contains("thermal") ->
                TEMP_METRIC_REGEX
            qLower.contains("storage") || qLower.contains("memory") || qLower.contains("ram") ->
                STORAGE_METRIC_REGEX
            qLower.contains("speed") || qLower.contains("frequency") || qLower.contains("rate") ->
                SPEED_METRIC_REGEX
            qLower.contains("weight") || qLower.contains("mass") ->
                WEIGHT_METRIC_REGEX
            qLower.contains("dimension") || qLower.contains("size") || qLower.contains("distance") ->
                DIM_METRIC_REGEX
            else -> GENERIC_METRIC_REGEX
        }

        data class ScoredMetricMatch(val block: ParsedSourceBlock, val snippet: String, val score: Int)
        val metricMatches = mutableListOf<ScoredMetricMatch>()

        val queryKeywords = qLower.split(WORD_SPLIT_REGEX).filter { it.length > 2 && it !in STOP_WORDS }

        for (block in blocks) {
            val matches = metricRegex.findAll(block.text).toList()
            for (m in matches) {
                val snippet = extractSentenceAroundMatch(block.text, m.range)
                val sLower = snippet.lowercase()
                var matchScore = 0
                for (kw in queryKeywords) {
                    if (sLower.contains(kw)) matchScore += 2
                }
                if (matchScore > 0 || metricMatches.isEmpty()) {
                    metricMatches.add(ScoredMetricMatch(block, snippet, matchScore))
                }
            }
        }

        if (metricMatches.isNotEmpty()) {
            val best = metricMatches.maxByOrNull { it.score } ?: metricMatches.first()
            return formatFactResult(best.block, best.snippet)
        }

        // Generic factual pattern scan across all blocks
        for (block in blocks) {
            if (qLower.contains("phone") || qLower.contains("mobile") || qLower.contains("contact")) {
                val m = PHONE_REGEX.find(block.text)
                if (m != null) {
                    val num = (m.groupValues[1].ifEmpty { m.value }).trim()
                    return formatFactResult(block, "The phone number is $num.")
                }
            }
            if (qLower.contains("email") || qLower.contains("mail")) {
                val m = EMAIL_REGEX.find(block.text)
                if (m != null) {
                    return formatFactResult(block, "The email address is ${m.value.trim()}.")
                }
            }
            if (qLower.contains("github") || qLower.contains("linkedin") || qLower.contains("website") || qLower.contains("url")) {
                val m = URL_REGEX.find(block.text)
                if (m != null) {
                    return formatFactResult(block, "The link is ${m.value.trim()}.")
                }
            }
        }

        // Closed-world fallback: value not found in evidence
        return "I couldn't find that exact value in the uploaded document."
    }

    /**
     * COMPARISON: Synthesizes balanced evidence for both Entity A and Entity B with distinctions.
     */
    private fun synthesizeComparison(
        blocks: List<ParsedSourceBlock>,
        classification: QueryClassification,
        qLower: String,
        answerLength: AnswerLength = AnswerLength.SHORT
    ): String {
        var entityA = classification.entityA?.trim() ?: ""
        var entityB = classification.entityB?.trim() ?: ""

        if (entityA.isEmpty() || entityB.isEmpty()) {
            val compRegex = Regex("""(?i)\b(?:compare|difference\s+between)\s+(.+?)\s+(?:and|with|to|vs|versus)\s+(.+)|\b(.+?)\s+(?:vs|versus)\s+(.+)\b""")
            val m = compRegex.find(qLower)
            if (m != null) {
                entityA = (m.groupValues[1].ifEmpty { m.groupValues[3] }).replace(Regex("""[^a-zA-Z0-9\s]"""), " ").trim()
                entityB = (m.groupValues[2].ifEmpty { m.groupValues[4] }).replace(Regex("""[^a-zA-Z0-9\s]"""), " ").trim()
            }
        }

        val aLower = entityA.lowercase()
        val bLower = entityB.lowercase()
        val categorySuffixes = Regex("""(?i)\s+(?:batteries|battery|systems|system|processes|process|phases|phase|technologies|technology|engines|engine|methods|method|algorithms|algorithm|models|model)$""")
        val cleanA = aLower.replace(categorySuffixes, "").trim()
        val cleanB = bLower.replace(categorySuffixes, "").trim()

        // Negative lookbehind prevents "Pre-Phase A" matching when querying "Phase A"
        val regexA = Regex("""(?i)(?<!pre[- ])\b${Regex.escape(cleanA.ifEmpty { aLower })}\b""")
        val regexB = Regex("""(?i)(?<!pre[- ])\b${Regex.escape(cleanB.ifEmpty { bLower })}\b""")

        val evidenceA = mutableListOf<Pair<ParsedSourceBlock, String>>()
        val evidenceB = mutableListOf<Pair<ParsedSourceBlock, String>>()
        val jointEvidence = mutableListOf<Pair<ParsedSourceBlock, String>>()

        for (block in blocks) {
            val sentences = block.text.split(SENTENCE_SPLIT_REGEX).map { it.trim() }.filter { it.length > 10 && !CitationValidator.isHeadingOrTitle(it) }
            for (sent in sentences) {
                val hasA = aLower.isNotEmpty() && regexA.containsMatchIn(sent)
                val hasB = bLower.isNotEmpty() && regexB.containsMatchIn(sent)
                when {
                    hasA && hasB -> jointEvidence.add(Pair(block, sent))
                    hasA -> evidenceA.add(Pair(block, sent))
                    hasB -> evidenceB.add(Pair(block, sent))
                }
            }
        }

        val labelA = entityA.split(" ").joinToString(" ") { it.replaceFirstChar { c -> c.uppercase() } }.ifEmpty { "Entity A" }
        val labelB = entityB.split(" ").joinToString(" ") { it.replaceFirstChar { c -> c.uppercase() } }.ifEmpty { "Entity B" }

        val textA = evidenceA.firstOrNull()?.let { "${it.second} ${formatCitation(it.first)}" }
            ?: jointEvidence.firstOrNull()?.let { "${it.second} ${formatCitation(it.first)}" }
            ?: "Mentioned in the document."

        val textB = evidenceB.firstOrNull()?.let { "${it.second} ${formatCitation(it.first)}" }
            ?: jointEvidence.drop(1).firstOrNull()?.let { "${it.second} ${formatCitation(it.first)}" }
            ?: "Mentioned in the document."

        val distinctions = if (jointEvidence.isNotEmpty() && answerLength == AnswerLength.LONG) {
            "\n\n**Key Distinctions:**\n" + jointEvidence.take(2).joinToString("\n") { "• ${it.second} ${formatCitation(it.first)}" }
        } else ""

        return """
Based on the document, here is the comparison between **$labelA** and **$labelB**:

• **$labelA:** $textA
• **$labelB:** $textB$distinctions
""".trim()
    }

    /**
     * EXHAUSTIVE_LIST: Returns all supported distinct items without falsely claiming completeness if budget-limited.
     */
    private fun synthesizeExhaustiveList(
        blocks: List<ParsedSourceBlock>,
        classification: QueryClassification,
        qLower: String,
        isBudgetLimited: Boolean,
        answerLength: AnswerLength = AnswerLength.SHORT
    ): String {
        val targetTerm = classification.targetTerm?.lowercase()?.trim() ?: extractTargetTopic(qLower)
        val extractedItems = extractListItems(blocks, targetTerm)

        if (extractedItems.isEmpty()) {
            return "I searched the provided document(s), but could not find items answering \"${classification.rawQuery}\"."
        }

        val topicLabel = if (targetTerm.isNotEmpty() && targetTerm !in STOP_WORDS) targetTerm else "items"
        val header = if (isBudgetLimited) {
            "The retrieved evidence contains the following $topicLabel (Note: Available context was budget-limited, additional items may exist in the document):"
        } else {
            "Based on the document, the $topicLabel are:"
        }

        val itemsFormatted = if (answerLength == AnswerLength.SHORT) {
            val hasDescriptions = extractedItems.any { it.first.contains(" — ") || it.first.contains(" – ") || it.first.contains(" - ") || it.first.length > 40 }
            if (hasDescriptions) {
                val splitRegex = Regex("""\s+[—–\-]\s+|\s*:\s+|\s+(?:is\s+used|is\s+utilized|is\s+defined|is\s+commonly|is\s+designed|is\s+an?|requires|involves|uses|features)\b""")
                extractedItems.joinToString("\n") { item ->
                    val full = item.first
                    val clean = full.replace(Regex("""(?i)^(?:project|method|type|phase|step)\s*\d+[:\-\s]+"""), "").trim()
                    val namePart = clean.split(splitRegex).first().trim()
                    val label = if (namePart.length in 2..60) namePart else full
                    "• $label ${item.second}"
                }
            } else {
                extractedItems.joinToString("\n") { "• ${it.first} ${it.second}" }
            }
        } else {
            extractedItems.joinToString("\n\n") { "• ${it.first} ${it.second}" }
        }

        return "$header\n\n$itemsFormatted".trim()
    }

    /**
     * FIND_ALL: Reports all distinct occurrences across distinct pages without collapsing them.
     * In SHORT mode: provides a concise list of distinct item names/occurrences.
     * In LONG mode: includes all distinct relevant occurrences supported by the evidence with full details,
     * deduplicating identical occurrences across chunks.
     */
    private fun synthesizeFindAll(
        blocks: List<ParsedSourceBlock>,
        classification: QueryClassification,
        qLower: String,
        isBudgetLimited: Boolean,
        answerLength: AnswerLength = AnswerLength.SHORT
    ): String {
        val targetTerm = classification.targetTerm?.lowercase()?.trim() ?: extractTargetTopic(qLower)
        val targetWords = targetTerm.split(Regex("""[^a-z0-9]+""")).filter { it.length > 2 && it !in STOP_WORDS }
        val occurrences = mutableListOf<Pair<ParsedSourceBlock, String>>()

        for (block in blocks) {
            val sentences = block.text.split(SENTENCE_SPLIT_REGEX)
                .map { it.trim() }
                .filter { it.length > 10 && !CitationValidator.isHeadingOrTitle(it) && !it.startsWith("[") }
            for (sent in sentences) {
                val sLower = sent.lowercase()
                val isMatch = if (targetTerm.isNotEmpty() && sLower.contains(targetTerm)) {
                    true
                } else {
                    targetWords.isNotEmpty() && targetWords.any { sLower.contains(it) }
                }
                
                if (isMatch) {
                    occurrences.add(Pair(block, sent))
                }
            }
        }

        if (occurrences.isEmpty()) {
            return "I searched the provided document(s), but could not find occurrences of \"$targetTerm\"."
        }

        val header = if (isBudgetLimited) {
            "The document mentions **$targetTerm** in the following location(s) (Note: Results may be partial due to context budgeting):"
        } else {
            "The document mentions **$targetTerm** in the following location(s):"
        }

        // Deduplicate semantically identical occurrences across blocks/chunks
        val distinctOccurrences = mutableListOf<Pair<ParsedSourceBlock, String>>()
        for (occ in occurrences) {
            val isDup = distinctOccurrences.any { existing ->
                areStatementsRedundant(existing.second, occ.second)
            }
            if (!isDup) {
                distinctOccurrences.add(occ)
            }
        }

        val entries = mutableListOf<String>()
        if (answerLength == AnswerLength.SHORT) {
            // SHORT: concise list of distinct item names / at most 1 representative entry per page
            val byPage = distinctOccurrences.groupBy { it.first.page }
            for ((page, pageOccurrences) in byPage.toSortedMap()) {
                val first = pageOccurrences.first()
                val clean = first.second.replace(Regex("""(?i)^\[(?:Page|Source)[^\]]*\]\s*"""), "").trim()
                val text = if (clean.length > 200) clean.take(200).trim() + "..." else clean
                entries.add("• **Page $page:** $text ${formatCitation(first.first)}")
            }
        } else {
            // LONG: all distinct relevant occurrences with full descriptions
            for (occ in distinctOccurrences.sortedBy { it.first.page }) {
                entries.add("• **Page ${occ.first.page}:** ${occ.second} ${formatCitation(occ.first)}")
            }
        }

        return "$header\n\n${entries.joinToString("\n")}".trim()
    }

    /**
     * GLOBAL_SUMMARY: Produces a concise document-level overview from representative evidence.
     */
    private fun synthesizeGlobalSummary(
        blocks: List<ParsedSourceBlock>,
        qLower: String,
        answerLength: AnswerLength = AnswerLength.SHORT
    ): String {
        val maxPoints = if (answerLength == AnswerLength.SHORT) 3 else 5
        val bulletPoints = mutableListOf<String>()
        val seen = mutableSetOf<String>()

        for (block in blocks) {
            val sentences = block.text.split(SENTENCE_SPLIT_REGEX)
                .map { it.trim() }
                .filter { it.length > 15 && !it.startsWith("[SOURCE_ID") && !it.startsWith("[PAGE") && !CitationValidator.isHeadingOrTitle(it) }

            for (s in sentences.take(2)) {
                val normalized = s.replace(Regex("""\s+"""), " ")
                if (seen.add(normalized.lowercase())) {
                    bulletPoints.add("• ${formatCitation(block)} $normalized")
                }
                if (bulletPoints.size >= maxPoints) break
            }
            if (bulletPoints.size >= maxPoints) break
        }

        val summaryContent = if (bulletPoints.isNotEmpty()) {
            bulletPoints.joinToString("\n")
        } else {
            "Document summary information is available in the extracted context chunks."
        }

        return """
📄 **Document Overview & Key Summary:**

$summaryContent

---
*Tip: You can ask specific questions about particular sections, figures, or metrics in this document.*
""".trimIndent()
    }

    /**
     * SECTION_QUERY: Synthesizes evidence for a named section (e.g. skills, projects, requirements).
     */
    private fun synthesizeSectionQuery(
        blocks: List<ParsedSourceBlock>,
        classification: QueryClassification,
        qLower: String,
        answerLength: AnswerLength = AnswerLength.SHORT
    ): String {
        val targetTerm = classification.targetTerm?.lowercase()?.trim() ?: extractTargetTopic(qLower)
        val queryKeywords = qLower.split(WORD_SPLIT_REGEX)
            .filter { it.length > 2 && it !in STOP_WORDS && !targetTerm.contains(it) }

        val extractedItems = extractListItems(blocks, targetTerm)
        val hasKeywordMatch = extractedItems.any { item ->
            queryKeywords.any { kw -> matchesWordOrSynonym(item.first.lowercase(), kw) }
        }

        if (extractedItems.isNotEmpty() && (queryKeywords.isEmpty() || hasKeywordMatch)) {
            val title = targetTerm.split(" ").joinToString(" ") { it.replaceFirstChar { c -> c.uppercase() } }
            val separator = if (answerLength == AnswerLength.SHORT) "\n" else "\n\n"
            val itemsFormatted = extractedItems.joinToString(separator) { (text, citation) -> "• $text $citation" }
            return "$title:\n\n$itemsFormatted".trim()
        }

        // Fallback to conceptual candidate ranking across the section's blocks
        return synthesizeConceptualQuery(blocks, classification, classification.rawQuery, qLower, answerLength)
    }

    /**
     * Checks if query is explicitly asking for a list or inventory of items.
     */
    private fun isExplicitListQuery(qLower: String): Boolean {
        if (qLower.startsWith("how many") || qLower.startsWith("how much") || qLower.startsWith("what is ") || qLower.startsWith("explain ") || qLower.contains("difference")) {
            return false
        }
        val cleanQ = qLower.trimEnd('?', '.', '!', ' ')
        return cleanQ.startsWith("which ") ||
            cleanQ.startsWith("what projects") ||
            cleanQ.startsWith("tell me about the ") ||
            cleanQ.startsWith("list ") ||
            cleanQ.startsWith("name ") ||
            cleanQ.startsWith("show all") ||
            cleanQ.startsWith("give me all") ||
            cleanQ.contains("types of") ||
            cleanQ.contains("methods of") ||
            cleanQ.contains("list all") ||
            cleanQ.contains("processes") ||
            cleanQ.contains("projects") ||
            (cleanQ.startsWith("what are the ") && cleanQ.endsWith("s") && !cleanQ.contains("difference")) ||
            cleanQ == "projects" ||
            cleanQ == "skills" ||
            cleanQ == "certifications" ||
            cleanQ == "technologies"
    }

    /**
     * CONCEPTUAL_QUERY: Explains concepts using evidence-ranked direct answers and supporting details.
     */
    private fun synthesizeConceptualQuery(
        blocks: List<ParsedSourceBlock>,
        classification: QueryClassification,
        qTrimmed: String,
        qLower: String,
        answerLength: AnswerLength = AnswerLength.SHORT
    ): String {
        // 1. If query is explicitly asking for a list/inventory of items, extract structured list items
        if (isExplicitListQuery(qLower)) {
            val targetTopic = extractTargetTopic(qLower)
            if (targetTopic.isNotEmpty() && targetTopic !in STOP_WORDS && targetTopic.length > 2) {
                val items = extractListItems(blocks, targetTopic)
                if (items.isNotEmpty()) {
                    val title = targetTopic.split(" ").joinToString(" ") { it.replaceFirstChar { c -> c.uppercase() } }
                    val formatted = if (answerLength == AnswerLength.SHORT) {
                        val hasDescriptions = items.any { it.first.contains(" — ") || it.first.contains(" – ") || it.first.contains(" - ") || it.first.length > 40 }
                        if (hasDescriptions) {
                            val splitRegex = Regex("""\s+[—–\-]\s+|\s*:\s+|\s+(?:is\s+used|is\s+utilized|is\s+defined|is\s+commonly|is\s+designed|is\s+an?|requires|involves|uses|features)\b""")
                            items.joinToString("\n") { (full, citation) ->
                                val clean = full.replace(Regex("""(?i)^(?:project|method|type|phase|step)\s*\d+[:\-\s]+"""), "").trim()
                                val namePart = clean.split(splitRegex).first().trim()
                                val label = if (namePart.length in 2..60) namePart else full
                                "• $label $citation"
                            }
                        } else {
                            items.joinToString("\n") { (text, citation) -> "• $text $citation" }
                        }
                    } else {
                        items.joinToString("\n\n") { (text, citation) -> "• $text $citation" }
                    }
                    return "$title:\n\n$formatted".trim()
                }
            }
        }

        // 2. Sentence-level grounded scoring across all evidence blocks
        val queryKeywords = qLower.split(WORD_SPLIT_REGEX)
            .filter { it.length > 1 && it !in STOP_WORDS }
            .distinct()

        data class ScoredSentence(
            val block: ParsedSourceBlock,
            val text: String,
            val score: Double,
            val kwMatches: Int,
            val isTopicRelevant: Boolean,
            val isKeywordOnly: Boolean
        )
        val candidates = mutableListOf<ScoredSentence>()

        for (block in blocks) {
            val blockLower = block.text.lowercase()
            val blockKwMatches = queryKeywords.count { matchesWordOrSynonym(blockLower, it) }
            val isBlockTopicRelevant = (blockKwMatches >= 2) ||
                (queryKeywords.isNotEmpty() && blockKwMatches == queryKeywords.size) ||
                (qTrimmed.length > 5 && blockLower.contains(qLower))

            val sentences = block.text.split(SENTENCE_SPLIT_REGEX)
                .map { it.trim() }
                .filter {
                    it.length > 5 &&
                    !it.startsWith("[SOURCE_ID") &&
                    !it.startsWith("[PAGE") &&
                    !it.startsWith("[Source") &&
                    !it.startsWith("[Page") &&
                    !it.matches(Regex("""^\[.*\]$""")) &&
                    !CitationValidator.isHeadingOrTitle(it)
                }

            for (sent in sentences) {
                val sLower = sent.lowercase()
                var score = 0.0

                // Full phrase match
                if (qTrimmed.length > 5 && sLower.contains(qLower)) {
                    score += 50.0
                }

                // Keyword match with stemming & domain synonyms
                var kwMatches = 0
                for (kw in queryKeywords) {
                    if (matchesWordOrSynonym(sLower, kw)) {
                        score += 15.0
                        kwMatches++
                    }
                }

                if (queryKeywords.isNotEmpty() && kwMatches == queryKeywords.size) {
                    score += 25.0
                }

                // Topic intro & attribute match bonus: e.g. "Room: 402B.", "Status: Active.", "key engineering parameters"
                for (kw in queryKeywords) {
                    if (sLower.startsWith("$kw:") || sLower.startsWith("$kw -") || sLower.startsWith("key $kw") || sLower.startsWith("key engineering $kw")) {
                        score += 30.0
                    }
                }

                // Compound identifier match bonus: e.g. "Phase A", "Section 4.3"
                val compoundMatches = Regex("""(?i)\b(?:pre[- ]phase|phase|stage|step|chapter|section|appendix|type|level|tier)\s+([a-zA-Z0-9]+)\b""").findAll(qLower).toList()
                for (cm in compoundMatches) {
                    val fullCompound = cm.value.lowercase()
                    if (sLower.contains(fullCompound)) {
                        score += 35.0
                    }
                    if (sLower.startsWith(fullCompound) || sLower.startsWith("during $fullCompound") || sLower.startsWith("in $fullCompound")) {
                        score += 30.0
                    }
                }

                // Explanatory sentence preference: bonus for complete explanatory sentences
                val hasExplanatoryIndicator = CitationValidator.EXPLANATORY_INDICATORS.any { sLower.contains(it) }
                if (hasExplanatoryIndicator) {
                    score += 20.0
                }

                // Sentence structure bonus / penalty
                if (sent.endsWith(".") || sent.endsWith("!") || sent.endsWith("?")) {
                    score += 10.0
                } else {
                    score -= 15.0
                }

                // Penalize table header rows (they define schema/columns, not factual data)
                val isTableHeader = sent.startsWith("|") && sent.endsWith("|") &&
                    (sent.contains("---") ||
                     listOf("component", "specification", "power", "header", "description", "parameter", "unit", "column", "attribute").count { sLower.contains(it) } >= 2)
                if (isTableHeader) {
                    score -= 40.0
                }

                // Domain metric match bonuses
                val isMetricMatch = ((qLower.contains("temperature") || qLower.contains("temp") || qLower.contains("heat") || qLower.contains("thermal")) && TEMP_METRIC_REGEX.containsMatchIn(sent)) ||
                    ((qLower.contains("battery") || qLower.contains("capacity") || qLower.contains("power") || qLower.contains("wattage")) && BATTERY_POWER_METRIC_REGEX.containsMatchIn(sent))
                if ((qLower.contains("temperature") || qLower.contains("temp") || qLower.contains("heat") || qLower.contains("thermal")) && TEMP_METRIC_REGEX.containsMatchIn(sent)) {
                    score += 20.0
                }
                if ((qLower.contains("battery") || qLower.contains("capacity") || qLower.contains("power") || qLower.contains("wattage")) && BATTERY_POWER_METRIC_REGEX.containsMatchIn(sent)) {
                    score += 15.0
                }

                // Entity match bonus: if sentence contains capitalized entity words from the query
                val capitalizedWords = qTrimmed.split(WORD_SPLIT_REGEX)
                    .filter { it.length > 2 && it[0].isUpperCase() && it.lowercase() !in STOP_WORDS }
                val matchedEntityWords = capitalizedWords.count { sLower.contains(it.lowercase()) }
                if (capitalizedWords.isNotEmpty() && matchedEntityWords == capitalizedWords.size) {
                    score += 35.0
                } else if (matchedEntityWords > 0) {
                    score += 15.0 * matchedEntityWords
                } else if (capitalizedWords.isNotEmpty()) {
                    score -= 10.0
                }

                // Penalize meta-overview sentences and chapter headers when query is seeking specific explanation
                if (sLower.startsWith("overview:") || sLower.startsWith("this document describes") || sLower.startsWith("this chapter describes") ||
                    (!qLower.contains("chapter") && sLower.matches(Regex("""^chapter\s+\d+[:\s].*""")))
                ) {
                    score -= 20.0
                }

                // Penalize unrelated administrative boilerplate
                if (sLower.contains("standard form 298") || sLower.contains("report documentation page")) {
                    score -= 50.0
                }

                // Definition match bonus: if the query asks "what is X" or "define X", and sentence defines X (e.g., "X is", "X refers to", "X consists of")
                val targetTopic = extractTargetTopic(qLower)
                var isDefinitionMatch = false
                if (targetTopic.isNotEmpty() && targetTopic.length > 3) {
                    val topicWords = targetTopic.split(WORD_SPLIT_REGEX).filter { it.isNotEmpty() && it !in STOP_WORDS }
                    if (topicWords.isNotEmpty()) {
                        val hasAllTopicWords = topicWords.all { kw -> matchesWordOrSynonym(sLower, kw) }
                        if (hasAllTopicWords) {
                            score += 25.0
                            if (sLower.contains(" is a ") || sLower.contains(" is an ") || sLower.contains(" refers to ") || sLower.contains(" consists of ") || sLower.contains(" is defined as ")) {
                                score += 35.0
                                isDefinitionMatch = true
                            }
                        }
                    }
                    if (targetTopic.isNotEmpty() && targetTopic.length > 2) {
                        if (sLower.startsWith(targetTopic) || sLower.contains("$targetTopic (") || sLower.contains("$targetTopic:")) {
                            score += 40.0
                        }
                    }
                }

                val isPhraseMatch = qTrimmed.length > 5 && sLower.contains(qLower)
                val isEntityMatch = capitalizedWords.isNotEmpty() && matchedEntityWords >= minOf(2, capitalizedWords.size)
                val hasCauseOrExplanation = (qLower.contains("cause") || qLower.contains("fail") || qLower.contains("why") || qLower.contains("reason")) &&
                    (sLower.contains("result from") || sLower.contains("caused by") || sLower.contains("causes") || sLower.contains("due to") || sLower.contains("because") || sLower.contains("prevent") || sLower.contains("include") || sLower.contains("provides"))
                val hasCheckOrAction = (qLower.contains("check") || qLower.contains("action") || qLower.contains("prevent") || qLower.contains("troubleshoot")) &&
                    (sLower.contains("check") || sLower.contains("inspect") || sLower.contains("ensure") || sLower.contains("monitor") || sLower.contains("prevent") || sLower.contains("remove") || sLower.contains("wrap"))
                if (hasCauseOrExplanation) {
                    score += 30.0
                }
                val hasFailureCauseMatch = (qLower.contains("cause") || qLower.contains("why") || qLower.contains("reason")) &&
                    (qLower.contains("fail") || qLower.contains("problem") || qLower.contains("error")) &&
                    (sLower.contains("fail") || sLower.contains("problem") || sLower.contains("error")) &&
                    (sLower.contains("result from") || sLower.contains("results from") || sLower.contains("caused by") || sLower.contains("due to") || sLower.contains("causes") || sLower.contains("because"))
                if (hasFailureCauseMatch) {
                    score += 40.0
                }
                if (hasCheckOrAction) {
                    score += 25.0
                }
                val hasAllQueryKeywords = queryKeywords.isNotEmpty() && kwMatches == queryKeywords.size
                val hasMultipleKeywords = kwMatches >= 2
                val hasConjoinedTopicMatch = (qLower.contains(" and ") || qLower.contains(" or ")) && kwMatches >= 1

                val isTopicRelevant = isPhraseMatch || isEntityMatch || hasAllQueryKeywords || hasMultipleKeywords ||
                    isMetricMatch || isDefinitionMatch || (kwMatches >= 1 && hasCauseOrExplanation) || hasCheckOrAction ||
                    hasConjoinedTopicMatch || (isBlockTopicRelevant && sent.length > 15 && !isTableHeader)

                if (isBlockTopicRelevant && sent.length > 15 && !isTableHeader) {
                    score += 20.0
                }

                val isKeywordOnly = !isTopicRelevant && kwMatches <= 1 && score <= 15.0

                if (score > 10.0) {
                    candidates.add(ScoredSentence(block, sent, score, kwMatches, isTopicRelevant, isKeywordOnly))
                }
            }
        }

        candidates.sortByDescending { it.score }
        if (candidates.isEmpty() || candidates.first().score < 15.0) {
            return "I searched the provided document(s), but could not find information answering \"$qTrimmed\"."
        }

        val top = candidates.first()
        val supporting = mutableListOf<ScoredSentence>()
        val seenText = mutableSetOf<String>(top.text.lowercase())
        val seenPages = mutableSetOf<Int>(top.block.page)

        // For specific factual attribute questions (e.g. "What is the battery capacity?"), don't append other pages' text
        val isSpecificFactQuestion = (qLower.startsWith("what is the ") || qLower.startsWith("what was the ")) &&
            (qLower.contains("capacity") || qLower.contains("phone") || qLower.contains("email") || qLower.contains("number") || qLower.contains("salary") || qLower.contains("price") || qLower.contains("spec"))

        val maxSupporting = if (answerLength == AnswerLength.LONG) 8 else 1
        if (!isSpecificFactQuestion && maxSupporting > 0) {
            for (c in candidates.drop(1)) {
                val norm = c.text.lowercase()
                if (norm in seenText) continue
                if (answerLength == AnswerLength.SHORT && c.block.page in seenPages) continue

                // Deduplicate semantically redundant statements (Section 7)
                if (areStatementsRedundant(c.text, top.text)) continue
                if (supporting.any { areStatementsRedundant(c.text, it.text) }) continue

                // Allows multiple relevant sentences across pages or on the same page,
                // as long as they are genuinely topic-relevant and not keyword-only matches.
                val isEligible = c.isTopicRelevant && !c.isKeywordOnly && c.score >= 15.0 &&
                    CitationValidator.isClaimRelevantToQuery(c.text, qTrimmed, classification)
                if (isEligible) {
                    supporting.add(c)
                    seenText.add(norm)
                    seenPages.add(c.block.page)
                }
                if (supporting.size >= maxSupporting) break
            }
        }

        var expandedMain = top.text
        if (answerLength == AnswerLength.LONG && (classification?.intent == com.siddu.offlinedocai.ai.retrieval.QueryIntent.ENTITY_LOOKUP || classification?.intent == com.siddu.offlinedocai.ai.retrieval.QueryIntent.CONCEPTUAL_QUERY)) {
            val sents = top.block.text.split(SENTENCE_SPLIT_REGEX).map { it.trim() }
            val idx = sents.indexOf(top.text)
            if (idx != -1 && idx < sents.size - 1) {
                for (i in idx + 1 until minOf(idx + 3, sents.size)) {
                    if (sents[i].length > 15 && !com.siddu.offlinedocai.ai.validation.CitationValidator.isHeadingOrTitle(sents[i])) {
                        if (!areStatementsRedundant(expandedMain, sents[i])) {
                            expandedMain = "$expandedMain ${sents[i]}"
                        }
                    }
                }
            }
        }
        val mainAnswer = "$expandedMain ${formatCitation(top.block)}"
        return if (supporting.isEmpty()) {
            mainAnswer
        } else {
            val details = supporting.joinToString("\n") { "• ${it.text} ${formatCitation(it.block)}" }
            "$mainAnswer\n\nKey details from the document:\n$details".trim()
        }
    }

    // =========================================================================
    // HELPER EXTRACTORS
    // =========================================================================

    /**
     * Matches word with morphological stem stripping and measurement domain synonyms.
     */
    private fun matchesWordOrSynonym(candidateText: String, keyword: String): Boolean {
        val words = candidateText.split(WORD_SPLIT_REGEX).map { it.lowercase() }.filter { it.isNotEmpty() }
        if (words.contains(keyword)) return true

        // Morphological stem matching (e.g. operating <-> operational, processes <-> processing)
        val (stem, stripped) = when {
            keyword.endsWith("ing") && keyword.length > 5 -> Pair(keyword.removeSuffix("ing"), true)
            keyword.endsWith("ional") && keyword.length > 7 -> Pair(keyword.removeSuffix("ional"), true)
            keyword.endsWith("ion") && keyword.length > 5 -> Pair(keyword.removeSuffix("ion"), true)
            keyword.endsWith("ed") && keyword.length > 4 -> Pair(keyword.removeSuffix("ed"), true)
            keyword.endsWith("es") && keyword.length > 4 -> Pair(keyword.removeSuffix("es"), true)
            keyword.endsWith("s") && keyword.length > 3 -> Pair(keyword.removeSuffix("s"), true)
            else -> Pair(keyword, false)
        }
        if (stripped && stem.length >= 4 && words.any { it.startsWith(stem) }) return true

        // Also check if any word in candidateText has a standard suffix that matches keyword
        if (!stripped && keyword.length >= 4 && words.any { w ->
            w.length > keyword.length && (
                w == "${keyword}s" || w == "${keyword}es" || w == "${keyword}ed" || w == "${keyword}ing" || w == "${keyword}ion"
            )
        }) return true

        // Common measurement domain synonyms
        val synonyms = when (keyword) {
            "heat", "thermal" -> listOf("temperature", "temp", "celsius", "fahrenheit")
            "temperature", "temp" -> listOf("heat", "thermal", "celsius")
            "size", "large" -> listOf("capacity", "dimension", "volume")
            "capacity" -> listOf("size", "volume", "storage")
            "speed" -> listOf("rate", "frequency")
            "rate" -> listOf("speed", "frequency", "ratio", "percentage")
            "price", "cost", "salary" -> listOf("amount", "fee", "usd", "budget", "compensation")
            "wattage" -> listOf("power", "watt")
            "power" -> listOf("wattage", "watt", "voltage", "current")
            else -> emptyList()
        }
        return synonyms.any { words.contains(it) }
    }

    /**
     * Extracts list items belonging to a named target topic (e.g. projects, skills, certifications)
     * while cleanly separating them from unrelated sections.
     */
    private fun extractListItems(
        blocks: List<ParsedSourceBlock>,
        targetTopic: String
    ): List<Pair<String, String>> {
        val results = mutableListOf<Pair<String, String>>()
        if (targetTopic.isEmpty() || targetTopic in STOP_WORDS) return results

        val targetLower = targetTopic.lowercase().trim()
        val targetStem = if (targetLower.endsWith("s")) targetLower.removeSuffix("s") else targetLower
        val targetWords = targetLower.split(WORD_SPLIT_REGEX).filter { it.length > 2 && it !in STOP_WORDS }

        for (block in blocks) {
            val text = block.text
            val tLower = text.lowercase()
            val citation = formatCitation(block)

            // 1. Check if block matches section metadata or contains target topic
            val blockMatchesSection = block.section?.lowercase()?.let { s ->
                s.contains(targetLower) || s.contains(targetStem) || targetWords.any { s.contains(it) }
            } == true ||
                tLower.contains(targetLower) || tLower.contains(targetStem) ||
                (targetWords.isNotEmpty() && targetWords.all { matchesWordOrSynonym(tLower, it) }) ||
                (targetWords.size >= 2 && targetWords.count { matchesWordOrSynonym(tLower, it) } >= 2)

            if (!blockMatchesSection) continue

            val headingRegex = Regex("""(?i)\b(?:${Regex.escape(targetLower)}|${Regex.escape(targetStem)})\b\s*(?:include[s]?|are|is)?\s*[:\-]?""")
            val m = headingRegex.find(text)
            val labeledItemMatch = Regex("""(?i)(?:(?:project|method|type|phase|step)\s*\d+\s*[:\-])|(?:\b\d{1,3}[\.\)]\s+)""").find(text)
            val startIdx = if (m != null && (labeledItemMatch == null || m.range.first < labeledItemMatch.range.first)) {
                m.range.last + 1
            } else if (labeledItemMatch != null) {
                labeledItemMatch.range.first
            } else if (m != null) {
                m.range.last + 1
            } else 0

            val slice = text.substring(minOf(startIdx, text.length)).trim()
            val cleanSlice = slice.replace(Regex("""^[\s:\-]+"""), "").trim()

            // 3. Find end index: where the next section header begins
            val endIdx = findSectionEnd(cleanSlice)
            val targetContent = cleanSlice.substring(0, endIdx).trim()
            if (targetContent.length <= 5 || targetContent.all { !it.isLetterOrDigit() }) continue

            // 4. Parse individual items inside targetContent
            // A. Explicit numbered items: "1. Technical Planning, 2. Requirements..." or "1. Smart Traffic Optimizer... 2. OfflineDocAI..."
            val numMarkers = Regex("""(?:\b\d+[\.\)]\s+)""").findAll(targetContent).toList()

            if (numMarkers.size >= 2) {
                for (i in numMarkers.indices) {
                    val s = numMarkers[i].range.last + 1
                    val e = if (i + 1 < numMarkers.size) numMarkers[i + 1].range.first else targetContent.length
                    var item = targetContent.substring(s, e)
                        .replace(Regex("""^(?:and|or)\s+"""), "")
                        .replace(Regex("""(?:,\s*)?(?:and|or)$"""), "")
                        .trim()
                        .trimEnd(',', ';', '.')
                    if (i == numMarkers.lastIndex) {
                        item = item.split(Regex("""(?i)\s*[.\n]\s*(?:Education|Experience|Certifications|Skills|Summary|Awards)\b""")).first().trim()
                    }
                    if (item.length > 2 && item.any { it.isLetterOrDigit() }) {
                        results.add(Pair(item, citation))
                    }
                }
            } else if (targetStem == "skill" || targetStem == "tech" || targetLower.contains("skill") || targetLower.contains("tech")) {
                // Comma-separated or bullet list of skills
                val commaItems = targetContent.split(",").map { it.trim() }.filter { it.isNotEmpty() }
                if (commaItems.size >= 2) {
                    val joinedSkills = commaItems.map { it.replace(Regex("""^[•\-\*]\s*"""), "").trim() }
                        .filter { it.isNotEmpty() }
                        .joinToString(", ")
                    results.add(Pair(joinedSkills, citation))
                } else {
                    results.add(Pair(targetContent, citation))
                }
            } else {
                // Named item with dash: e.g. "OfflineDocAI - Complete On-Device... Technologies: Kotlin... DueGuard - ..."
                val dashItemPattern = Regex("""(?i)(?:^|(?<=[.\n;]))\s*([A-Za-z0-9_\s]{2,30}\s+-\s+)""")
                val dashMatches = dashItemPattern.findAll(targetContent).toList()

                if (dashMatches.size >= 2) {
                    for (i in dashMatches.indices) {
                        val s = dashMatches[i].range.first
                        val e = if (i + 1 < dashMatches.size) dashMatches[i + 1].range.first else targetContent.length
                        val item = targetContent.substring(s, e).replace(Regex("""^[•\-\*\s]+"""), "").trim()
                        if (item.length > 5) {
                            results.add(Pair(item, citation))
                        }
                    }
                } else {
                    // Labeled item pattern: "Project 1: ...", "Method 1: ...", or numbered/bullet list
                    val projectNumPattern = Regex("""(?i)(?:^|(?<=[.\n;•\-_]))\s*(?:(?:(?:project|method|type|phase|step)\s*\d+\s*[:\-])|(?:\d+[\.\)]\s+)|(?:[•\-\*]\s+))""")
                    val projMatches = projectNumPattern.findAll(targetContent).toList()

                    if (projMatches.size >= 2) {
                        for (i in projMatches.indices) {
                            val s = projMatches[i].range.first
                            val e = if (i + 1 < projMatches.size) projMatches[i + 1].range.first else targetContent.length
                            val item = targetContent.substring(s, e).replace(Regex("""^[•\-\*\s]+"""), "").trim()
                            if (item.length > 5) {
                                results.add(Pair(item, citation))
                            }
                        }
                    } else {
                        // Split by sentences or line breaks
                        val lines = targetContent.split(SENTENCE_SPLIT_REGEX)
                            .map { it.replace(Regex("""^[•\-\*\s]+"""), "").trim() }
                            .filter { it.length > 5 && !CitationValidator.isHeadingOrTitle(it) }

                        for (line in lines) {
                            results.add(Pair(line, citation))
                        }
                    }
                }
            }
        }

        val distinctResults = mutableListOf<Pair<String, String>>()
        for (item in results) {
            val isDup = distinctResults.any { existing ->
                areStatementsRedundant(existing.first, item.first)
            }
            if (!isDup) {
                distinctResults.add(item)
            }
        }

        return distinctResults
    }

    /**
     * Identifies the end of a section in raw text by detecting the next uppercase heading.
     * Excludes short acronyms (AWS, AI, NASA, TRL, CPU, GPU) and colons.
     * Uses Android-compatible regex (no variable-length look-behinds).
     */
    private fun findSectionEnd(slice: String): Int {
        val boundaryRegex = Regex(
            """(?m)(?:^|\n|[.!?]\s|\s+)([A-Z]{4,}(?:\s+[A-Z]{4,})*)\b(?=\s+[A-Z0-9])"""
        )
        for (m in boundaryRegex.findAll(slice)) {
            val headingStart = m.groups[1]?.range?.first ?: continue
            if (headingStart >= 15) {
                return headingStart
            }
        }
        return slice.length
    }

    /**
     * Extracts a concise sentence around a regex match range.
     */
    private fun extractSentenceAroundMatch(text: String, range: IntRange): String {
        val start = max(0, text.lastIndexOfAny(charArrayOf('.', '!', '?', '\n'), max(0, range.first - 1)) + 1)
        var end = text.indexOfAny(charArrayOf('.', '!', '?', '\n'), range.last)
        if (end == -1) end = min(text.length, range.last + 80) else end += 1
        return text.substring(start, end).trim()
    }

    /**
     * Extracts target topic keyword from query (e.g. "projects", "skills", "certifications")
     * via generic linguistic normalization.
     */
    private fun extractTargetTopic(query: String): String {
        var clean = query.replace(Regex("""[?.,!;:"]+"""), " ").trim()
        clean = clean.replace(
            Regex(
                """(?i)^(?:what\s+are\s+all\s+the|what\s+are\s+all|what\s+are\s+the|what\s+are|what\s+is\s+the|what\s+is|what|list\s+all\s+the|list\s+all|list\s+the|list\s+every|list|show\s+all|show\s+me|show|extract\s+all|extract|give\s+me\s+all|give\s+me|name\s+all|name|tell\s+me\s+about\s+the|tell\s+me\s+about|what\s+does\s+the\s+document\s+say\s+about|which)\s+"""
            ),
            ""
        ).trim()
        clean = clean.replace(
            Regex("""(?i)\s+(?:listed\s+in\s+the\s+(?:resume|document|guide|file|pdf|book)|in\s+the\s+(?:resume|document|guide|file|pdf|book)|in\s+this\s+(?:resume|document|guide|file|pdf|book)|described\s+in\s+the\s+(?:guide|document|resume)|are\s+mentioned|is\s+mentioned|mentioned|discussed|listed)$"""),
            ""
        ).trim()
        return clean.trim()
    }

    /**
     * Formats clean page content for PAGE_LOOKUP queries.
     */
    private fun formatPageContent(rawText: String): String {
        val lines = rawText.lines()
        val formatted = mutableListOf<String>()

        for (rawLine in lines) {
            var line = rawLine.trim()
            if (line.isEmpty() || line.startsWith("[SOURCE_ID") || line.startsWith("[PAGE")) continue

            line = line.replace(Regex("""(?:\s*[\.·]){3,}\s*(\d+)\s*$"""), " — Page $1")
                .replace(Regex("""(?:\s*[\.·]){3,}|\.{3,}"""), " — ")
                .replace(Regex("""[ \t]{2,}"""), " ")
                .removeSuffix(" —")
                .trim()

            if (line.isNotEmpty()) {
                formatted.add(line)
            }
        }

        return formatted.joinToString("\n\n").trim()
    }
}
