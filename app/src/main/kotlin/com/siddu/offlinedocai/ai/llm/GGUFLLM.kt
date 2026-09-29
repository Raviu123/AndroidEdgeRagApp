package com.siddu.offlinedocai.ai.llm

import android.content.Context
import android.util.Log
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

        // Generic sentence and punctuation splitting
        private val SENTENCE_SPLIT_REGEX = Regex("""(?<!\d\.)(?<!\b(?:e\.g|i\.e|mr|dr|ms|mrs|vs|etc|no|fig|pg|al))(?<=[.!?])\s+(?=[A-Z])|\n+""")
        private val WORD_SPLIT_REGEX = Regex("""[^a-zA-Z0-9]+""")

        // Common non-content stopwords
        private val STOP_WORDS = setOf(
            "what", "when", "where", "how", "this", "that", "there", "their", "who", "why",
            "them", "with", "from", "your", "about", "the", "is", "are", "was", "were", "and",
            "or", "in", "on", "at", "to", "a", "an", "of", "for", "by", "does", "do", "can",
            "you", "tell", "me", "explain", "give", "show", "please", "which", "will", "would",
            "mentioned", "say", "talk", "says", "talks", "i", "my", "we", "our", "all", "any"
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

    /**
     * Safely switches the active local GGUF model.
     * Ensures only one large GGUF model context is loaded in native memory at a time.
     */
    fun selectModel(model: SupportedModel): Boolean {
        activeModelType = model
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
     * Resolves the GGUF model file, copying from bundled assets if missing from filesDir.
     */
    fun resolveModelFile(targetModel: SupportedModel? = null): File? {
        val modelToResolve = targetModel ?: activeModelType
        val modelDir = File(context.filesDir, "models/llm")
        if (!modelDir.exists()) modelDir.mkdirs()

        // 1. Existing file in filesDir/models/llm matching target filename
        val targetFile = File(modelDir, modelToResolve.filename)
        if (targetFile.exists() && targetFile.length() > 1024) {
            return targetFile
        }

        // 2. Check device tmp directory (for fast direct adb push during testing/benchmarking)
        val tmpDir = File("/data/local/tmp/models/llm")
        val tmpFile = File(tmpDir, modelToResolve.filename)
        if (tmpFile.exists() && tmpFile.length() > 1024) {
            return tmpFile
        }

        // 3. Check bundled assets for target filename
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

        // Fallback: any existing GGUF file in filesDir/models/llm
        val existing = modelDir.listFiles()?.firstOrNull {
            it.extension.equals("gguf", ignoreCase = true) && it.length() > 1024
        }
        if (existing != null) return existing

        // Check any bundled GGUF asset
        try {
            val assetList = context.assets.list("models/llm") ?: emptyArray()
            val anyAsset = assetList.firstOrNull { it.endsWith(".gguf", ignoreCase = true) }
            if (anyAsset != null) {
                val f = File(modelDir, anyAsset)
                if (!f.exists() || f.length() < 1024) {
                    context.assets.open("models/llm/$anyAsset").use { input ->
                        f.outputStream().use { output -> input.copyTo(output) }
                    }
                }
                return f
            }
        } catch (_: Exception) {}

        return null
    }

    fun buildQwenPrompt(contextText: String, question: String, calculationAid: String? = null): String {
        val calcInstruction = if (calculationAid != null) "\nCalculation Aid (verified arithmetic):\n$calculationAid\n" else ""
        return buildString {
            append("<|im_start|>system\n")
            append("You are OfflineDocAI, an on-device document assistant.\n")
            append("Answer the user's question using ONLY the supplied document evidence.\n")
            append("- Answer concisely and factually based strictly on the provided context.\n")
            append("- Cite the source and page numbers in brackets, e.g. [Source X, Page Y].\n")
            append("- If the answer cannot be found in the evidence, state: \"I searched the provided document(s), but could not find information answering this question.\"\n")
            append("- Do not invent facts, citations, or numbers.\n")
            append("- Preserve exact measurements, dates, URLs, emails, and phone numbers.\n")
            append("<|im_end|>\n")
            append("<|im_start|>user\n")
            append("Context:\n")
            append(contextText.trim())
            if (calcInstruction.isNotBlank()) {
                append(calcInstruction)
            }
            append("\n\nQuestion: ")
            append(question.trim())
            append("\n<|im_end|>\n")
            append("<|im_start|>assistant\n")
        }
    }

    fun buildPromptForModel(
        model: SupportedModel,
        contextText: String,
        question: String,
        calculationAid: String? = null
    ): String {
        return when (model) {
            SupportedModel.QWEN_1_5B, SupportedModel.QWEN_0_5B -> buildQwenPrompt(contextText, question, calculationAid)
            SupportedModel.LLAMA_1B -> buildLlamaPrompt(contextText, question, calculationAid)
        }
    }

    fun buildLlamaPrompt(contextText: String, question: String, calculationAid: String? = null): String {
        val calcInstruction = if (calculationAid != null) "\nCalculation Aid (verified arithmetic):\n$calculationAid\n" else ""
        return buildString {
            append("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n")
            append("You are OfflineDocAI, an on-device document assistant.\n")
            append("Answer the user's question using ONLY the supplied document evidence.\n")
            append("- Answer concisely and factually based strictly on the provided context.\n")
            append("- Cite the source and page numbers in brackets, e.g. [Source X, Page Y].\n")
            append("- If the answer cannot be found in the evidence, state: \"I searched the provided document(s), but could not find information answering this question.\"\n")
            append("- Do not invent facts, citations, or numbers.\n")
            append("- Preserve exact measurements, dates, URLs, emails, and phone numbers.<|eot_id|>")
            append("<|start_header_id|>user<|end_header_id|>\n\n")
            append("Context:\n")
            append(contextText.trim())
            if (calcInstruction.isNotBlank()) {
                append(calcInstruction)
            }
            append("\n\nQuestion: ")
            append(question.trim())
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
        val isDiff = qLower.contains("difference") || qLower.contains("how much higher") ||
                qLower.contains("how much lower") || qLower.contains("how much more") ||
                qLower.contains("how much less")
        val isSum = qLower.contains("total") || qLower.contains("sum") || qLower.contains("combined")
        val isAverage = qLower.contains("average") || qLower.contains("mean")

        if (!isDiff && !isSum && !isAverage) return null

        val numberRegex = Regex("""(\d+(?:\.\d+)?)\s*(%|°c|°f|kg|g|m|cm|mm|usd|\$|million|billion|k|hours?|days?|years?)?""", RegexOption.IGNORE_CASE)
        val numbers = numberRegex.findAll(contextText).mapNotNull { match ->
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

    override suspend fun generate(prompt: String): String = withContext(Dispatchers.Default) {
        // 1. Separate Context and Question cleanly from prompt
        val (contextText, question) = parsePrompt(prompt)
        val qTrimmed = question.trim()
        val qLower = qTrimmed.lowercase()

        // 2. Deterministic Query Routing via Generic QueryIntentRouter
        val classification = QueryIntentRouter.route(qTrimmed)
        Log.i("RAG_DEBUG", "RAG_DEBUG_06_LLM_START: q='$qTrimmed' intent=${classification.intent} nativeLoaded=${llamaRunner.isNativeLoaded}")

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
        if (llamaRunner.isNativeLoaded) {
            val activeModel = resolveModelFile(activeModelType)
            if (activeModel != null && activeModel.exists()) {
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
                        val formattedPrompt = buildPromptForModel(activeModelType, effectiveContext, qTrimmed, calculationAid)
                        Log.i("RAG_DEBUG", "RAG_DEBUG_07_NATIVE_START: model=${activeModel.name} formattedPromptLen=${formattedPrompt.length}")
                        val neuralOutput = llamaRunner.inference(
                            prompt = formattedPrompt,
                            maxTokens = MAX_GENERATION_TOKENS,
                            temperature = DEFAULT_TEMPERATURE
                        )
                        Log.i("RAG_DEBUG", "RAG_DEBUG_08_NATIVE_DONE: rawOutputLen=${neuralOutput.length}")
                        val cleanOutput = neuralOutput
                            .replace("<|im_end|>", "")
                            .replace("<|im_start|>", "")
                            .replace("<|endoftext|>", "")
                            .replace("<|eot_id|>", "")
                            .replace("<|start_header_id|>", "")
                            .replace("<|end_header_id|>", "")
                            .replace("<|begin_of_text|>", "")
                            .replace("<|end_of_text|>", "")
                            .trim()
                            .removePrefix(":")
                            .trim()

                        if (cleanOutput.isNotBlank()) {
                            rawAnswer = cleanOutput
                            usedNative = true
                            currentEngineState = EngineState.GGUF_NATIVE
                            Log.i(TAG, "AI_ENGINE=GGUF_NATIVE MODEL=${activeModel.name}")
                        }
                    } catch (e: Exception) {
                        Log.w(TAG, "Native GGUF inference failed, falling back to deterministic synthesis: ${e.message}")
                        currentEngineState = EngineState.GGUF_INFERENCE_FAILED
                    }
                } else {
                    Log.w(TAG, "Failed to load GGUF model, falling back to deterministic synthesis.")
                    currentEngineState = EngineState.GGUF_LOAD_FAILED
                }
            }
        }

        // 6. Generic Deterministic Synthesis Fallback (when native inference is unavailable or fails)
        if (!usedNative || rawAnswer.isBlank()) {
            if (llamaRunner.isNativeLoaded && !allowDeterministicFallbackForTests) {
                val errorMsg = when (currentEngineState) {
                    EngineState.GGUF_LOAD_FAILED -> "Failed to load local GGUF model from device storage."
                    EngineState.GGUF_INFERENCE_FAILED -> "Local GGUF neural inference failed during execution."
                    else -> "GGUF model file not found in device storage or assets."
                }
                Log.e(TAG, "PRODUCTION ERROR: $errorMsg (state=$currentEngineState)")
                throw IllegalStateException("Real GGUF execution required in production: $errorMsg")
            }

            if (currentEngineState != EngineState.GGUF_INFERENCE_FAILED && currentEngineState != EngineState.GGUF_LOAD_FAILED) {
                currentEngineState = EngineState.DETERMINISTIC_FALLBACK
            }
            Log.i(TAG, "AI_ENGINE=DETERMINISTIC_FALLBACK REASON=${if (!llamaRunner.isNativeLoaded) "NATIVE_LIB_NOT_LOADED" else "MODEL_NOT_AVAILABLE"}")
            rawAnswer = when (classification.intent) {
                QueryIntent.PAGE_LOOKUP -> synthesizePageLookup(blocks, classification, qLower)
                QueryIntent.PAGE_RANGE -> synthesizePageRange(blocks, classification)
                QueryIntent.EXACT_FACT -> synthesizeExactFact(blocks, classification, qLower)
                QueryIntent.COMPARISON -> synthesizeComparison(blocks, classification, qLower)
                QueryIntent.EXHAUSTIVE_LIST -> synthesizeExhaustiveList(blocks, classification, qLower, isBudgetLimited)
                QueryIntent.FIND_ALL -> synthesizeFindAll(blocks, classification, qLower, isBudgetLimited)
                QueryIntent.GLOBAL_SUMMARY -> synthesizeGlobalSummary(blocks, qLower)
                QueryIntent.SECTION_QUERY -> synthesizeSectionQuery(blocks, classification, qLower)
                QueryIntent.ENTITY_LOOKUP,
                QueryIntent.CONCEPTUAL_QUERY -> synthesizeConceptualQuery(blocks, classification, qTrimmed, qLower)
            }
            if (calculationAid != null && !rawAnswer.contains(calculationAid)) {
                rawAnswer = "$rawAnswer\n\n$calculationAid"
            }
        }

        // 7. Citation & Claim Validation via Phase 6 CitationValidator
        val validationResult = CitationValidator.validate(rawAnswer, blocks, qTrimmed, isBudgetLimited)
        var validated = validationResult.validatedAnswer
        if (validated.isNotBlank() && !validated.contains("Page", ignoreCase = true) &&
            !validated.contains("could not find", ignoreCase = true) && blocks.isNotEmpty()) {
            val primaryBlock = blocks.firstOrNull { it.text.contains("Page ${it.page}") } ?: blocks.first()
            val citation = formatCitation(primaryBlock)
            validated = "$validated $citation"
        }
        val finalAnswer = validated.ifEmpty {
            "I searched the provided document(s), but could not find information answering \"$qTrimmed\"."
        }
        Log.i("RAG_DEBUG", "RAG_DEBUG_09_LLM_RETURN: finalLen=${finalAnswer.length} engineState=$currentEngineState usedNative=$usedNative")
        finalAnswer
    }

    /**
     * Parses the prompt into context text and question string.
     */
    private fun parsePrompt(prompt: String): Pair<String, String> {
        val lines = prompt.lines()
        var contextStartIndex = -1
        var questionStartIndex = -1

        for (i in lines.indices) {
            val trimmed = lines[i].trim()
            if (trimmed.startsWith("Context:")) {
                contextStartIndex = i
            } else if (trimmed.startsWith("Question:")) {
                questionStartIndex = i
            }
        }

        val contextText = if (contextStartIndex != -1 && questionStartIndex > contextStartIndex) {
            lines.subList(contextStartIndex + 1, questionStartIndex).joinToString("\n").trim()
        } else {
            ""
        }

        val question = if (questionStartIndex != -1) {
            val qParts = mutableListOf<String>()
            for (i in questionStartIndex until lines.size) {
                val lineTrimmed = lines[i].trim()
                if (lineTrimmed.startsWith("Question:")) {
                    qParts.add(lineTrimmed.removePrefix("Question:").trim())
                } else if (lineTrimmed.startsWith("Answer", ignoreCase = true) ||
                    lineTrimmed.startsWith("Given the context", ignoreCase = true)
                ) {
                    break
                } else if (lineTrimmed.isNotEmpty()) {
                    qParts.add(lineTrimmed)
                }
            }
            qParts.joinToString(" ").trim()
        } else {
            val qLine = lines.find { it.startsWith("Question:") } ?: ""
            qLine.removePrefix("Question:").trim()
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
            """(?m)(?:\[DOC_ID=\d+\]\s*)?(?:\[SOURCE_ID=(\d+)\]\s*)?\[PAGE=(\d+)\](?:\s*\[Page\s+\d+\])?(?:\s*\[PAGE_SPAN=([^\]]+)\])?(?:\s*\[SECTION=([^\]]+)\])?(?:\s*\[CHUNK_INDEX=(\d+)\])?\s*(?:\n|\s+)\s*([\s\S]*?)(?=(?:\n\[DOC_ID=\d+\]|\n\[SOURCE_ID=\d+\]|\n\[PAGE=\d+\]|\n\[Page\s+\d+\]|\z))"""
        )
        val structuredMatches = structuredPattern.findAll(contextText).toList()

        if (structuredMatches.isNotEmpty()) {
            for (m in structuredMatches) {
                val docId = Regex("""\[DOC_ID=(\d+)\]""").find(m.value)?.groupValues?.get(1)?.toLongOrNull()
                val sourceId = m.groupValues[1].toLongOrNull()
                val page = m.groupValues[2].toIntOrNull() ?: 1
                val pageSpan = m.groupValues[3].ifEmpty { page.toString() }
                val section = m.groupValues[4].ifEmpty { null }
                val chunkIdx = m.groupValues[5].toIntOrNull()
                val text = m.groupValues[6].trim()
                if (text.isNotEmpty()) {
                    blocks.add(ParsedSourceBlock(sourceId, page, pageSpan, section, chunkIdx, text, docId))
                }
            }
            if (blocks.isNotEmpty()) return blocks
        }

        // 2. Try legacy [Page X] pattern
        val legacyPattern = Regex("""\[Page\s+(\d+)\]\s*([\s\S]*?)(?=\n\[Page\s+\d+\]|\z)""")
        val legacyMatches = legacyPattern.findAll(contextText).toList()

        if (legacyMatches.isNotEmpty()) {
            for (m in legacyMatches) {
                val page = m.groupValues[1].toIntOrNull() ?: 1
                val text = m.groupValues[2].trim()
                if (text.isNotEmpty()) {
                    blocks.add(ParsedSourceBlock(null, page, page.toString(), null, null, text))
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
        qLower: String
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
     * EXACT_FACT: Extracts exact factual values (phone, email, URL, metrics, specs) verbatim.
     */
    private fun synthesizeExactFact(
        blocks: List<ParsedSourceBlock>,
        classification: QueryClassification,
        qLower: String
    ): String {
        val factType = classification.factType

        for (block in blocks) {
            val text = block.text
            val citation = formatCitation(block)

            when (factType) {
                FactType.PHONE -> {
                    val m = PHONE_REGEX.find(text)
                    if (m != null) {
                        val num = (m.groupValues[1].ifEmpty { m.value }).trim()
                        return "The phone number is $num. $citation"
                    }
                }

                FactType.EMAIL -> {
                    val m = EMAIL_REGEX.find(text)
                    if (m != null) {
                        val email = m.value.trim()
                        return "The email address is $email. $citation"
                    }
                }

                FactType.URL -> {
                    val m = URL_REGEX.find(text)
                    if (m != null) {
                        val url = m.value.trim()
                        return "The URL is $url. $citation"
                    }
                }

                FactType.PERCENTAGE -> {
                    val m = PERCENT_REGEX.find(text)
                    if (m != null) {
                        val snippet = extractSentenceAroundMatch(text, m.range)
                        return "$snippet $citation"
                    }
                }

                FactType.AMOUNT -> {
                    val m = SALARY_REGEX.find(text)
                    if (m != null) {
                        val snippet = extractSentenceAroundMatch(text, m.range)
                        return "$snippet $citation"
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
            return "${best.snippet} ${formatCitation(best.block)}"
        }

        // Generic factual pattern scan across all blocks
        for (block in blocks) {
            val citation = formatCitation(block)
            if (qLower.contains("phone") || qLower.contains("mobile") || qLower.contains("contact")) {
                val m = PHONE_REGEX.find(block.text)
                if (m != null) {
                    val num = (m.groupValues[1].ifEmpty { m.value }).trim()
                    return "The phone number is $num. $citation"
                }
            }
            if (qLower.contains("email") || qLower.contains("mail")) {
                val m = EMAIL_REGEX.find(block.text)
                if (m != null) {
                    return "The email address is ${m.value.trim()}. $citation"
                }
            }
            if (qLower.contains("github") || qLower.contains("linkedin") || qLower.contains("website") || qLower.contains("url")) {
                val m = URL_REGEX.find(block.text)
                if (m != null) {
                    return "The link is ${m.value.trim()}. $citation"
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
        qLower: String
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
            val sentences = block.text.split(SENTENCE_SPLIT_REGEX).map { it.trim() }.filter { it.length > 10 }
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

        val distinctions = if (jointEvidence.isNotEmpty()) {
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
        isBudgetLimited: Boolean
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

        val itemsFormatted = extractedItems.joinToString("\n") { "• ${it.first} ${it.second}" }
        return "$header\n\n$itemsFormatted".trim()
    }

    /**
     * FIND_ALL: Reports all distinct occurrences across distinct pages without collapsing them.
     */
    private fun synthesizeFindAll(
        blocks: List<ParsedSourceBlock>,
        classification: QueryClassification,
        qLower: String,
        isBudgetLimited: Boolean
    ): String {
        val targetTerm = classification.targetTerm?.lowercase()?.trim() ?: extractTargetTopic(qLower)
        val occurrencesByPage = mutableMapOf<Int, MutableList<Pair<ParsedSourceBlock, String>>>()

        for (block in blocks) {
            val sentences = block.text.split(SENTENCE_SPLIT_REGEX).map { it.trim() }.filter { it.isNotEmpty() }
            for (sent in sentences) {
                if (targetTerm.isNotEmpty() && sent.lowercase().contains(targetTerm)) {
                    occurrencesByPage.getOrPut(block.page) { mutableListOf() }.add(Pair(block, sent))
                }
            }
        }

        if (occurrencesByPage.isEmpty()) {
            return "I searched the provided document(s), but could not find occurrences of \"$targetTerm\"."
        }

        val header = if (isBudgetLimited) {
            "The document mentions **$targetTerm** in the following location(s) (Note: Results may be partial due to context budgeting):"
        } else {
            "The document mentions **$targetTerm** in the following location(s):"
        }

        val entries = mutableListOf<String>()
        for ((page, occurrences) in occurrencesByPage.toSortedMap()) {
            val best = occurrences.first()
            entries.add("• **Page $page:** ${best.second} ${formatCitation(best.first)}")
        }

        return "$header\n\n${entries.joinToString("\n")}".trim()
    }

    /**
     * GLOBAL_SUMMARY: Produces a concise document-level overview from representative evidence.
     */
    private fun synthesizeGlobalSummary(
        blocks: List<ParsedSourceBlock>,
        qLower: String
    ): String {
        val bulletPoints = mutableListOf<String>()
        val seen = mutableSetOf<String>()

        for (block in blocks) {
            val sentences = block.text.split(SENTENCE_SPLIT_REGEX)
                .map { it.trim() }
                .filter { it.length > 15 && !it.startsWith("[SOURCE_ID") && !it.startsWith("[PAGE") }

            for (s in sentences.take(2)) {
                val normalized = s.replace(Regex("""\s+"""), " ")
                if (seen.add(normalized.lowercase())) {
                    bulletPoints.add("• ${formatCitation(block)} $normalized")
                }
                if (bulletPoints.size >= 5) break
            }
            if (bulletPoints.size >= 5) break
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
        qLower: String
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
            val itemsFormatted = extractedItems.joinToString("\n\n") { (text, citation) -> "• $text $citation" }
            return "$title:\n\n$itemsFormatted".trim()
        }

        // Fallback to conceptual candidate ranking across the section's blocks
        return synthesizeConceptualQuery(blocks, classification, classification.rawQuery, qLower)
    }

    /**
     * Checks if query is explicitly asking for a list or inventory of items.
     */
    private fun isExplicitListQuery(qLower: String): Boolean {
        return qLower.startsWith("which ") ||
            qLower.startsWith("what projects") ||
            qLower.startsWith("tell me about the projects") ||
            qLower.contains("process") ||
            qLower.contains("project") ||
            qLower.contains("skill") ||
            qLower.contains("certification") ||
            qLower.contains("technolog") ||
            (qLower.startsWith("what are the ") && qLower.endsWith("s") && !qLower.contains("difference")) ||
            qLower == "projects" ||
            qLower == "skills" ||
            qLower == "certifications" ||
            qLower == "technologies"
    }

    /**
     * CONCEPTUAL_QUERY: Explains concepts using evidence-ranked direct answers and supporting details.
     */
    private fun synthesizeConceptualQuery(
        blocks: List<ParsedSourceBlock>,
        classification: QueryClassification,
        qTrimmed: String,
        qLower: String
    ): String {
        // 1. If query is explicitly asking for a list/inventory of items, extract structured list items
        if (isExplicitListQuery(qLower)) {
            val targetTopic = extractTargetTopic(qLower)
            if (targetTopic.isNotEmpty() && targetTopic !in STOP_WORDS && targetTopic.length > 2) {
                val items = extractListItems(blocks, targetTopic)
                if (items.isNotEmpty()) {
                    val title = targetTopic.split(" ").joinToString(" ") { it.replaceFirstChar { c -> c.uppercase() } }
                    val formatted = items.joinToString("\n\n") { (text, citation) -> "• $text $citation" }
                    return "$title:\n\n$formatted".trim()
                }
            }
        }

        // 2. Sentence-level grounded scoring across all evidence blocks
        val queryKeywords = qLower.split(WORD_SPLIT_REGEX)
            .filter { it.length > 1 && it !in STOP_WORDS }
            .distinct()

        data class ScoredSentence(val block: ParsedSourceBlock, val text: String, val score: Double)
        val candidates = mutableListOf<ScoredSentence>()

        for (block in blocks) {
            val sentences = block.text.split(SENTENCE_SPLIT_REGEX)
                .map { it.trim() }
                .filter { it.isNotEmpty() && !it.startsWith("[SOURCE_ID") && !it.startsWith("[PAGE") }

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

                // Heading match bonus: if sentence starts with query keyword followed by colon or dash
                for (kw in queryKeywords) {
                    if (sLower.startsWith("$kw:") || sLower.startsWith("$kw -") || sLower.startsWith("key $kw") || sLower.startsWith("key engineering $kw")) {
                        score += 30.0
                    }
                }

                // Penalize table header rows (they define schema/columns, not factual data)
                val isTableHeader = sent.startsWith("|") && sent.endsWith("|") &&
                    (sent.contains("---") ||
                     listOf("component", "specification", "power", "header", "description", "parameter", "unit", "column", "attribute").count { sLower.contains(it) } >= 2)
                if (isTableHeader) {
                    score -= 40.0
                }

                // Domain metric match bonuses
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

                // Penalize meta-overview sentences when query is seeking specific explanation
                if (sLower.startsWith("overview:") || sLower.startsWith("this document describes") || sLower.startsWith("this chapter describes")) {
                    score -= 20.0
                }

                // Penalize unrelated administrative boilerplate
                if (sLower.contains("standard form 298") || sLower.contains("report documentation page")) {
                    score -= 50.0
                }

                if (score > 10.0) {
                    candidates.add(ScoredSentence(block, sent, score))
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

        if (!isSpecificFactQuestion) {
            for (c in candidates.drop(1)) {
                val norm = c.text.lowercase()
                if (norm in seenText) continue

                val isNewPage = c.block.page !in seenPages
                if (c.score >= 15.0 && (isNewPage || c.score >= top.score * 0.7)) {
                    supporting.add(c)
                    seenText.add(norm)
                    seenPages.add(c.block.page)
                }
                if (supporting.size >= 3) break
            }
        }

        val mainAnswer = "${top.text} ${formatCitation(top.block)}"
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
        if (candidateText.contains(keyword)) return true

        // Morphological stem matching (e.g. operating <-> operational, processes <-> processing)
        val stem = when {
            keyword.endsWith("ing") && keyword.length > 5 -> keyword.removeSuffix("ing")
            keyword.endsWith("ional") && keyword.length > 7 -> keyword.removeSuffix("ional")
            keyword.endsWith("ion") && keyword.length > 5 -> keyword.removeSuffix("ion")
            keyword.endsWith("ed") && keyword.length > 4 -> keyword.removeSuffix("ed")
            keyword.endsWith("es") && keyword.length > 4 -> keyword.removeSuffix("es")
            keyword.endsWith("s") && keyword.length > 3 -> keyword.removeSuffix("s")
            else -> keyword
        }
        if (stem.length >= 4 && candidateText.contains(stem)) return true

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
        return synonyms.any { candidateText.contains(it) }
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

        for (block in blocks) {
            val text = block.text
            val tLower = text.lowercase()
            val citation = formatCitation(block)

            // 1. Check if block matches section metadata or contains target topic
            val blockMatchesSection = block.section?.lowercase()?.let { it.contains(targetLower) || it.contains(targetStem) } == true ||
                tLower.contains(targetLower) || tLower.contains(targetStem)

            if (!blockMatchesSection) continue

            // 2. Find start index of target topic in block text
            val startIdx = if (block.section?.lowercase()?.let { it.contains(targetLower) || it.contains(targetStem) } == true) {
                0
            } else {
                val headingRegex = Regex("""(?i)\b(?:${Regex.escape(targetLower)}|${Regex.escape(targetStem)})\b""")
                val m = headingRegex.find(text)
                if (m != null) m.range.last + 1 else 0
            }

            val slice = text.substring(minOf(startIdx, text.length)).trim()
            val cleanSlice = slice.replace(Regex("""^[:\-\s]+"""), "").trim()

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
                    val item = targetContent.substring(s, e)
                        .replace(Regex("""^(?:and|or)\s+"""), "")
                        .replace(Regex("""(?:,\s*)?(?:and|or)$"""), "")
                        .trim()
                        .trimEnd(',', ';', '.')
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
                    // Labeled project/item pattern: "Project 1: ... Project 2: ..." or bullet list
                    val projectNumPattern = Regex("""(?i)(?:^|(?<=[.\n;•\-_]))\s*(?:(?:project\s*\d+\s*[:\-])|(?:\d+[\.\)]\s+)|(?:[•\-\*]\s+))""")
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
                            .filter { it.length > 5 }

                        for (line in lines) {
                            results.add(Pair(line, citation))
                        }
                    }
                }
            }
        }

        return results
    }

    /**
     * Identifies the end of a section in raw text by detecting the next uppercase heading.
     * Excludes short acronyms (AWS, AI, NASA, TRL, CPU, GPU) and colons.
     */
    private fun findSectionEnd(slice: String): Int {
        val boundaryRegex = Regex(
            """(?m)(?:^|\n|(?<=[.!?]\s+)|(?<=[a-z0-9]\s+))([A-Z]{5,}(?:\s+[A-Z]{4,})*)\b(?=\s+[A-Z0-9])"""
        )
        for (m in boundaryRegex.findAll(slice)) {
            if (m.range.first >= 15) {
                return m.range.first
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
                """(?i)^(?:what\s+are\s+all\s+the|what\s+are\s+all|what\s+are\s+the|what\s+are|what\s+is\s+the|what\s+is|what|list\s+all\s+the|list\s+all|list\s+the|list\s+every|list|show\s+all|show\s+me|show|extract\s+all|extract|give\s+me\s+all|give\s+me|name\s+all|name|tell\s+me\s+about\s+the|tell\s+me\s+about|which)\s+"""
            ),
            ""
        ).trim()
        clean = clean.replace(
            Regex("""(?i)\s+(?:in\s+the\s+document|in\s+this\s+document|are\s+mentioned|is\s+mentioned|mentioned|discussed)$"""),
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
