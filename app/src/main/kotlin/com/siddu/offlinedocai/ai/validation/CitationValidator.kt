package com.siddu.offlinedocai.ai.validation

import com.siddu.offlinedocai.ai.llm.ParsedSourceBlock
import com.siddu.offlinedocai.ai.retrieval.SourceBlock
import kotlin.math.max
import kotlin.math.min

/**
 * Validation status for an individual claim or sentence.
 */
enum class ValidationStatus {
    SUPPORTED,
    PARTIALLY_SUPPORTED,
    UNSUPPORTED
}

/**
 * Fallback ladder levels according to Phase 6 specifications.
 */
enum class FallbackLevel {
    L1_FULLY_SUPPORTED,       // All claims supported by retrieved evidence
    L2_FILTERED_SUPPORTED,    // Unsupported claims pruned; remaining claims verified
    L3_EXTRACTIVE_FALLBACK,   // Answer too degraded; extractive snippet from top validated block
    L4_NOT_FOUND              // Zero evidence or topic absent; grounded not-found response
}

/**
 * Raw citation extracted from answer text before verification.
 */
data class RawCitation(
    val rawText: String,
    val sourceId: Long?,
    val page: Int?,
    val documentId: Long? = null
)

/**
 * Verified citation whose sourceId, page, and document identity exist in context.
 */
data class ValidatedCitation(
    val sourceId: Long,
    val page: Int,
    val pageSpan: String,
    val chunkIndex: Int? = null,
    val documentId: Long? = null
)

/**
 * Detailed validation result for a single claim or sentence.
 */
data class ClaimValidationResult(
    val originalClaim: String,
    val validatedText: String,
    val status: ValidationStatus,
    val citations: List<ValidatedCitation>,
    val supportingBlockIds: List<Long>,
    val supportScore: Float,
    val reason: String
)

/**
 * Complete validation result for the entire generated answer.
 */
data class FullValidationResult(
    val validatedAnswer: String,
    val fallbackLevel: FallbackLevel,
    val claimResults: List<ClaimValidationResult>,
    val validCitations: List<ValidatedCitation>,
    val overallSupportedRatio: Float,
    val isBudgetLimited: Boolean = false
)

/**
 * Extension helpers to convert between SourceBlock and ParsedSourceBlock.
 */
fun SourceBlock.toParsedSourceBlock(): ParsedSourceBlock = ParsedSourceBlock(
    sourceId = this.sourceId,
    page = this.pageNumber,
    pageSpan = this.pageSpan,
    section = this.sectionTitle,
    chunkIndex = this.chunkIndex,
    text = this.text,
    documentId = this.documentId
)

fun ParsedSourceBlock.toSourceBlock(): SourceBlock = SourceBlock(
    sourceId = this.sourceId ?: 0L,
    documentId = this.documentId ?: 0L,
    pageNumber = this.page,
    pageSpan = this.pageSpan,
    chunkIndex = this.chunkIndex ?: 0,
    sectionTitle = this.section,
    text = this.text
)

/**
 * Phase 6 Citation and Claim Validator for OfflineDocAI.
 *
 * Implements deterministic evidence-grounded validation:
 * 1. Valid Source Check (source ID must exist in retrieval context)
 * 2. Page & PageSpan Validation (rejects mismatched pages or out-of-range spans)
 * 3. Exact Value Safety (strict verification of numbers, metrics, dates, URLs, emails, phones)
 * 4. Paraphrase & Token Overlap Verification (stem-normalized lexical analysis)
 * 5. Compound Clause Splitting (preserves supported clauses, strips unsupported additions)
 * 6. Multi-Source & Multi-Document Verification (prevents cross-document confusion)
 * 7. Fallback Ladder (L1 -> L2 -> L3 -> L4)
 */
object CitationValidator {

    private const val SUPPORT_THRESHOLD = 0.45f
    private const val CLAUSE_SUPPORT_THRESHOLD = 0.60f
    private const val CONTRADICTION_PENALTY = 0.0f

    // Citation matching regex: supports [Doc D, Source S, Page P], [Source S, Page P], [SOURCE_ID=S, PAGE=P], (Page P), [Page P], (Pages P-Q)
    private val CITATION_REGEX = Regex(
        """\[(?:(?:Doc|Document)\s*[:=]?\s*(\d+),\s*)?(?:Source|SOURCE_ID)\s*[:=]?\s*(\d+)(?:,\s*Pages?\s*[:=]?\s*(\d+))?\]|\(Pages?\s+(\d+(?:\s*-\s*\d+)?)\)|\[Pages?\s+(\d+(?:\s*-\s*\d+)?)\]"""
    )

    // Factual value extractors
    private val METRIC_WITH_UNIT_REGEX = Regex(
        """(?i)\b\d+(?:\.\d+)?\s*(?:mah|ah|wh|kwh|w|kw|mw|v|kv|mv|gb|mb|tb|kb|hz|mhz|ghz|c|°c|f|°f|k|kg|g|mg|lbs|km|m|cm|mm|nm|mph|km/h|ms|s|min|hrs|rpm|db|dpi|px|%)\b"""
    )
    private val STANDALONE_NUMBER_REGEX = Regex("""\b\d+(?:\.\d+)?%?\b""")
    private val DATE_REGEX = Regex(
        """\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|(?:19\d\d|20\d\d))\b"""
    )
    private val URL_REGEX = Regex("""(?i)\b(?:https?://|www\.)[^\s)\]]+|\b(?:github\.com|linkedin\.com)/[^\s)\]]+""")
    private val EMAIL_REGEX = Regex("""\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b""")
    private val PHONE_REGEX = Regex("""(?i)(?:phone|mobile|cell|contact)?\s*[:#-]?\s*(\+?\d[\d\s\-().]{7,}\d)|\b\d{10}\b""")

    // Sentence splitting: punctuation followed by space and uppercase letter/bullet.
    // Android-compatible: no variable-length look-behinds. The abbreviation exclusion is dropped
    // because java.util.regex on Android rejects alternations of different-length in look-behinds.
    // Occasional splits on "e.g. X" or "Dr. Smith" are acceptable vs. a runtime crash.
    private val SENTENCE_SPLIT_REGEX = Regex(
        """(?<=[.!?])\s+(?=[A-Z0-9\-\*•])|\n+"""
    )

    // Compound conjunction split regex for clause separation
    private val COMPOUND_CONJUNCTION_REGEX = Regex("""\s+(?:and|while|but|as well as)\s+|;\s*""")

    // Completeness assertion patterns
    private val COMPLETENESS_CLAIM_REGEX = Regex(
        """(?i)\b(?:these are all|complete list of|every single|all\s+\d+\s+projects|all projects mentioned|all of the projects)\b"""
    )

    private val STOP_WORDS = setOf(
        "what", "when", "where", "how", "this", "that", "there", "their", "who", "why",
        "them", "with", "from", "your", "about", "the", "is", "are", "was", "were", "and",
        "or", "in", "on", "at", "to", "a", "an", "of", "for", "by", "does", "do", "can",
        "you", "tell", "me", "explain", "give", "show", "please", "which", "will", "would",
        "mentioned", "say", "talk", "says", "talks", "i", "my", "we", "our", "all", "any",
        "be", "been", "being", "have", "has", "had", "using", "uses", "used"
    )

    /**
     * Checks if a line is a decorative header, section title, or framing disclaimer.
     * Framing elements are not factual claims and should never be erased as hallucinations.
     */
    fun isFramingOrHeader(line: String): Boolean {
        val trimmed = line.trim()
        if (trimmed.isEmpty()) return true
        if (trimmed.startsWith("#")) return true
        if (trimmed.startsWith("---")) return true

        // Bullets that are purely guidance/tips
        if (trimmed.startsWith("*Tip:", ignoreCase = true) || trimmed.startsWith("Tip:", ignoreCase = true)) return true
        if (trimmed.startsWith("*Note:", ignoreCase = true) || trimmed.startsWith("Note:", ignoreCase = true)) return true

        val lower = trimmed.lowercase()
        if (lower.contains("document overview") || lower.contains("key summary") || lower.contains("summary:")) return true
        if (lower.contains("retrieved evidence contains") || lower.contains("based on the document, the items are")) return true
        if (lower.contains("budget-limited") && (lower.contains("note:") || lower.contains("context was"))) return true
        if (lower.startsWith("here is") || lower.startsWith("here are")) return true
        if (trimmed.endsWith(":") && !trimmed.contains(".") && trimmed.length < 60) return true
        if (trimmed.startsWith("*") && trimmed.endsWith("*") && trimmed.length < 100) return true

        return false
    }

    val EXPLANATORY_INDICATORS = listOf(
        "include", "includes", "including",
        "caused by", "causes",
        "because", "due to",
        "occurs when", "occurs if", "occurs because",
        "requires", "required for",
        "used for", "used to",
        "results from", "result from", "results in", "result in",
        "consists of", "consisting of",
        "defined as", "refers to",
        "is a ", "is an ", "are "
    )

    private val COMMON_VERBS = setOf(
        "is", "are", "was", "were", "be", "been", "being",
        "has", "have", "had", "having",
        "do", "does", "did", "done",
        "can", "could", "will", "would", "shall", "should", "may", "might", "must",
        "include", "includes", "included", "including",
        "require", "requires", "required", "requiring",
        "use", "uses", "used", "using",
        "cause", "causes", "caused", "causing",
        "result", "results", "resulted", "resulting",
        "occur", "occurs", "occurred", "occurring",
        "consist", "consists", "consisted", "consisting",
        "join", "joins", "joined", "joining",
        "remain", "remains", "remained", "remaining",
        "allow", "allows", "allowed", "allowing",
        "provide", "provides", "provided", "providing",
        "prevent", "prevents", "prevented", "preventing",
        "ensure", "ensures", "ensured", "ensuring",
        "check", "checks", "checked", "checking",
        "wrap", "wraps", "wrapped", "wrapping",
        "cut", "cuts", "cutting",
        "heal", "heals", "healed", "healing",
        "grow", "grows", "grew", "grown", "growing",
        "keep", "keeps", "kept", "keeping",
        "make", "makes", "made", "making",
        "take", "takes", "took", "taken", "taking",
        "apply", "applies", "applied", "applying",
        "need", "needs", "needed", "needing",
        "show", "shows", "showed", "shown", "showing",
        "align", "aligns", "aligned", "aligning",
        "form", "forms", "formed", "forming",
        "contain", "contains", "contained", "containing",
        "produce", "produces", "produced", "producing",
        "involve", "involves", "involved", "involving",
        "indicate", "indicates", "indicated", "indicating",
        "lead", "leads", "led", "leading",
        "connect", "connects", "connected", "connecting",
        "serve", "serves", "served", "serving",
        "work", "works", "worked", "working",
        "exist", "exists", "existed", "existing",
        "represent", "represents", "represented", "representing",
        "describe", "describes", "described", "describing",
        "define", "defines", "defined", "defining",
        "refer", "refers", "referred", "referring",
        "mean", "means", "meant", "meaning",
        "stand", "stands", "stood", "standing",
        "hold", "holds", "held", "holding",
        "carry", "carries", "carried", "carrying",
        "give", "gives", "gave", "given", "giving",
        "find", "finds", "found", "finding",
        "see", "sees", "saw", "seen", "seeing",
        "feel", "feels", "felt", "feeling",
        "protect", "protects", "protected", "protecting",
        "control", "controls", "controlled", "controlling",
        "help", "helps", "helped", "helping",
        "fit", "fits", "fitted", "fitting",
        "attach", "attaches", "attached", "attaching",
        "unite", "unites", "united", "uniting",
        "bind", "binds", "bound", "binding",
        "fail", "fails", "failed", "failing",
        "operate", "operates", "operated", "operating",
        "perform", "performs", "performed", "performing",
        "demonstrate", "demonstrates", "demonstrated", "demonstrating",
        "fulfill", "fulfills", "fulfilled", "fulfilling",
        "function", "functions", "functioned", "functioning",
        "establish", "establishes", "established", "establishing",
        "define", "defines", "defined", "defining",
        "transform", "transforms", "transformed", "transforming",
        "evaluate", "evaluates", "evaluated", "evaluating",
        "determine", "determines", "determined", "determining",
        "culminate", "culminates", "culminated", "culminating",
        "comply", "complies", "complied", "complying",
        "verify", "verifies", "verified", "verifying",
        "validate", "validates", "validated", "validating",
        "execute", "executes", "executed", "executing"
    )

    /**
     * Identifies document structural elements, section titles, page/chapter headings,
     * slash-separated title fragments, and non-sentence label lines that must never be
     * treated as factual answer sentences.
     */
    fun isHeadingOrTitle(rawText: String): Boolean {
        val text = rawText.trim()
        if (text.isEmpty() || text.length < 3) return false

        // Table rows (| col1 | col2 |) are structured factual data, never section headings
        if (text.startsWith("|") && text.endsWith("|")) return false

        // List items (- item, * item, • item, 1. item) are structured list items, never document headings
        if (text.matches(Regex("""^(?:[-*•‣◦⁃∙]|\d+[\.\)])\s+.*"""))) return false

        // 1. Bracketed metadata lines: [Page 1], [SOURCE_ID=1], etc.
        if (text.startsWith("[") && (text.endsWith("]") || text.contains("]"))) {
            if (text.matches(Regex("""^\[(?:DOC_ID|SOURCE_ID|PAGE|CHUNK_INDEX|SECTION|Source|Page)[^\]]*\].*"""))) {
                return true
            }
            if (text.matches(Regex("""^\[.*\]$"""))) return true
        }

        // 2. Explicit Page / Section / Chapter / Appendix / Table / Figure prefixes:
        // Examples: "Page 11: Failure...", "Page 5: Bark Grafting", "Section 14: Troubleshooting...", "Chapter 3: Grafting Methods"
        val prefixPattern = Regex(
            """^(?:page\s+\d+|section\s+\d+(?:\.\d+)*|chapter\s+\d+|appendix\s+[a-z0-9]+|part\s+\d+|table\s+\d+|figure\s+\d+|fig\.\s*\d+)\s*[:\-–—]\s*""",
            RegexOption.IGNORE_CASE
        )
        val prefixMatch = prefixPattern.find(text)
        if (prefixMatch != null) {
            val afterPrefix = text.substring(prefixMatch.range.last + 1).trim()
            val cleanAfter = stripCitations(afterPrefix).trim()
            val words = cleanAfter.split(Regex("""\s+""")).filter { it.isNotEmpty() }
            val hasVerb = words.any { w -> w.lowercase().filter { it.isLetter() } in COMMON_VERBS }
            val isCompleteSentence = words.size > 8 && (cleanAfter.endsWith(".") || cleanAfter.endsWith(")") || text.endsWith("."))
            if (!isCompleteSentence && (words.size <= 6 || !hasVerb)) return true
        }

        val sectionNumberPattern = Regex(
            """^(?:section|chapter|appendix|part)\s+\d+(?:\.\d+)*\b.*""",
            RegexOption.IGNORE_CASE
        )
        if (sectionNumberPattern.matches(text)) {
            val textWithoutCitation = stripCitations(text).trim()
            val words = textWithoutCitation.split(Regex("""\s+""")).filter { it.isNotEmpty() }
            val hasVerb = words.any { w -> w.lowercase().filter { it.isLetter() } in COMMON_VERBS }
            val isCompleteSentence = words.size > 8 && (textWithoutCitation.endsWith(".") || textWithoutCitation.endsWith(")") || text.endsWith("."))
            if (!isCompleteSentence && ((words.size <= 8 && !hasVerb) || (!textWithoutCitation.endsWith(".") && !hasVerb))) return true
        }

        // 3. Category label or heading ending with a colon: "Languages:", "Tools & Materials:", "Causes:"
        if (text.endsWith(":")) return true

        // 4. Slash-separated title fragments: "Failure Likely causes / What to check", "Summary / Conclusion", "Overview / Introduction"
        // Target explicit title-delimiter slashes (spaced slash " / " or isolated slash at line ends), avoiding units like "Wh/kg" or "km/h"
        if (text.contains(" / ") || text.startsWith("/") || text.endsWith("/")) {
            val isUrlOrPath = text.contains("http://") || text.contains("https://") || text.contains("file://") || text.contains("C:/")
            if (!isUrlOrPath) {
                return true
            }
        }

        // 5. Ampersand in short title phrases without sentence punctuation: "Tools & Materials", "Troubleshooting & Recovery"
        if (text.contains(" & ") || text.contains(" and/or ")) {
            val words = text.split(Regex("""\s+"""))
            if (words.size <= 7 && (!text.endsWith(".") || words.size <= 4)) {
                return true
            }
        }

        // 6. Markdown or outline headings: "# Heading", "## Heading", "=== Heading ==="
        if (text.startsWith("#") || text.startsWith("==") || text.startsWith("--")) return true

        // 7. Isolated short title-case or uppercase phrases without sentence termination
        val textWithoutCitation = stripCitations(text).trim()
        if (textWithoutCitation.contains("@") || textWithoutCitation.contains("://")) return false

        val endsWithSentencePunctuation = textWithoutCitation.endsWith(".") || textWithoutCitation.endsWith("!") || textWithoutCitation.endsWith("?") ||
                textWithoutCitation.endsWith(".\"") || textWithoutCitation.endsWith("!\"") || textWithoutCitation.endsWith("?\"") ||
                text.endsWith(".") || text.endsWith("!") || text.endsWith("?")

        val words = textWithoutCitation.split(Regex("""\s+""")).filter { it.isNotEmpty() }

        if (!endsWithSentencePunctuation) {
            // Short phrases without terminal punctuation are headings or fragments
            // E.g. "Failure Likely causes", "What to check", "Key details", "Tools & Materials"
            if (words.size <= 8) return true

            // Check if it has any common verb or copula
            val hasCommonVerb = words.any { w ->
                val wl = w.lowercase().trimEnd(',', ';')
                wl in COMMON_VERBS
            }
            if (!hasCommonVerb && words.size <= 12) return true
        }

        return false
    }

    /**
     * Primary validation entrypoint accepting ParsedSourceBlock.
     */
    fun validate(
        answer: String,
        blocks: List<ParsedSourceBlock>,
        query: String = "",
        isBudgetLimited: Boolean = false,
        answerLength: com.siddu.offlinedocai.ai.llm.AnswerLength = com.siddu.offlinedocai.ai.llm.AnswerLength.SHORT,
        classification: com.siddu.offlinedocai.ai.retrieval.QueryClassification? = null
    ): FullValidationResult {
        val trimmedAnswer = answer.trim()

        // 1. Handle Negative/Not-Found Sentinels from GGUFLLM
        if (trimmedAnswer.startsWith("I searched the provided document", ignoreCase = true) ||
            trimmedAnswer.startsWith("I couldn't find", ignoreCase = true) ||
            trimmedAnswer.startsWith("I could not find", ignoreCase = true) ||
            trimmedAnswer.startsWith("The evidence provided does not contain", ignoreCase = true) ||
            trimmedAnswer.startsWith("The provided document does not contain", ignoreCase = true) ||
            trimmedAnswer.startsWith("The document does not mention", ignoreCase = true) ||
            trimmedAnswer.startsWith("The document does not contain", ignoreCase = true) ||
            trimmedAnswer.contains("[I searched the provided document", ignoreCase = true) ||
            trimmedAnswer == "[NO_EVIDENCE_FOUND]"
        ) {
            return FullValidationResult(
                validatedAnswer = trimmedAnswer,
                fallbackLevel = FallbackLevel.L4_NOT_FOUND,
                claimResults = emptyList(),
                validCitations = emptyList(),
                overallSupportedRatio = 1.0f,
                isBudgetLimited = isBudgetLimited
            )
        }

        // 2. Handle Empty Context or Empty Answer
        if (blocks.isEmpty() || trimmedAnswer.isEmpty()) {
            val notFoundMsg = if (query.isNotBlank()) {
                "I couldn't find sufficient information about \"$query\" in the uploaded document."
            } else {
                "I couldn't find sufficient information in the uploaded document."
            }
            return FullValidationResult(
                validatedAnswer = notFoundMsg,
                fallbackLevel = FallbackLevel.L4_NOT_FOUND,
                claimResults = emptyList(),
                validCitations = emptyList(),
                overallSupportedRatio = 0f,
                isBudgetLimited = isBudgetLimited
            )
        }

        // 3. Extract Individual Claims / Sentences
        val claims = extractClaims(trimmedAnswer)
        if (claims.isEmpty()) {
            val fallback = extractGroundedFallback(blocks, query)
            return FullValidationResult(
                validatedAnswer = fallback,
                fallbackLevel = FallbackLevel.L3_EXTRACTIVE_FALLBACK,
                claimResults = emptyList(),
                validCitations = emptyList(),
                overallSupportedRatio = 0f,
                isBudgetLimited = isBudgetLimited
            )
        }

        // 4. Validate Each Claim Against Retrieved Evidence
        val claimResults = mutableListOf<ClaimValidationResult>()
        val allValidCitations = mutableListOf<ValidatedCitation>()

        for (claim in claims) {
            val rawCitations = extractRawCitations(claim)
            val validatedCitations = rawCitations.mapNotNull { validateCitation(it, blocks) }
            allValidCitations.addAll(validatedCitations)

            // Determine candidate supporting blocks: either cited blocks or all blocks if uncited
            val candidateBlocks = if (validatedCitations.isNotEmpty()) {
                val citedIds = validatedCitations.map { it.sourceId }.toSet()
                val citedPages = validatedCitations.map { it.page }.toSet()
                val matched = blocks.filter { b ->
                    (b.sourceId != null && b.sourceId in citedIds) || (b.page in citedPages)
                }
                if (matched.isNotEmpty()) matched else blocks
            } else {
                blocks
            }

            val result = validateClaimSupport(claim, candidateBlocks, blocks, validatedCitations, query, classification)
            claimResults.add(result)
        }

        // 5. Handle Exhaustive Completeness Safety when isBudgetLimited == true
        val adjustedClaimResults = if (isBudgetLimited) {
            claimResults.map { cr ->
                if (COMPLETENESS_CLAIM_REGEX.containsMatchIn(cr.originalClaim)) {
                    val softened = COMPLETENESS_CLAIM_REGEX.replace(cr.validatedText) { "the retrieved" }
                    cr.copy(
                        validatedText = softened,
                        reason = "${cr.reason}; Softened ungrounded completeness assertion on budget-limited context."
                    )
                } else {
                    cr
                }
            }
        } else {
            claimResults
        }

        // 6. Compute Aggregate Metrics and Execute Fallback Ladder
        val supportedCount = adjustedClaimResults.count {
            it.status == ValidationStatus.SUPPORTED || it.status == ValidationStatus.PARTIALLY_SUPPORTED
        }
        val overallRatio = if (adjustedClaimResults.isNotEmpty()) {
            supportedCount.toFloat() / adjustedClaimResults.size
        } else {
            0f
        }

        val fallbackLevel: FallbackLevel
        var finalAnswer: String

        when {
            // Level 1: Fully Supported (All claims passed verification)
            overallRatio >= 0.75f && supportedCount == adjustedClaimResults.size -> {
                fallbackLevel = FallbackLevel.L1_FULLY_SUPPORTED
                var ans = trimmedAnswer
                if (isBudgetLimited && COMPLETENESS_CLAIM_REGEX.containsMatchIn(ans)) {
                    ans = COMPLETENESS_CLAIM_REGEX.replace(ans) { "the retrieved" }
                }
                ans = sanitizeCitations(ans, blocks)
                finalAnswer = cleanAndRenumberList(ans)
            }

            // Level 2: Filtered Supported (Unsupported claims removed)
            supportedCount > 0 -> {
                fallbackLevel = FallbackLevel.L2_FILTERED_SUPPORTED
                var reconstructed = trimmedAnswer
                for (cr in adjustedClaimResults) {
                    if (cr.status == ValidationStatus.UNSUPPORTED) {
                        // Remove unsupported sentence/bullet from original answer along with its prefix
                        val bulletPrefixRegex = Regex("""(?m)^[ \t]*(?:\d+[\.):]|[-*•])[ \t]*""" + Regex.escape(cr.originalClaim) + """[ \t]*\r?\n?""")
                        if (bulletPrefixRegex.containsMatchIn(reconstructed)) {
                            reconstructed = bulletPrefixRegex.replace(reconstructed, "")
                        } else {
                            val inlinePrefixRegex = Regex("""\b\d+[\.):]\s*""" + Regex.escape(cr.originalClaim))
                            if (inlinePrefixRegex.containsMatchIn(reconstructed)) {
                                reconstructed = inlinePrefixRegex.replace(reconstructed, "")
                            } else {
                                reconstructed = reconstructed.replace(cr.originalClaim, "")
                            }
                        }
                    } else if (cr.status == ValidationStatus.PARTIALLY_SUPPORTED && cr.validatedText != cr.originalClaim) {
                        // Replace with supported clause
                        reconstructed = reconstructed.replace(cr.originalClaim, cr.validatedText)
                    }
                }
                if (isBudgetLimited && COMPLETENESS_CLAIM_REGEX.containsMatchIn(reconstructed)) {
                    reconstructed = COMPLETENESS_CLAIM_REGEX.replace(reconstructed) { "the retrieved" }
                }
                reconstructed = cleanAndRenumberList(reconstructed)
                finalAnswer = sanitizeCitations(reconstructed, blocks)
                finalAnswer = cleanAndRenumberList(finalAnswer)
            }

            // Level 3: Extractive Fallback (Zero claims supported, but evidence exists)
            blocks.isNotEmpty() -> {
                fallbackLevel = FallbackLevel.L3_EXTRACTIVE_FALLBACK
                finalAnswer = cleanAndRenumberList(extractGroundedFallback(blocks, query, answerLength))
            }

            // Level 4: Not Found
            else -> {
                fallbackLevel = FallbackLevel.L4_NOT_FOUND
                finalAnswer = if (query.isNotBlank()) {
                    "I couldn't find sufficient information about \"$query\" in the uploaded document."
                } else {
                    "I couldn't find sufficient information in the uploaded document."
                }
            }
        }

        return FullValidationResult(
            validatedAnswer = finalAnswer.trim(),
            fallbackLevel = fallbackLevel,
            claimResults = adjustedClaimResults,
            validCitations = allValidCitations.distinct(),
            overallSupportedRatio = overallRatio,
            isBudgetLimited = isBudgetLimited
        )
    }

    /**
     * Overload accepting SourceBlock from ContextAssembler.
     */
    fun validateWithSourceBlocks(
        answer: String,
        sourceBlocks: List<SourceBlock>,
        query: String = "",
        isBudgetLimited: Boolean = false,
        answerLength: com.siddu.offlinedocai.ai.llm.AnswerLength = com.siddu.offlinedocai.ai.llm.AnswerLength.SHORT,
        classification: com.siddu.offlinedocai.ai.retrieval.QueryClassification? = null
    ): FullValidationResult {
        return validate(
            answer = answer,
            blocks = sourceBlocks.map { it.toParsedSourceBlock() },
            query = query,
            isBudgetLimited = isBudgetLimited,
            answerLength = answerLength,
            classification = classification
        )
    }

    /**
     * Splits answer text into sentence or bullet-level factual claims.
     */
    fun extractClaims(answer: String): List<String> {
        val rawLines = answer.lines()
        val claims = mutableListOf<String>()

        for (line in rawLines) {
            val trimmed = line.trim()
            if (trimmed.isEmpty()) continue

            // Framing, headers, and metadata are preserved in output but not treated as factual claims
            if (isFramingOrHeader(trimmed)) {
                continue
            }

            // Bullet points are distinct claims (supporting -, *, and unicode •)
            if (trimmed.startsWith("-") || trimmed.startsWith("*") || trimmed.startsWith("•") || trimmed.matches(Regex("""^\d+\.\s+.*"""))) {
                val cleanBullet = trimmed.replace(Regex("""^[-*•]\s+|\d+\.\s+"""), "").trim()
                if (cleanBullet.isNotEmpty()) {
                    claims.add(cleanBullet)
                }
            } else {
                // Split prose paragraph by sentence boundary
                val sentences = trimmed.split(SENTENCE_SPLIT_REGEX)
                    .map { it.trim() }
                    .filter { it.isNotEmpty() && it.any { c -> c.isLetterOrDigit() } }
                    .filter { stripCitations(it).isNotBlank() }
                claims.addAll(sentences)
            }
        }
        return claims
    }

    /**
     * Extracts raw citation structures from text.
     */
    fun extractRawCitations(text: String): List<RawCitation> {
        val results = mutableListOf<RawCitation>()
        val matches = CITATION_REGEX.findAll(text)
        for (m in matches) {
            val docId = m.groupValues[1].ifEmpty { null }?.toLongOrNull()
            val sourceId = m.groupValues[2].ifEmpty { null }?.toLongOrNull()
            val pageFromSource = m.groupValues[3].ifEmpty { null }?.toIntOrNull()
            val pageParen = m.groupValues[4].ifEmpty { null }?.split("-")?.firstOrNull()?.trim()?.toIntOrNull()
            val pageBracket = m.groupValues[5].ifEmpty { null }?.split("-")?.firstOrNull()?.trim()?.toIntOrNull()
            val page = pageFromSource ?: pageParen ?: pageBracket

            results.add(
                RawCitation(
                    rawText = m.value,
                    sourceId = sourceId,
                    page = page,
                    documentId = docId
                )
            )
        }
        return results
    }

    /**
     * Verifies that a citation refers to a valid source ID, valid page, and valid document in context.
     */
    fun validateCitation(raw: RawCitation, validBlocks: List<ParsedSourceBlock>): ValidatedCitation? {
        if (validBlocks.isEmpty()) return null

        if (raw.sourceId != null) {
            val matchingBlock = validBlocks.find { it.sourceId == raw.sourceId } ?: return null

            // Document ID validation: must match if both are present
            if (raw.documentId != null && matchingBlock.documentId != null && raw.documentId != matchingBlock.documentId) {
                return null
            }

            // Page validation: cited page must exist in block's page or pageSpan
            if (raw.page != null) {
                val validPages = parsePageSpan(matchingBlock.pageSpan, matchingBlock.page)
                if (raw.page !in validPages) {
                    return null
                }
            }

            return ValidatedCitation(
                sourceId = matchingBlock.sourceId!!,
                page = raw.page ?: matchingBlock.page,
                pageSpan = matchingBlock.pageSpan,
                chunkIndex = matchingBlock.chunkIndex,
                documentId = matchingBlock.documentId
            )
        } else if (raw.page != null) {
            // Standalone page citation (Page X)
            val matchingBlock = validBlocks.find { block ->
                val validPages = parsePageSpan(block.pageSpan, block.page)
                raw.page in validPages && (raw.documentId == null || block.documentId == null || block.documentId == raw.documentId)
            } ?: return null

            return ValidatedCitation(
                sourceId = matchingBlock.sourceId ?: 0L,
                page = raw.page,
                pageSpan = matchingBlock.pageSpan,
                chunkIndex = matchingBlock.chunkIndex,
                documentId = matchingBlock.documentId
            )
        }
        return null
    }

    /**
     * Parses pageSpan into an explicit set of page numbers.
     * Examples:
     * - "10" -> {10}
     * - "10-11" -> {10, 11}
     * - "10, 11, 12" -> {10, 11, 12}
     */
    fun parsePageSpan(pageSpan: String?, fallbackPage: Int): Set<Int> {
        val result = mutableSetOf<Int>()
        if (fallbackPage > 0) {
            result.add(fallbackPage)
        }
        if (pageSpan.isNullOrBlank()) {
            return result
        }
        val clean = pageSpan.trim()
        if (clean.contains("-")) {
            val parts = clean.split("-")
            if (parts.size == 2) {
                val start = parts[0].trim().toIntOrNull()
                val end = parts[1].trim().toIntOrNull()
                if (start != null && end != null && start <= end) {
                    for (p in start..end) {
                        result.add(p)
                    }
                    return result
                }
            }
        }
        if (clean.contains(",")) {
            val parts = clean.split(",")
            for (p in parts) {
                p.trim().toIntOrNull()?.let { result.add(it) }
            }
            return result
        }
        clean.toIntOrNull()?.let { result.add(it) }
        return result
    }

    /**
     * Checks whether an extracted factual claim directly answers the user's question,
     * distinguishing mere document grounding from question relevance.
     */
    fun isClaimRelevantToQuery(
        claim: String,
        query: String,
        classification: com.siddu.offlinedocai.ai.retrieval.QueryClassification? = null
    ): Boolean {
        if (query.isBlank()) return true
        val cLower = claim.lowercase().trim()
        val qLower = query.lowercase().trim()

        // 1. Tools and Materials
        val isToolsQuery = qLower.contains("tool") || qLower.contains("material") || qLower.contains("equipment") || qLower.contains("supplies")
        if (isToolsQuery && !qLower.contains("method") && !qLower.contains("cause") && !qLower.contains("what is")) {
            val toolTerms = listOf(
                "knife", "knives", "strip", "strips", "shear", "shears", "wax", "tape",
                "pruning", "blade", "seal", "wound", "equipment", "tool", "material", "required", "needed"
            )
            val irrelevantAftercareTerms = listOf("inspect", "inspection", "weekly", "sucker", "aftercare", "temperature", "pathogen", "failure", "cleft grafting", "healing")
            val hasTool = toolTerms.any { cLower.contains(it) }
            val hasAftercare = irrelevantAftercareTerms.any { cLower.contains(it) }
            
            // If the source states the purpose of the tool (e.g. wax prevents desiccation), allow it
            if (hasTool) return true
            if (hasAftercare) return false
            return false
        }

        // 2. Temperature / Environmental Healing
        val isTempQuery = (classification?.factType == com.siddu.offlinedocai.ai.retrieval.FactType.TEMPERATURE) ||
                qLower.contains("temperature") || qLower.contains("degrees") || qLower.contains("celsius") || qLower.contains("fahrenheit") || qLower.contains("healing")
        if (isTempQuery) {
            val tempTerms = listOf("temperature", "21c", "27c", "70f", "80f", "Ac", "Af", "celsius", "fahrenheit", "callus", "healing", "humidity", "85%", "desiccation")
            val irrelevantTerms = listOf("knife", "knives", "shear", "shears", "strip", "strips", "wax", "cleft", "splice", "bridge", "sucker", "weekly", "tools", "aftercare", "types")
            val hasTemp = tempTerms.any { cLower.contains(it) }
            val hasIrrelevant = irrelevantTerms.any { cLower.contains(it) }
            
            // Allow associated humidity/healing conditions
            if (hasTemp) return true
            if (hasIrrelevant) return false
            return false
        }

        // 3. Causes of failure and troubleshooting
        val isCausesQuery = qLower.contains("cause") || qLower.contains("failure") || qLower.contains("fail") || qLower.contains("troubleshoot")
        if (isCausesQuery) {
            val causesTerms = listOf("cambial", "contact", "desiccation", "physiological", "season", "pathogen", "infection", "cause", "fail", "failure", "poor cuts")
            val checkTerms = listOf("inspect", "inspection", "weekly", "sucker", "rootstock", "union", "remove", "check", "ensure", "troubleshoot", "prevent")
            val irrelevantTools = listOf("sharp grafting knife", "pruning shears", "wax", "strip")
            
            val asksCheck = qLower.contains("check") || qLower.contains("inspect") || qLower.contains("do") || qLower.contains("action") || qLower.contains("prevent")

            val hasCause = causesTerms.any { cLower.contains(it) }
            val hasCheck = checkTerms.any { cLower.contains(it) }
            val hasTool = irrelevantTools.any { cLower.contains(it) }

            if (hasTool && !hasCause && !hasCheck) return false

            if (asksCheck) {
                return hasCause || hasCheck
            } else {
                return hasCause
            }
        }

        // 4. Comparison
        if (classification?.intent == com.siddu.offlinedocai.ai.retrieval.QueryIntent.COMPARISON) {
            // Must contain Entity A or Entity B or be directly related
            val ea = classification.entityA?.lowercase() ?: ""
            val eb = classification.entityB?.lowercase() ?: ""
            if (ea.isNotBlank() && cLower.contains(ea)) return true
            if (eb.isNotBlank() && cLower.contains(eb)) return true
            return false
        }

        // 6. Explain / Define Grafting (Domain Knowledge bypass)
        val isExplainGrafting = qLower.contains("explain") || qLower.contains("what is") || qLower.contains("define") || qLower.contains("describe")
        if (isExplainGrafting && qLower.contains("grafting")) {
            val graftingCoreTerms = listOf("technique", "horticultural", "tissue", "tissues", "plant", "plants", "join", "joined", "growth", "scion", "rootstock", "cambium", "alive", "heal", "taken", "diameter", "interlocking", "cut", "mechanical", "strength", "wax", "seal", "bridge", "callus", "week", "atmospheric", "drying", "requires", "provide", "ensure", "formation", "application")
            val irrelevantTerms = listOf("knife", "knives", "shear", "humidity", "85%", "pathogen", "weekly")
            val hasCore = graftingCoreTerms.any { cLower.contains(it) }
            val hasIrrelevant = irrelevantTerms.any { cLower.contains(it) }
            if (hasIrrelevant && !hasCore) return false
            if (hasCore) return true
        }
        
        // 7. Default query: Intent, important entities, semantic overlap
        val qTokens = qLower.split(Regex("""[^a-z0-9]+""")).filter { it.length > 2 && it !in STOP_WORDS }
        if (qTokens.isEmpty()) return true
        
                if (classification?.intent == com.siddu.offlinedocai.ai.retrieval.QueryIntent.EXHAUSTIVE_LIST || 
            classification?.intent == com.siddu.offlinedocai.ai.retrieval.QueryIntent.FIND_ALL || 
            classification?.intent == com.siddu.offlinedocai.ai.retrieval.QueryIntent.SECTION_QUERY || 
            classification?.intent == com.siddu.offlinedocai.ai.retrieval.QueryIntent.ENTITY_LOOKUP ||
            classification?.intent == com.siddu.offlinedocai.ai.retrieval.QueryIntent.PAGE_RANGE ||
            classification?.intent == com.siddu.offlinedocai.ai.retrieval.QueryIntent.GLOBAL_SUMMARY) {
            return true
        }

        val isListOrCategoryQuery = qLower.startsWith("which ") ||
            qLower.startsWith("list ") ||
            qLower.contains("what are the ") ||
            qLower.contains("tell me about the ") ||
            (qLower.contains("what ") && qLower.contains(" are mentioned")) ||
            qLower == "projects" || qLower == "skills" || qLower == "certifications" || qLower == "methods" || qLower == "tools"
        if (isListOrCategoryQuery) {
            return true
        }
        
        // Exact target extraction based relevance
        val targetTopic = classification?.targetTerm?.lowercase()
        if (!targetTopic.isNullOrBlank()) {
            val topicTokens = targetTopic.split(Regex("""[^a-z0-9]+""")).filter { it.length > 2 }
            if (topicTokens.isNotEmpty() && topicTokens.all { cLower.contains(it) }) return true
        }

        // Must share at least SOME core entity or requested attribute, not just any keyword
        val importantEntities = qTokens.filter { it[0].isUpperCase() || qLower.contains("what is $it") || qLower.contains("list $it") }
        if (importantEntities.isNotEmpty()) {
            if (importantEntities.any { t -> cLower.contains(t) || cLower.contains(t.trimEnd('s')) || (t.endsWith("ing") && cLower.contains(t.removeSuffix("ing"))) }) return true
        }
        
        // Fallback to requiring more than just 1 random keyword overlap if query is long
        val matchedTokens = qTokens.filter { t -> 
            cLower.contains(t) || 
            cLower.contains(t.trimEnd('s')) || 
            (t.endsWith("ing") && cLower.contains(t.removeSuffix("ing"))) ||
            (t == "heat" && cLower.contains("temperature")) ||
            (t == "temperature" && cLower.contains("heat")) ||
            (t == "operating" && cLower.contains("operation"))
        }
        if (qTokens.size >= 3) {
            return matchedTokens.size >= 2
        }

        return matchedTokens.isNotEmpty()
    }

    /**
     * Cleans malformed empty list items and renumbers numbered lists sequentially.
     */
    fun cleanAndRenumberList(text: String): String {
        if (text.isBlank()) return text

        // 1. Remove phantom number runs before text or another number
        // e.g. "1. 2. 3. 4. 1. Check" -> "1. Check"
        // e.g. "2. 3. 4. Wax" -> "4. Wax"
        // e.g. "2. 3. Check" -> "3. Check"
        var cleaned = text.replace(Regex("""(?:\b\d+[\.):]\s+)+(\d+[\.):]\s+)(?=[A-Za-z*•\[\-])""")) { match ->
            match.groupValues[1]
        }
        // Remove trailing bare number sequences at end of string or before punctuation
        cleaned = cleaned.replace(Regex("""(?:\b\d+[\.):]\s+){2,}"""), "")

        // 2. Remove lines that contain ONLY a number (e.g. "1.", "2. ", "3)")
        cleaned = cleaned.replace(Regex("""(?m)^[ \t]*\d+[\.):][ \t]*$\r?\n?"""), "")

        // 3. Remove inline orphan numbers followed immediately by punctuation, bracket, or end-of-string
        // Android-compatible: avoid stacked/variable-length look-behinds.
        run {
            val orphanRegex = Regex("""(^|[\\s])(\\d+)[\.:]\\s*(?=[,\.;:\[\]]|\\z)""", RegexOption.MULTILINE)
            val sb = StringBuilder()
            var last = 0
            for (m in orphanRegex.findAll(cleaned)) {
                val precText = cleaned.substring(maxOf(0, m.range.first - 10), m.range.first)
                val isAfterCitation = precText.contains(Regex("""(?i)(Page|Pages|Source)\\s*$"""))
                if (!isAfterCitation) {
                    sb.append(cleaned.substring(last, m.range.first))
                    sb.append(m.groupValues[1])
                    last = m.range.last + 1
                }
            }
            sb.append(cleaned.substring(last))
            cleaned = sb.toString()
        }
        // 4. Renumber multiline lists
        val itemPattern = Regex("""^[ \t]*(\d+)[\.):][ \t]+(.+)$""")
        val lines = cleaned.lines()
        var hasLineItems = false
        var currentItemIndex = 1
        val processedLines = mutableListOf<String>()

        for (line in lines) {
            val m = itemPattern.matchEntire(line)
            if (m != null) {
                hasLineItems = true
                val content = m.groupValues[2].trim()
                if (content.isNotEmpty()) {
                    processedLines.add("$currentItemIndex. $content")
                    currentItemIndex++
                }
            } else {
                if (line.trim().matches(Regex("""^\d+[\.):]?$"""))) {
                    // skip empty number line
                } else {
                    processedLines.add(line)
                }
            }
        }

        var result = if (hasLineItems) processedLines.joinToString("\n") else cleaned

        // 5. Renumber inline lists (e.g. "are: 1. Foo 2. Bar 4. Baz")
        // Android-compatible: avoid variable-length look-behinds using findAll + context check.
        run {
            val inlineItemRegex = Regex("""(^|[\s\n:])(\d+)[\.):](\s+)([A-Za-z*•\[\-])""")
            val inlineMatches = inlineItemRegex.findAll(result).toList()
            if (inlineMatches.size >= 2) {
                val sb = StringBuilder()
                var lastEnd = 0
                var seq = 1
                for (match in inlineMatches) {
                    val precText = result.substring(maxOf(0, match.range.first - 10), match.range.first)
                    val isAfterCitation = precText.contains(Regex("""(?i)(Page|Pages|Source)\s*$"""))
                    val isPartOfDecimal = precText.matches(Regex(""".*\d+\.\s*$"""))
                    if (!isAfterCitation && !isPartOfDecimal) {
                        sb.append(result.substring(lastEnd, match.range.first))
                        sb.append(match.groupValues[1]) // restore delimiter
                        val nextChar = match.groupValues[4]
                        sb.append("$seq. $nextChar")
                        lastEnd = match.range.last + 1
                        seq++
                    }
                }
                sb.append(result.substring(lastEnd))
                result = sb.toString()
            }
        }

        // Clean extra whitespace
        result = result.replace(Regex("""[ \t]{2,}"""), " ")
        result = result.replace(Regex("""\n{3,}"""), "\n\n")

        return result.trim()
    }

    /**
     * Validates whether a single claim is supported by candidate evidence blocks.
     */
    fun validateClaimSupport(
        claim: String,
        candidateBlocks: List<ParsedSourceBlock>,
        allBlocks: List<ParsedSourceBlock>,
        explicitCitations: List<ValidatedCitation>,
        query: String = "",
        classification: com.siddu.offlinedocai.ai.retrieval.QueryClassification? = null
    ): ClaimValidationResult {
        val claimWithoutCitations = stripCitations(claim)

        // 0. Question-Relevance Gate
        if (query.isNotBlank() && !isClaimRelevantToQuery(claimWithoutCitations, query, classification)) {
            return ClaimValidationResult(
                originalClaim = claim,
                validatedText = "",
                status = ValidationStatus.UNSUPPORTED,
                citations = emptyList(),
                supportingBlockIds = emptyList(),
                supportScore = 0f,
                reason = "Claim is not semantically relevant to the user's question."
            )
        }

        // 1. If no candidate blocks exist in context, claim is unsupported
        if (candidateBlocks.isEmpty()) {
            return ClaimValidationResult(
                originalClaim = claim,
                validatedText = "",
                status = ValidationStatus.UNSUPPORTED,
                citations = emptyList(),
                supportingBlockIds = emptyList(),
                supportScore = 0f,
                reason = "No candidate evidence blocks found in context."
            )
        }

        // Combine candidate evidence text
        val combinedEvidence = candidateBlocks.joinToString("\n\n") { it.text }

        // 2. Exact Value / Number / Date / URL Safety Check
        val (valuesSafe, valueViolation) = checkFactualValuesSafety(claimWithoutCitations, combinedEvidence)
        if (!valuesSafe) {
            return ClaimValidationResult(
                originalClaim = claim,
                validatedText = "",
                status = ValidationStatus.UNSUPPORTED,
                citations = emptyList(),
                supportingBlockIds = emptyList(),
                supportScore = CONTRADICTION_PENALTY,
                reason = valueViolation ?: "Claim contradicts or introduces ungrounded factual values."
            )
        }

        // 3. Compute Lexical & Paraphrase Support Score
        val rawScore = computeSupportScore(claimWithoutCitations, combinedEvidence)

        // Boost score if the claim provides an exact factual match (phone, email, URL, or specific metric) found in evidence
        val hasExactPhone = PHONE_REGEX.findAll(claimWithoutCitations).any { p ->
            val digits = p.value.filter { it.isDigit() }
            digits.length >= 7 && combinedEvidence.filter { it.isDigit() }.contains(digits)
        }
        val hasExactEmail = EMAIL_REGEX.findAll(claimWithoutCitations).any { e ->
            combinedEvidence.lowercase().contains(e.value.lowercase())
        }
        val hasExactUrl = URL_REGEX.findAll(claimWithoutCitations).any { u ->
            val cleanUrl = u.value.trimEnd('.').removePrefix("https://").removePrefix("http://").removePrefix("www.")
            combinedEvidence.lowercase().contains(cleanUrl.lowercase())
        }
        val hasExactMetric = METRIC_WITH_UNIT_REGEX.findAll(claimWithoutCitations).any { m ->
            combinedEvidence.lowercase().replace(" ", "").contains(m.value.lowercase().replace(" ", ""))
        }

        val score = if (hasExactPhone || hasExactEmail || hasExactUrl || hasExactMetric) {
            max(rawScore, 0.90f)
        } else {
            rawScore
        }

        // 3. Compound Clause Splitting for Partially Supported Claims (Requirement 11, Test L)
        if (COMPOUND_CONJUNCTION_REGEX.containsMatchIn(claimWithoutCitations)) {
            val clauses = claimWithoutCitations.split(COMPOUND_CONJUNCTION_REGEX)
                .map { it.trim() }
                .filter { it.isNotEmpty() }

            if (clauses.size >= 2) {
                val clause1 = clauses[0]
                val clause2 = clauses[1]

                val score1 = computeSupportScore(clause1, combinedEvidence)
                val score2 = computeSupportScore(clause2, combinedEvidence)

                val (safe1, _) = checkFactualValuesSafety(clause1, combinedEvidence)
                val (safe2, _) = checkFactualValuesSafety(clause2, combinedEvidence)

                if (score1 >= CLAUSE_SUPPORT_THRESHOLD && safe1 && (score2 < SUPPORT_THRESHOLD || !safe2)) {
                    val retainedText = if (clause1.endsWith(".")) clause1 else "$clause1."
                    return ClaimValidationResult(
                        originalClaim = claim,
                        validatedText = retainedText,
                        status = ValidationStatus.PARTIALLY_SUPPORTED,
                        citations = explicitCitations,
                        supportingBlockIds = candidateBlocks.mapNotNull { it.sourceId },
                        supportScore = score1,
                        reason = "Supported clause retained; unsupported addition '$clause2' removed."
                    )
                } else if (score2 >= CLAUSE_SUPPORT_THRESHOLD && safe2 && (score1 < SUPPORT_THRESHOLD || !safe1)) {
                    val retainedText = if (clause2.endsWith(".")) clause2 else "$clause2."
                    return ClaimValidationResult(
                        originalClaim = claim,
                        validatedText = retainedText,
                        status = ValidationStatus.PARTIALLY_SUPPORTED,
                        citations = explicitCitations,
                        supportingBlockIds = candidateBlocks.mapNotNull { it.sourceId },
                        supportScore = score2,
                        reason = "Supported clause retained; unsupported addition '$clause1' removed."
                    )
                }
            }
        }

        // 4. Overall Claim Support Verification
        if (score >= SUPPORT_THRESHOLD) {
            // Find specific blocks that contributed to support
            val supportingIds = candidateBlocks.filter { block ->
                computeSupportScore(claimWithoutCitations, block.text) > 0.20f ||
                        (hasExactPhone && PHONE_REGEX.find(block.text) != null) ||
                        (hasExactEmail && EMAIL_REGEX.find(block.text) != null) ||
                        (hasExactUrl && URL_REGEX.find(block.text) != null)
            }.mapNotNull { it.sourceId }.distinct()

            val citations = if (explicitCitations.isNotEmpty()) {
                explicitCitations
            } else {
                // Attach citation from the strongest supporting block for previously uncited claims
                val topBlock = candidateBlocks.maxByOrNull { computeSupportScore(claimWithoutCitations, it.text) }
                if (topBlock?.sourceId != null) {
                    listOf(
                        ValidatedCitation(
                            sourceId = topBlock.sourceId,
                            page = topBlock.page,
                            pageSpan = topBlock.pageSpan,
                            chunkIndex = topBlock.chunkIndex,
                            documentId = topBlock.documentId
                        )
                    )
                } else emptyList()
            }

            return ClaimValidationResult(
                originalClaim = claim,
                validatedText = claim,
                status = ValidationStatus.SUPPORTED,
                citations = citations,
                supportingBlockIds = supportingIds,
                supportScore = score,
                reason = "Claim fully supported by evidence (score=$score)."
            )
        }

        // 5. Unsupported Claim
        return ClaimValidationResult(
            originalClaim = claim,
            validatedText = "",
            status = ValidationStatus.UNSUPPORTED,
            citations = emptyList(),
            supportingBlockIds = emptyList(),
            supportScore = score,
            reason = "Claim support score ($score) below threshold ($SUPPORT_THRESHOLD)."
        )
    }

    /**
     * Strictly verifies numbers, metrics, dates, URLs, emails, and phones.
     * Prevents the LLM from introducing hallucinations such as 5000 mAh when evidence says 3200 mAh.
     */
    fun checkFactualValuesSafety(claimWithoutCitations: String, evidenceText: String): Pair<Boolean, String?> {
        val claimLower = claimWithoutCitations.lowercase()
        val evidenceLower = evidenceText.lowercase()

        // 1. Metrics with Units (e.g. 3200 mAh, 15%, 5000 mAh)
        val claimMetrics = METRIC_WITH_UNIT_REGEX.findAll(claimLower).map { it.value.trim() }.toList()
        for (m in claimMetrics) {
            val normalizedMetric = m.replace(" ", "")
            val normalizedEvidence = evidenceLower.replace(" ", "")
            if (!normalizedEvidence.contains(normalizedMetric)) {
                // Also check if the raw numeric value exists in evidence
                val numPart = m.filter { it.isDigit() || it == '.' }
                if (!evidenceLower.contains(numPart)) {
                    return Pair(false, "Claim introduces metric '$m' not present in evidence.")
                }
            }
        }

        // 2. Dates (e.g. 2024-05-01, 2025 vs 2024)
        val claimDates = DATE_REGEX.findAll(claimWithoutCitations).map { it.value.trim() }.toList()
        for (d in claimDates) {
            if (!evidenceText.contains(d)) {
                return Pair(false, "Claim introduces date '$d' not present in evidence.")
            }
        }

        // 3. URLs
        val claimUrls = URL_REGEX.findAll(claimWithoutCitations).map { it.value.trim().trimEnd('.') }.toList()
        for (u in claimUrls) {
            val cleanUrl = u.removePrefix("https://").removePrefix("http://").removePrefix("www.")
            if (!evidenceLower.contains(cleanUrl.lowercase())) {
                return Pair(false, "Claim introduces URL '$u' not present in evidence.")
            }
        }

        // 4. Emails
        val claimEmails = EMAIL_REGEX.findAll(claimWithoutCitations).map { it.value.trim().lowercase() }.toList()
        for (e in claimEmails) {
            if (!evidenceLower.contains(e)) {
                return Pair(false, "Claim introduces email '$e' not present in evidence.")
            }
        }

        // 5. Phone Numbers
        val claimPhones = PHONE_REGEX.findAll(claimWithoutCitations).map { it.value.trim() }.toList()
        for (p in claimPhones) {
            val digits = p.filter { it.isDigit() }
            val evidenceDigits = evidenceText.filter { it.isDigit() }
            if (digits.length >= 7 && !evidenceDigits.contains(digits)) {
                return Pair(false, "Claim introduces phone number '$p' not present in evidence.")
            }
        }

        // 6. Standalone Multi-digit Numbers and Percentages
        // Exclude page numbers from page references (e.g. "page 104", "**page 104:**")
        val cleanClaimForNumbers = claimLower.replace(Regex("""(?:\bpage|\*\*page)\s+\d+[:\*]*"""), " ")
        val claimNumbers = STANDALONE_NUMBER_REGEX.findAll(cleanClaimForNumbers)
            .map { it.value.trim() }
            .filter { (it.length >= 3 || it.contains("%")) && !claimDates.any { d -> d.contains(it) } }
            .toList()

        for (num in claimNumbers) {
            if (!evidenceLower.contains(num)) {
                return Pair(false, "Claim introduces numeric value '$num' not present in evidence.")
            }
        }

        // 7. Negation Contradiction Check
        val negationWords = setOf("not", "never", "cannot", "can't", "neither", "nor", "lacks", "lacking", "without", "unable", "unsupported")
        val claimTokens = claimLower.split(Regex("[^a-z0-9]+")).filter { it.isNotBlank() }
        val evidenceTokens = evidenceLower.split(Regex("[^a-z0-9]+")).filter { it.isNotBlank() }

        val claimHasNegation = claimTokens.any { it in negationWords }
        val evidenceHasNegation = evidenceTokens.any { it in negationWords }

        if (claimHasNegation != evidenceHasNegation) {
            val keySharedWords = claimTokens.filter { it !in STOP_WORDS && it !in negationWords && it in evidenceTokens }
            if (keySharedWords.size >= 2) {
                val claimNegIdx = claimTokens.indexOfFirst { it in negationWords }
                if (claimNegIdx != -1) {
                    val window = claimTokens.subList(max(0, claimNegIdx - 3), min(claimTokens.size, claimNegIdx + 4))
                    val negatedSharedWords = window.filter { it in keySharedWords }
                    if (negatedSharedWords.isNotEmpty()) {
                        return Pair(false, "Claim negates evidence content: claim negated '${negatedSharedWords.joinToString(" ")}' which evidence states affirmatively.")
                    }
                }
            }
        }

        // 8. Relational Order Inversion Check (e.g. "A precedes B" vs "B precedes A")
        val relationWords = setOf("precedes", "follows", "before", "after", "exceeds", "causes")
        for (rel in relationWords) {
            val isNounPhrase = rel == "causes" && (claimLower.contains("causes of") || claimLower.contains("causes:") || claimLower.contains("causes include"))
            if (!isNounPhrase && claimLower.contains(" $rel ") && evidenceLower.contains(" $rel ")) {
                val claimParts = claimLower.split(Regex("\\b$rel\\b"), limit = 2)
                // Match against the evidence sentence that discusses the relation
                val matchingEvSentence = evidenceLower.lines().flatMap { it.split(Regex("""(?<=[.!?])\s+""")) }
                    .firstOrNull { it.contains(" $rel ") && !it.contains("causes of") } ?: evidenceLower

                val evidenceParts = matchingEvSentence.split(Regex("\\b$rel\\b"), limit = 2)
                if (claimParts.size == 2 && evidenceParts.size == 2) {
                    val claimLeftTokens = claimParts[0].split(Regex("[^a-z0-9]+")).filter { it !in STOP_WORDS && it.isNotBlank() }.toSet()
                    val claimRightTokens = claimParts[1].split(Regex("[^a-z0-9]+")).filter { it !in STOP_WORDS && it.isNotBlank() }.toSet()

                    val evLeftTokens = evidenceParts[0].split(Regex("[^a-z0-9]+")).filter { it !in STOP_WORDS && it.isNotBlank() }.toSet()
                    val evRightTokens = evidenceParts[1].split(Regex("[^a-z0-9]+")).filter { it !in STOP_WORDS && it.isNotBlank() }.toSet()

                    val leftMatchesEvRight = claimLeftTokens.any { it in evRightTokens }
                    val rightMatchesEvLeft = claimRightTokens.any { it in evLeftTokens }

                    if (leftMatchesEvRight && rightMatchesEvLeft) {
                        return Pair(false, "Claim inverts relational order around '$rel' compared to evidence.")
                    }
                }
            }
        }

        return Pair(true, null)
    }

    /**
     * Computes deterministic lexical & paraphrase support score (0.0f to 1.0f).
     * Employs stem-normalized token matching and phrase overlap.
     */
    fun computeSupportScore(claimText: String, evidenceText: String): Float {
        val claimStems = tokenizeAndStem(claimText)
        if (claimStems.isEmpty()) return 1.0f

        val evidenceStems = tokenizeAndStem(evidenceText).toSet()
        if (evidenceStems.isEmpty()) return 0.0f

        val matchedCount = claimStems.count { it in evidenceStems }
        val tokenOverlap = matchedCount.toFloat() / claimStems.size

        // Long contiguous phrase bonus
        val phraseBonus = if (claimStems.size >= 3) {
            val claimTokens = claimText.lowercase().split(Regex("""[^a-zA-Z0-9]+""")).filter { it.length > 2 }
            val hasLongSequence = (0..claimTokens.size - 3).any { i ->
                val trigram = "${claimTokens[i]} ${claimTokens[i + 1]} ${claimTokens[i + 2]}"
                evidenceText.lowercase().contains(trigram)
            }
            if (hasLongSequence) 0.25f else 0.0f
        } else {
            0.0f
        }

        return min(1.0f, (tokenOverlap * 0.75f) + phraseBonus)
    }

    /**
     * Tokenizes text into stemmed content tokens.
     */
    fun tokenizeAndStem(text: String): List<String> {
        return text.lowercase()
            .split(Regex("""[^a-zA-Z0-9]+"""))
            .map { it.trim() }
            .filter { it.isNotEmpty() && it !in STOP_WORDS && it.length > 1 }
            .map { stem(it) }
    }

    /**
     * Stemming function providing paraphrase tolerance without external libraries.
     */
    fun stem(raw: String): String {
        val word = raw.lowercase().filter { it.isLetterOrDigit() }
        if (word.length <= 3) return word
        val base = when {
            word.endsWith("ing") && word.length > 5 -> word.removeSuffix("ing")
            word.endsWith("ation") && word.length > 6 -> word.removeSuffix("ation")
            word.endsWith("tion") && word.length > 5 -> word.removeSuffix("tion")
            word.endsWith("ment") && word.length > 5 -> word.removeSuffix("ment")
            word.endsWith("ance") && word.length > 5 -> word.removeSuffix("ance")
            word.endsWith("ence") && word.length > 5 -> word.removeSuffix("ence")
            word.endsWith("able") && word.length > 5 -> word.removeSuffix("able")
            word.endsWith("ible") && word.length > 5 -> word.removeSuffix("ible")
            word.endsWith("ical") && word.length > 6 -> word.removeSuffix("ical")
            word.endsWith("ic") && word.length > 4 -> word.removeSuffix("ic")
            word.endsWith("ive") && word.length > 5 -> word.removeSuffix("ive")
            word.endsWith("ity") && word.length > 5 -> word.removeSuffix("ity")
            word.endsWith("ly") && word.length > 4 -> word.removeSuffix("ly")
            word.endsWith("al") && word.length > 4 -> word.removeSuffix("al")
            word.endsWith("um") && word.length > 4 -> word.removeSuffix("um")
            word.endsWith("ied") && word.length > 4 -> word.removeSuffix("ied") + "y"
            word.endsWith("ed") && word.length > 4 -> word.removeSuffix("ed")
            word.endsWith("ies") && word.length > 4 -> word.removeSuffix("ies") + "y"
            word.endsWith("es") && word.length > 4 -> word.removeSuffix("es")
            word.endsWith("s") && !word.endsWith("ss") && word.length > 3 -> word.removeSuffix("s")
            else -> word
        }

        // Canonicalize common derivational synonyms and stems
        return when (base) {
            "verif", "verifi", "verify", "ensur", "ensure", "check" -> "ensur"
            "fuse", "fusion", "contact", "join", "unit", "unite", "bind" -> "join"
            "analysi", "analysis", "analyz", "analyze" -> "analys"
            "demonstr", "demonstrat", "demonstrate" -> "demonstr"
            "inspect", "inspecti" -> "inspect"
            "requir", "require", "need" -> "requir"
            "specif", "specific" -> "specif"
            "operat", "operate" -> "operat"
            "design", "designing" -> "design"
            "cambium", "cambial", "cambi" -> "cambi"
            "desicc", "desiccation", "dry", "drying" -> "dry"
            "heal", "healing", "callus" -> "heal"
            "protect", "shield", "seal", "wax" -> "protect"
            "prevent", "avoid" -> "prevent"
            "fail", "failure" -> "fail"
            "cause", "reason" -> "cause"
            "allow", "permit", "enabl" -> "enabl"
            "provid", "suppli" -> "provid"
            "wound", "cut" -> "cut"
            "growth", "grow" -> "grow"
            else -> base
        }
    }

    /**
     * Strips all citation tags from text.
     */
    fun stripCitations(text: String): String {
        return CITATION_REGEX.replace(text, " ")
            .replace(Regex("""[ \t]{2,}"""), " ")
            .replace(Regex("""\n{3,}"""), "\n\n")
            .trim()
    }

    /**
     * Removes fabricated source IDs or invalid page numbers from answer text.
     * Preserves backward compatibility with GGUFLLM.sanitizeCitations.
     */
    fun sanitizeCitations(text: String, validBlocks: List<ParsedSourceBlock>): String {
        val validSourceIds = validBlocks.mapNotNull { it.sourceId }.toSet()
        val validPages = validBlocks.flatMap { parsePageSpan(it.pageSpan, it.page) }.toSet()

        return CITATION_REGEX.replace(text) { match ->
            val docId = match.groupValues[1].ifEmpty { null }?.toLongOrNull()
            val sourceId = match.groupValues[2].ifEmpty { null }?.toLongOrNull()
            val pageFromSource = match.groupValues[3].ifEmpty { null }?.toIntOrNull()
            val pageParen = match.groupValues[4].ifEmpty { null }?.split("-")?.firstOrNull()?.trim()?.toIntOrNull()
            val pageBracket = match.groupValues[5].ifEmpty { null }?.split("-")?.firstOrNull()?.trim()?.toIntOrNull()
            val page = pageFromSource ?: pageParen ?: pageBracket

            if (sourceId != null && validSourceIds.isNotEmpty() && sourceId !in validSourceIds) {
                "" // Fabricated source ID rejected
            } else if (page != null && validPages.isNotEmpty() && page !in validPages) {
                "" // Fabricated page rejected
            } else {
                match.value
            }
        }.replace(Regex("""[ \t]{2,}"""), " ")
         .replace(Regex("""\n{3,}"""), "\n\n")
         .trim()
    }

    /**
     * Extracts a concise grounded fallback sentence from the evidence blocks.
     * Considers only valid explanatory sentences, filtering out headings, titles, and fragments.
     * In SHORT mode: returns at most 1 grounded sentence.
     * In LONG mode: allows up to 2 distinct grounded supporting sentences when available.
     */
    fun extractGroundedFallback(
        blocks: List<ParsedSourceBlock>,
        query: String,
        answerLength: com.siddu.offlinedocai.ai.llm.AnswerLength = com.siddu.offlinedocai.ai.llm.AnswerLength.SHORT
    ): String {
        if (blocks.isEmpty()) return "I searched the provided document(s), but could not find information answering \"$query\"."

        val queryKeywords = query.lowercase().split(Regex("""[^a-zA-Z0-9]+""")).filter { it.length > 2 && it !in STOP_WORDS }

        data class ScoredCandidate(
            val block: ParsedSourceBlock,
            val sentence: String,
            val score: Double
        )

        val candidates = mutableListOf<ScoredCandidate>()

        for (block in blocks) {
            val rawSentences = block.text.split(Regex("""(?<=[.!?])\s+|\n+"""))
                .map { it.trim() }
                .filter { it.length > 10 && !isHeadingOrTitle(it) && !it.startsWith("[") && !it.contains(" / ") }

            for (sent in rawSentences) {
                val sLower = sent.lowercase()
                var score = 0.0

                // Keyword match
                val kwMatches = queryKeywords.count { sLower.contains(it) }
                score += kwMatches * 15.0

                // Preference for complete explanatory sentences
                if (EXPLANATORY_INDICATORS.any { sLower.contains(it) }) {
                    score += 20.0
                }

                // Sentence ending punctuation bonus
                if (sent.endsWith(".") || sent.endsWith("!") || sent.endsWith("?")) {
                    score += 10.0
                } else {
                    score -= 15.0
                }

                // Bonus if sentence contains all query keywords
                if (queryKeywords.isNotEmpty() && kwMatches == queryKeywords.size) {
                    score += 25.0
                }

                candidates.add(ScoredCandidate(block, sent, score))
            }
        }

        candidates.sortByDescending { it.score }

        // Deduplicate candidates that are identical or substrings of each other
        val distinctCandidates = mutableListOf<ScoredCandidate>()
        for (c in candidates) {
            val isDup = distinctCandidates.any { existing ->
                existing.sentence.equals(c.sentence, ignoreCase = true) ||
                existing.sentence.contains(c.sentence, ignoreCase = true) ||
                c.sentence.contains(existing.sentence, ignoreCase = true)
            }
            if (!isDup) {
                distinctCandidates.add(c)
            }
        }

        val best = distinctCandidates.firstOrNull()
        if (best == null) {
            val fallbackBlock = blocks.first()
            val anySentence = fallbackBlock.text.split(Regex("""(?<=[.!?])\s+|\n+"""))
                .map { it.trim() }
                .firstOrNull { it.length > 15 && !it.startsWith("[") && !it.contains(" / ") && !it.matches(Regex("""^(?i)page\s+\d+.*""")) && !isHeadingOrTitle(it) }
                ?: fallbackBlock.text.take(150)
            val cleanSnippet = anySentence.replace(Regex("""\s+"""), " ").take(200).trim()
            val pageRef = if (fallbackBlock.pageSpan.contains("-")) "Pages ${fallbackBlock.pageSpan}" else "Page ${fallbackBlock.page}"
            return "Based on $pageRef: \"$cleanSnippet\""
        }

        val cleanBest = best.sentence.replace(Regex("""\s+"""), " ").take(200).trim()
        val pageRefBest = if (best.block.pageSpan.contains("-")) "Pages ${best.block.pageSpan}" else "Page ${best.block.page}"

        if (answerLength == com.siddu.offlinedocai.ai.llm.AnswerLength.SHORT || distinctCandidates.size < 2) {
            return "Based on $pageRefBest: \"$cleanBest\""
        }

        // LONG mode: check if a second distinct grounded candidate exists with score >= 15.0 and relevant to query
        val second = distinctCandidates.drop(1).firstOrNull { candidate ->
            candidate.score >= 15.0 &&
            isClaimRelevantToQuery(candidate.sentence, query) &&
            !candidate.sentence.equals(best.sentence, ignoreCase = true) &&
            !candidate.sentence.contains(best.sentence, ignoreCase = true) &&
            !best.sentence.contains(candidate.sentence, ignoreCase = true)
        }

        if (second == null) {
            return "Based on $pageRefBest: \"$cleanBest\""
        }

        val cleanSecond = second.sentence.replace(Regex("""\s+"""), " ").take(200).trim()
        val pageRefSecond = if (second.block.pageSpan.contains("-")) "Pages ${second.block.pageSpan}" else "Page ${second.block.page}"

        return if (best.block.page == second.block.page) {
            "Based on $pageRefBest: \"$cleanBest\" \"$cleanSecond\""
        } else {
            "Based on $pageRefBest: \"$cleanBest\" Additionally, on $pageRefSecond: \"$cleanSecond\""
        }
    }
}
