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

    // Sentence splitting: punctuation followed by space and uppercase letter/bullet, ignoring abbreviations and preserving trailing citations
    private val SENTENCE_SPLIT_REGEX = Regex(
        """(?<!\b(?:e\.g|i\.e|mr|dr|ms|mrs|vs|etc|no|fig|pg|al|\d))(?<=[.!?])\s+(?=[A-Z0-9\-\*•])|\n+"""
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

    /**
     * Primary validation entrypoint accepting ParsedSourceBlock.
     */
    fun validate(
        answer: String,
        blocks: List<ParsedSourceBlock>,
        query: String = "",
        isBudgetLimited: Boolean = false
    ): FullValidationResult {
        val trimmedAnswer = answer.trim()

        // 1. Handle Negative/Not-Found Sentinels from GGUFLLM
        if (trimmedAnswer.startsWith("I searched the provided document", ignoreCase = true) ||
            trimmedAnswer.startsWith("I couldn't find", ignoreCase = true) ||
            trimmedAnswer.startsWith("I could not find", ignoreCase = true) ||
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

            val result = validateClaimSupport(claim, candidateBlocks, blocks, validatedCitations)
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
                finalAnswer = sanitizeCitations(ans, blocks)
            }

            // Level 2: Filtered Supported (Unsupported claims removed)
            supportedCount > 0 -> {
                fallbackLevel = FallbackLevel.L2_FILTERED_SUPPORTED
                var reconstructed = trimmedAnswer
                for (cr in adjustedClaimResults) {
                    if (cr.status == ValidationStatus.UNSUPPORTED) {
                        // Remove unsupported sentence/bullet from original answer
                        reconstructed = reconstructed.replace(cr.originalClaim, "").trim()
                    } else if (cr.status == ValidationStatus.PARTIALLY_SUPPORTED && cr.validatedText != cr.originalClaim) {
                        // Replace with supported clause
                        reconstructed = reconstructed.replace(cr.originalClaim, cr.validatedText)
                    }
                }
                if (isBudgetLimited && COMPLETENESS_CLAIM_REGEX.containsMatchIn(reconstructed)) {
                    reconstructed = COMPLETENESS_CLAIM_REGEX.replace(reconstructed) { "the retrieved" }
                }
                finalAnswer = sanitizeCitations(reconstructed, blocks)
            }

            // Level 3: Extractive Fallback (Zero claims supported, but evidence exists)
            blocks.isNotEmpty() -> {
                fallbackLevel = FallbackLevel.L3_EXTRACTIVE_FALLBACK
                finalAnswer = extractGroundedFallback(blocks, query)
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
        isBudgetLimited: Boolean = false
    ): FullValidationResult {
        return validate(
            answer = answer,
            blocks = sourceBlocks.map { it.toParsedSourceBlock() },
            query = query,
            isBudgetLimited = isBudgetLimited
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
     * Validates whether a single claim is supported by candidate evidence blocks.
     */
    fun validateClaimSupport(
        claim: String,
        candidateBlocks: List<ParsedSourceBlock>,
        allBlocks: List<ParsedSourceBlock>,
        explicitCitations: List<ValidatedCitation>
    ): ClaimValidationResult {
        val claimWithoutCitations = stripCitations(claim)

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
        val claimNumbers = STANDALONE_NUMBER_REGEX.findAll(claimLower)
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
            if (claimLower.contains(" $rel ") && evidenceLower.contains(" $rel ")) {
                val claimParts = claimLower.split(Regex("\\b$rel\\b"), limit = 2)
                val evidenceParts = evidenceLower.split(Regex("\\b$rel\\b"), limit = 2)
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
            word.endsWith("ity") && word.length > 5 -> word.removeSuffix("ity")
            word.endsWith("ied") && word.length > 4 -> word.removeSuffix("ied") + "y"
            word.endsWith("ed") && word.length > 4 -> word.removeSuffix("ed")
            word.endsWith("ies") && word.length > 4 -> word.removeSuffix("ies") + "y"
            word.endsWith("es") && word.length > 4 -> word.removeSuffix("es")
            word.endsWith("s") && !word.endsWith("ss") && word.length > 3 -> word.removeSuffix("s")
            else -> word
        }

        // Canonicalize common derivational synonyms
        return when (base) {
            "verif", "verifi", "verify" -> "verif"
            "analysi", "analysis", "analyz", "analyze" -> "analys"
            "demonstr", "demonstrat", "demonstrate" -> "demonstr"
            "inspect", "inspecti" -> "inspect"
            "requir", "require" -> "requir"
            "specif", "specific" -> "specif"
            "operat", "operate" -> "operat"
            "design", "designing" -> "design"
            else -> base
        }
    }

    /**
     * Strips all citation tags from text.
     */
    fun stripCitations(text: String): String {
        return CITATION_REGEX.replace(text, " ")
            .replace(Regex("""\s{2,}"""), " ")
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
        }.replace(Regex("""\s{2,}"""), " ").trim()
    }

    /**
     * Extracts a concise grounded fallback sentence from the top evidence block.
     */
    private fun extractGroundedFallback(blocks: List<ParsedSourceBlock>, query: String): String {
        val topBlock = blocks.firstOrNull() ?: return "I searched the provided document(s), but could not find information answering \"$query\"."
        val sentences = topBlock.text.split(Regex("""(?<=[.!?])\s+|\n+"""))
            .map { it.trim() }
            .filter { it.length > 15 }

        val queryKeywords = query.lowercase().split(Regex("""[^a-zA-Z0-9]+""")).filter { it.length > 2 }
        val bestSentence = sentences.maxByOrNull { s ->
            val sLower = s.lowercase()
            queryKeywords.count { sLower.contains(it) }
        } ?: sentences.firstOrNull() ?: topBlock.text.take(150)

        val cleanSnippet = bestSentence.replace(Regex("""\s+"""), " ").take(200).trim()
        val pageRef = if (topBlock.pageSpan.contains("-")) "Pages ${topBlock.pageSpan}" else "Page ${topBlock.page}"
        return "Based on $pageRef: \"$cleanSnippet\""
    }
}
