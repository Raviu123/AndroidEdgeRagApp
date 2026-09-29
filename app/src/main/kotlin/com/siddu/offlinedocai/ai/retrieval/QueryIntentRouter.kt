package com.siddu.offlinedocai.ai.retrieval

import android.util.Log

/**
 * Supported high-level retrieval intents.
 */
enum class QueryIntent {
    PAGE_LOOKUP,
    PAGE_RANGE,
    EXACT_FACT,
    ENTITY_LOOKUP,
    SECTION_QUERY,
    EXHAUSTIVE_LIST,
    FIND_ALL,
    COMPARISON,
    GLOBAL_SUMMARY,
    CONCEPTUAL_QUERY
}

/**
 * Specialized sub-categories for exact factual value extraction.
 */
enum class FactType {
    PHONE,
    EMAIL,
    URL,
    DATE,
    ID,
    NUMBER,
    PERCENTAGE,
    AMOUNT,
    TEMPERATURE,
    EXACT_VALUE
}

/**
 * Structured query classification result produced by [QueryIntentRouter].
 * Contains routing decision and extracted parameter metadata for the retrieval layer.
 */
data class QueryClassification(
    val intent: QueryIntent,
    val confidence: Float,
    val pageStart: Int? = null,
    val pageEnd: Int? = null,
    val targetTerm: String? = null,
    val entityA: String? = null,
    val entityB: String? = null,
    val factType: FactType? = null,
    val isAmbiguous: Boolean = false,
    val isDefinition: Boolean = false,
    val rawQuery: String
)

/**
 * Generic, deterministic Query Intent Router.
 * Classifies arbitrary user questions into appropriate document retrieval strategies
 * without using hardcoded document-specific keywords.
 */
object QueryIntentRouter {

    private const val TAG = "QueryIntentRouter"

    // 1. PAGE_RANGE Patterns (Checked BEFORE single page lookup)
    private val PAGE_RANGE_REGEX = Regex(
        """(?i)\b(?:pages?|pgs?|p\.)\s*(\d+)\s*(?:-|–|—|to|through|\.\.)\s*(?:pages?|pgs?|p\.)?\s*(\d+)\b|\bfrom\s+(?:page\s*)?(\d+)\s+(?:to|through|-)\s+(?:page\s*)?(\d+)\b|\bsummarize\s+pages?\s*(\d+)\s*(?:-|to|through)\s*(\d+)\b|\bbetween\s+(?:pages?\s*)?(\d+)\s+(?:and|-)\s+(?:pages?\s*)?(\d+)\b"""
    )

    // 2. PAGE_LOOKUP Patterns
    private val PAGE_LOOKUP_REGEX = Regex(
        """(?i)\b(?:page|pg|p\.|page\s*no|page\s*number|page\s*#)[\s\.\-\#:]*(\d+)\b|\b(\d+)(?:st|nd|rd|th)?\s+page\b"""
    )

    // 3. COMPARISON Patterns
    private val COMPARISON_REGEX = Regex(
        """(?i)\b(?:compare|difference\s+between)\s+(.+?)\s+(?:and|with|to|vs|versus)\s+(.+?)(?:\s+in\s+the\s+document|\?|$|\.)|\bhow\s+does\s+(.+?)\s+differ\s+from\s+(.+?)(?:\?|$|\.)|^\s*(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\?|$|\.)"""
    )

    // 4. FIND_ALL Patterns (All occurrences/mentions of a specific target)
    private val FIND_ALL_REGEX = Regex(
        """(?i)\b(?:find|show|list|get)\s+(?:every|all)\s+(?:mention|occurrence|instance|reference)s?\s+of\s+(.+?)(?:\?|$|\.)|\bwhere\s+(?:all\s+)?(?:is|are)\s+(.+?)\s+(?:mentioned|discussed|covered)(?:\?|$|\.)|\bshow\s+every\s+place\s+where\s+(.+?)\s+(?:appears|is\s+mentioned|is\s+discussed)(?:\?|$|\.)|\bevery\s+mention\s+of\s+(.+?)(?:\?|$|\.)"""
    )

    // 5. EXHAUSTIVE_LIST Patterns (All items belonging to a category)
    private val EXHAUSTIVE_LIST_REGEX = Regex(
        """(?i)\b(?:list|give\s+me|show|extract|name)\s+(?:all(?:\s+the)?|every|the\s+complete\s+list(?:\s+of)?|the\s+entire\s+list(?:\s+of)?)\s+(.+?)(?:\?|$|\.)|\bwhat\s+are\s+all(?:\s+the)?\s+(.+?)(?:\?|$|\.)|\b(?:complete|entire)\s+list\s+of\s+(.+?)(?:\?|$|\.)|\blist\s+every\s+(.+?)\s+(?:mentioned|in\s+the\s+document)?(?:\?|$|\.)|\b(?:list|extract|name)\s+(?:all\s+)?(?:the\s+)?([a-zA-Z0-9\-_]+s)\b(?:\s+in\s+the\s+document|\?|$|\.)|\bwhat\s+are\s+the\s+([a-zA-Z0-9\-_]+s)\b(?:\s+in\s+the\s+document|\?|$|\.)"""
    )

    // 6. EXACT_FACT Patterns
    private val FACT_PHONE_REGEX = Regex("""(?i)\b(?:phone|phone\s*number|mobile|mobile\s*number|cell|cell\s*number|telephone|contact\s*number)\b""")
    private val FACT_EMAIL_REGEX = Regex("""(?i)\b(?:email|email\s*address|e-mail|mail\s*address|inbox)\b""")
    private val FACT_URL_REGEX = Regex("""(?i)\b(?:url|website|web\s*site|web\s*page|web\s*link|link|linkedin|github|git\s*repo|repository)\b""")
    private val FACT_DATE_REGEX = Regex("""(?i)\b(?:what\s+date|on\s+what\s+date|when\s+did|when\s+was|date\s+of\s+birth|expiration\s+date|release\s+date|deadline|effective\s+date|publication\s+date)\b""")
    private val FACT_PERCENTAGE_REGEX = Regex("""(?i)\b(?:what\s+percentage|what\s+percent|accuracy\s+rate|percentage|growth\s+rate)\b""")
    private val FACT_AMOUNT_REGEX = Regex("""(?i)\b(?:salary|ctc|compensation|package|price|cost|fee|budget|dollar\s+amount|revenue)\b""")
    private val FACT_ID_REGEX = Regex("""(?i)\b(?:employee\s*id|roll\s*number|registration\s*number|account\s*number|serial\s*number|license\s*number|reference\s*id|id\s*number)\b|\bwhat\s+is\s+the\s+id\b""")
    private val FACT_NUMBER_REGEX = Regex("""(?i)\b(?:what\s+is\s+the\s+number\s+of|how\s+many|what\s+is\s+the\s+quantity|what\s+is\s+the\s+count)\b""")
    private val FACT_VALUE_REGEX = Regex("""(?i)\b(?:what\s+is\s+the\s+(?:exact\s+)?value|exact\s+value|value\s+of)\b""")
    private val FACT_TEMPERATURE_REGEX = Regex("""(?i)\b(?:what\s+(?:is\s+the\s+)?temperature|what\s+temp|temperature\s+recommended|recommended\s+temperature|healing\s+temperature|optimal\s+temperature|temperature\s+for|temperatures|degrees)\b""")

    // Generic Definition / Explanation Query Patterns
    private val DEFINITION_QUERY_REGEX_1 = Regex(
        """^\s*(?i:what\s+does)\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+(?i:mean)[\s\?\.\!]*$"""
    )
    private val DEFINITION_QUERY_REGEX_2 = Regex(
        """^\s*(?i:what\s+is\s+meant\s+by|meaning\s+of|definition\s+of)\s+(?:a\s+|an\s+|the\s+)?(.+?)[\s\?\.\!]*$"""
    )
    private val DEFINITION_QUERY_REGEX_3 = Regex(
        """^\s*(?i:what\s+is)\s+(?!(?:the|all|every)\b)(?:a\s+|an\s+)?([a-zA-Z0-9\-_]+(?:\s+[a-zA-Z0-9\-_]+){0,3})[\s\?\.\!]*$|^\s*(?i:define|explain|describe)\s+(?:a\s+|an\s+|the\s+)?([a-zA-Z0-9\-_]+(?:\s+[a-zA-Z0-9\-_]+){0,3})[\s\?\.\!]*$"""
    )

    // 7. GLOBAL_SUMMARY Patterns
    private val GLOBAL_SUMMARY_REGEX = Regex(
        """(?i)\bwhat\s+is\s+this\s+document\s+about\b|\b(?:summarize|summary\s+of|overview\s+of)\s+(?:the\s+document|the\s+file|this\s+pdf|the\s+book)\b|\b(?:give\s+me\s+a\s+)?(?:summary|overview)\s+of\s+(?:this|the)\s+(?:document|file|book|pdf)\b|\bwhat\s+are\s+the\s+main\s+(?:points|topics|takeaways)\s+of\s+(?:this|the)\s+document\b"""
    )

    // 8. SECTION_QUERY Patterns
    private val SECTION_QUERY_REGEX = Regex(
        """(?i)\b(?:explain|summarize|what\s+is\s+in|what\s+does|read|show)\s+(?:the\s+)?(chapter\s+[0-9a-zA-Z\.]+|section\s+[0-9a-zA-Z\.]+|appendix\s+[0-9a-zA-Z\.]+|methodology(?:\s+section)?|introduction(?:\s+section)?|conclusion(?:\s+section)?|abstract|overview\s+section|architecture\s+section)\b|\bwhat\s+(?:does\s+the|is\s+in\s+the)\s+(.+?)\s+section\s+(?:say|contain|state|discuss)\b|\bwhat\s+are\s+the\s+(objectives|requirements|guidelines|prerequisites|specifications|deliverables|milestones)\b"""
    )

    // 9. ENTITY_LOOKUP Patterns
    // Specifically targets named proper-noun entities (capitalized) or direct person queries ("who is ...")
    private val ENTITY_LOOKUP_REGEX = Regex(
        """^\s*(?i:what\s+is|tell\s+me\s+about)\s+([A-Z][a-zA-Z0-9\-_]+(?:\s+[A-Z0-9][a-zA-Z0-9\-_]+)*)[\s\?\.\!]*$|^\s*(?i:who\s+is)\s+(.+?)[\s\?\.\!]*$|(?i)\bexplain\s+the\s+([A-Z][a-zA-Z0-9\-_]+(?:\s+[A-Z0-9][a-zA-Z0-9\-_]+)*)\s+(?:mission|project|system|protocol|architecture|algorithm)\b"""
    )


    /**
     * Classifies a user query into a structured [QueryClassification] using
     * strict deterministic priority resolution.
     */
    fun route(query: String): QueryClassification {
        val cleanQuery = query.trim()
        val lowerQuery = cleanQuery.lowercase()

        // -------------------------------------------------------------
        // PRIORITY 1: PAGE_RANGE (Must take precedence over PAGE_LOOKUP)
        // -------------------------------------------------------------
        val rangeMatch = PAGE_RANGE_REGEX.find(cleanQuery)
        if (rangeMatch != null) {
            val digits = rangeMatch.groupValues.drop(1).filter { it.isNotEmpty() }.mapNotNull { it.toIntOrNull() }
            if (digits.size >= 2) {
                val start = minOf(digits[0], digits[1])
                val end = maxOf(digits[0], digits[1])
                return QueryClassification(
                    intent = QueryIntent.PAGE_RANGE,
                    confidence = 0.98f,
                    pageStart = start,
                    pageEnd = end,
                    rawQuery = cleanQuery
                ).also { logClassification(it) }
            }
        }

        // -------------------------------------------------------------
        // PRIORITY 2: PAGE_LOOKUP (Single page query)
        // -------------------------------------------------------------
        val pageMatch = PAGE_LOOKUP_REGEX.find(cleanQuery)
        if (pageMatch != null) {
            val pageNum = pageMatch.groupValues.drop(1).firstOrNull { it.isNotEmpty() }?.toIntOrNull()
            if (pageNum != null) {
                return QueryClassification(
                    intent = QueryIntent.PAGE_LOOKUP,
                    confidence = 0.95f,
                    pageStart = pageNum,
                    pageEnd = pageNum,
                    rawQuery = cleanQuery
                ).also { logClassification(it) }
            }
        }

        // -------------------------------------------------------------
        // PRIORITY 3: COMPARISON (Between two entities)
        // -------------------------------------------------------------
        val compMatch = COMPARISON_REGEX.find(cleanQuery)
        if (compMatch != null) {
            val nonEmpties = compMatch.groupValues.drop(1).filter { it.isNotBlank() }
            if (nonEmpties.size >= 2) {
                val cleanA = cleanEntityName(nonEmpties[0])
                val cleanB = cleanEntityName(nonEmpties[1])

                if (cleanA.isNotEmpty() && cleanB.isNotEmpty() && !cleanA.equals(cleanB, ignoreCase = true)) {
                    return QueryClassification(
                        intent = QueryIntent.COMPARISON,
                        confidence = 0.92f,
                        entityA = cleanA,
                        entityB = cleanB,
                        rawQuery = cleanQuery
                    ).also { logClassification(it) }
                }
            }
        }

        // -------------------------------------------------------------
        // PRIORITY 4: FIND_ALL (Occurrences / mentions across document)
        // -------------------------------------------------------------
        val findAllMatch = FIND_ALL_REGEX.find(cleanQuery)
        if (findAllMatch != null) {
            val target = findAllMatch.groupValues.drop(1).firstOrNull { it.isNotBlank() }
            val cleanTarget = cleanTargetPhrase(target ?: "")
            if (cleanTarget.isNotEmpty()) {
                return QueryClassification(
                    intent = QueryIntent.FIND_ALL,
                    confidence = 0.92f,
                    targetTerm = cleanTarget,
                    rawQuery = cleanQuery
                ).also { logClassification(it) }
            }
        }

        // -------------------------------------------------------------
        // PRIORITY 5: SECTION_QUERY (Specific section, chapter, or standard doc headings)
        // -------------------------------------------------------------
        val sectionMatch = SECTION_QUERY_REGEX.find(cleanQuery)
        if (sectionMatch != null) {
            val sectionName = sectionMatch.groupValues.drop(1).firstOrNull { it.isNotBlank() }?.trim()
            if (!sectionName.isNullOrEmpty()) {
                return QueryClassification(
                    intent = QueryIntent.SECTION_QUERY,
                    confidence = 0.90f,
                    targetTerm = cleanTargetPhrase(sectionName),
                    rawQuery = cleanQuery
                ).also { logClassification(it) }
            }
        }

        // -------------------------------------------------------------
        // PRIORITY 6: EXHAUSTIVE_LIST (Complete list of items in category)
        // -------------------------------------------------------------
        val listMatch = EXHAUSTIVE_LIST_REGEX.find(cleanQuery)
        if (listMatch != null) {
            val target = listMatch.groupValues.drop(1).firstOrNull { it.isNotBlank() }
            val cleanTarget = cleanTargetPhrase(target ?: "")
            // Guard against false positives like "Is all of this correct?"
            val isGenuineList = cleanTarget.isNotEmpty() &&
                !cleanTarget.startsWith("this") &&
                !cleanTarget.startsWith("that")

            if (isGenuineList) {
                return QueryClassification(
                    intent = QueryIntent.EXHAUSTIVE_LIST,
                    confidence = 0.90f,
                    targetTerm = cleanTarget,
                    rawQuery = cleanQuery
                ).also { logClassification(it) }
            }
        }

        // -------------------------------------------------------------
        // PRIORITY 7: EXACT_FACT (Structured factual value query)
        // -------------------------------------------------------------
        val exactFactClassification = classifyExactFact(cleanQuery, lowerQuery)
        if (exactFactClassification != null) {
            return exactFactClassification.also { logClassification(it) }
        }

        // -------------------------------------------------------------
        // PRIORITY 8: GLOBAL_SUMMARY (Document-wide overview / summary)
        // -------------------------------------------------------------
        val isExplicitGlobalSummary = GLOBAL_SUMMARY_REGEX.containsMatchIn(cleanQuery) ||
            (lowerQuery.split(Regex("\\s+")).size <= 4 && listOf("summary", "summarize", "overview", "give me a summary", "give me an overview").any { lowerQuery == it })
        if (isExplicitGlobalSummary) {
            return QueryClassification(
                intent = QueryIntent.GLOBAL_SUMMARY,
                confidence = 0.90f,
                rawQuery = cleanQuery
            ).also { logClassification(it) }
        }

        // -------------------------------------------------------------
        // PRIORITY 9: ENTITY_LOOKUP (Direct info about a named entity)
        // -------------------------------------------------------------
        val entityMatch = ENTITY_LOOKUP_REGEX.find(cleanQuery)
        if (entityMatch != null) {
            val entityName = entityMatch.groupValues.drop(1).firstOrNull { it.isNotBlank() }?.trim()
            val cleanEntity = cleanEntityName(entityName ?: "")
            if (cleanEntity.isNotEmpty() && !isCommonPronounOrStopWord(cleanEntity)) {
                return QueryClassification(
                    intent = QueryIntent.ENTITY_LOOKUP,
                    confidence = 0.85f,
                    targetTerm = cleanEntity,
                    rawQuery = cleanQuery
                ).also { logClassification(it) }
            }
        }

        // -------------------------------------------------------------
        // PRIORITY 9.5: DEFINITION_QUERY (Generic Concept Definition / Explanation)
        // -------------------------------------------------------------
        val defConcept = extractDefinitionConcept(cleanQuery)
        if (defConcept != null && !isNonDefinitionalPhrase(defConcept)) {
            return QueryClassification(
                intent = QueryIntent.CONCEPTUAL_QUERY,
                confidence = 0.92f,
                targetTerm = defConcept,
                isDefinition = true,
                rawQuery = cleanQuery
            ).also { logClassification(it) }
        }

        // -------------------------------------------------------------
        // PRIORITY 10: CONCEPTUAL_QUERY (Default / Ambiguous Safe Fallback)
        // -------------------------------------------------------------
        return QueryClassification(
            intent = QueryIntent.CONCEPTUAL_QUERY,
            confidence = 0.70f,
            isAmbiguous = isQueryAmbiguous(lowerQuery),
            rawQuery = cleanQuery
        ).also { logClassification(it) }
    }

    private fun classifyExactFact(cleanQuery: String, lowerQuery: String): QueryClassification? {
        // Compound queries asking multiple questions should NOT be treated as a single atomic exact fact
        val isCompoundQuery = Regex("""(?i)\b(?:and\s+(?:what|how|why|which|who|where|explain|describe|list))\b""").containsMatchIn(cleanQuery) ||
            Regex("""(?i)\b(?:as\s+well\s+as\s+(?:what|how|why))\b""").containsMatchIn(cleanQuery)
        if (isCompoundQuery) return null

        val factType = when {
            FACT_PHONE_REGEX.containsMatchIn(lowerQuery) -> FactType.PHONE
            FACT_EMAIL_REGEX.containsMatchIn(lowerQuery) -> FactType.EMAIL
            FACT_URL_REGEX.containsMatchIn(lowerQuery) -> FactType.URL
            FACT_DATE_REGEX.containsMatchIn(lowerQuery) -> FactType.DATE
            FACT_PERCENTAGE_REGEX.containsMatchIn(lowerQuery) -> FactType.PERCENTAGE
            FACT_AMOUNT_REGEX.containsMatchIn(lowerQuery) -> FactType.AMOUNT
            FACT_ID_REGEX.containsMatchIn(lowerQuery) -> FactType.ID
            FACT_TEMPERATURE_REGEX.containsMatchIn(lowerQuery) -> FactType.TEMPERATURE
            FACT_VALUE_REGEX.containsMatchIn(lowerQuery) -> FactType.EXACT_VALUE
            FACT_NUMBER_REGEX.containsMatchIn(lowerQuery) -> FactType.NUMBER
            else -> null
        }

        return factType?.let {
            QueryClassification(
                intent = QueryIntent.EXACT_FACT,
                confidence = 0.90f,
                factType = it,
                rawQuery = cleanQuery
            )
        }
    }

    private fun extractDefinitionConcept(query: String): String? {
        val m1 = DEFINITION_QUERY_REGEX_1.find(query)
        if (m1 != null) {
            val term = cleanTargetPhrase(m1.groupValues[1])
            if (term.isNotEmpty() && !isCommonPronounOrStopWord(term)) return term
        }
        val m2 = DEFINITION_QUERY_REGEX_2.find(query)
        if (m2 != null) {
            val term = cleanTargetPhrase(m2.groupValues[1])
            if (term.isNotEmpty() && !isCommonPronounOrStopWord(term)) return term
        }
        val m3 = DEFINITION_QUERY_REGEX_3.find(query)
        if (m3 != null) {
            val raw = (m3.groupValues.drop(1).firstOrNull { it.isNotBlank() } ?: "").trim()
            val term = cleanTargetPhrase(raw)
            if (term.isNotEmpty() && !isCommonPronounOrStopWord(term)) return term
        }
        return null
    }

    private fun isNonDefinitionalPhrase(term: String): Boolean {
        val lower = term.lowercase()
        return listOf(
            "recommended", "needed", "required", "used", "better", "best", "difference",
            "why", "how", "when", "where", "temperature", "price", "cost", "salary",
            "capacity", "spec", "specs", "specification", "power", "wattage", "voltage",
            "process", "processes", "step", "steps", "list", "recipe"
        ).any { lower.contains(it) }
    }

    private fun cleanEntityName(entity: String): String {
        var cleaned = entity.replace(Regex("""[?.,!;:"]+"""), "").trim()
        cleaned = cleaned.replace(Regex("""^(?:the|a|an)\s+""", RegexOption.IGNORE_CASE), "").trim()
        cleaned = cleaned.replace(Regex("""\s+(?:in\s+the\s+document|in\s+this\s+document)$""", RegexOption.IGNORE_CASE), "").trim()
        return cleaned
    }

    private fun cleanTargetPhrase(target: String): String {
        var cleaned = target.replace(Regex("""[?.,!;:"]+"""), "").trim()
        cleaned = cleaned.replace(Regex("""^(?:all\s+the|all|every|the|a|an)\s+""", RegexOption.IGNORE_CASE), "").trim()
        cleaned = cleaned.replace(Regex("""\s+(?:mentioned|in\s+the\s+document|discussed|in\s+this\s+document)$""", RegexOption.IGNORE_CASE), "").trim()
        cleaned = cleaned.replace(Regex("""\s+section$""", RegexOption.IGNORE_CASE), "").trim()
        return cleaned
    }

    private fun isCommonPronounOrStopWord(word: String): Boolean {
        val lower = word.lowercase()
        return setOf(
            "this", "that", "there", "their", "it", "they", "we", "he", "she", "what",
            "which", "who", "whom", "whose", "why", "how", "the", "a", "an"
        ).contains(lower)
    }

    private fun isQueryAmbiguous(lowerQuery: String): Boolean {
        val words = lowerQuery.split(Regex("\\s+")).filter { it.isNotEmpty() }
        if (words.size <= 2 && !words.any { it.contains("page") || it.contains("summary") }) {
            return true
        }
        if (lowerQuery.contains("all") && !lowerQuery.contains("list") && !lowerQuery.contains("every")) {
            return true // Ambiguous use of "all"
        }
        val nonStopWords = words.map { it.trim('?', '.', '!', ',', ';', ':') }
            .filter { it !in setOf("what", "about", "that", "this", "it", "how", "and", "is", "of", "to", "the", "a", "an", "there", "do", "does", "did", "can", "so") }
        if (nonStopWords.isEmpty()) {
            return true // Query consists only of demonstratives/pronouns with no topical subject
        }
        return false
    }

    private fun logClassification(c: QueryClassification) {
        try {
            Log.i(TAG, "[INTENT_ROUTER] Query: \"${c.rawQuery}\" -> Intent=${c.intent} (Conf=${c.confidence}, Page=${c.pageStart}..${c.pageEnd}, Target='${c.targetTerm}', FactType=${c.factType}, Ambiguous=${c.isAmbiguous})")
        } catch (_: Throwable) {
            println("[INTENT_ROUTER] Query: \"${c.rawQuery}\" -> Intent=${c.intent} (Conf=${c.confidence}, Page=${c.pageStart}..${c.pageEnd}, Target='${c.targetTerm}', FactType=${c.factType}, Ambiguous=${c.isAmbiguous})")
        }
    }
}
