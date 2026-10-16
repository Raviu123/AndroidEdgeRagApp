package com.siddu.offlinedocai.ai.retrieval

/**
 * Retrieval strategies. Intents only decide *how evidence is gathered*; they never encode knowledge
 * about any particular document or domain.
 */
enum class QueryIntent {
    /** "What does page 12 say?" — deterministic page fetch. */
    PAGE_LOOKUP,
    /** "Summarize pages 3 to 7" — deterministic page-range fetch. */
    PAGE_RANGE,
    /** "Summarize this document" — evenly sampled coverage of the whole document. */
    SUMMARY,
    /** "Compare X and Y" — hybrid search for the question plus each side separately. */
    COMPARISON,
    /** "List all ..." / "What are the ..." — broader hybrid search, evidence presented in document order. */
    ENUMERATION,
    /** Everything else — hybrid dense + BM25 search. */
    GENERAL
}

data class QueryClassification(
    val intent: QueryIntent,
    val rawQuery: String,
    val pageStart: Int? = null,
    val pageEnd: Int? = null,
    val entityA: String? = null,
    val entityB: String? = null
) {
    /** Extra search strings whose results are fused with the main query (e.g. each side of a comparison). */
    val subQueries: List<String>
        get() = listOfNotNull(entityA, entityB).filter { it.isNotBlank() }
}

/**
 * Deterministic, document-agnostic query router. It is classified **once** per question (in
 * ChatRepository) and the result is passed to retrieval and context assembly.
 */
object QueryIntentRouter {

    private val PAGE_RANGE = Regex(
        """(?i)\b(?:pages?|pp?\.)\s*(\d{1,5})\s*(?:-|–|—|to|through|until|and)\s*(?:pages?\s*)?(\d{1,5})\b|\bfrom\s+page\s*(\d{1,5})\s+(?:to|through|until)\s+(?:page\s*)?(\d{1,5})\b"""
    )
    private val PAGE_LOOKUP = Regex(
        """(?i)\b(?:on|in|at|from|of)?\s*(?:page|pg\.?|p\.)\s*(?:no\.?|number|#)?\s*(\d{1,5})\b|\b(\d{1,5})(?:st|nd|rd|th)\s+page\b"""
    )
    /** Page questions must actually be *about* a page, not merely mention one ("as shown on page 4, why ..."). */
    private val PAGE_QUESTION = Regex(
        """(?i)^\s*(?:what|summari[sz]e|explain|describe|show|read|tell\s+me|give\s+me|what's|whats|list)\b.*\b(?:page|pages|pg|pp?\.)\s*\d|^\s*(?:page|pages|pg|p\.)\s*\d"""
    )

    private val COMPARISON = Regex(
        """(?i)\b(?:compare|contrast)\s+(.+?)\s+(?:and|with|to|vs\.?|versus|against)\s+(.+?)\s*[?.!]*$|\b(?:difference|differences|similarities)\s+between\s+(.+?)\s+and\s+(.+?)\s*[?.!]*$|\bhow\s+(?:does|do|is|are)\s+(.+?)\s+(?:differ|different)\s+from\s+(.+?)\s*[?.!]*$|^\s*(.+?)\s+(?:vs\.?|versus)\s+(.+?)\s*[?.!]*$"""
    )

    private val DOCUMENT_SUMMARY = Regex(
        """(?i)^\s*(?:please\s+)?(?:summari[sz]e|give\s+(?:me\s+)?(?:a|an)\s+(?:summary|overview)|provide\s+(?:a|an)\s+(?:summary|overview)|tl;?dr)\b(?:\s+(?:of\s+)?(?:this|the|my)?\s*(?:document|file|pdf|book|paper|report|text|it))?\s*[?.!]*$|\bwhat\s+(?:is|'s)\s+(?:this|the)\s+(?:document|file|pdf|book|paper|report)\s+about\b|\b(?:main|key)\s+(?:points|ideas|takeaways|topics|findings)\s+(?:of|in|from)\s+(?:this|the)\s+(?:document|file|pdf|book|paper|report)\b|^\s*(?:summary|overview|tl;?dr)\s*[?.!]*$"""
    )

    /**
     * "Summarize" / "summary" including common misspellings (summerise, sumarize, summery, sumary),
     * plus overview-style words. Used with [isWholeDocumentRequest] to catch free-form phrasings.
     */
    private val SUMMARY_WORD = Regex(
        """(?i)\bsumm?[aeiu]r+(?:[iy][sz](?:e|es|ed|ing|ation)?|y|ies)\b|\b(?:overview|synopsis|gist|recap|tl\s*;?\s*dr)\b"""
    )

    /** Words that only shape a summary request (length, format, politeness, the document itself). */
    private val SUMMARY_FILLER = setOf(
        "paper", "report", "book", "text", "article", "content", "contents", "attached", "uploaded", "selected",
        "given", "provided", "whole", "entire", "full", "complete", "overall", "brief", "briefly", "short", "shortly",
        "quick", "quickly", "simple", "simply", "detail", "details", "detailed", "detailedly", "elaborate", "proper",
        "properly", "nice", "good", "clear", "clearly", "word", "words", "line", "lines", "sentence", "sentences",
        "paragraph", "paragraphs", "point", "points", "pointwise", "bullet", "bullets", "key", "main", "important",
        "want", "need", "help", "provide", "write", "make", "create", "get", "kindly", "pls", "plz", "us",
        "one", "two", "three", "four", "five", "ten", "few", "within", "format", "form", "way", "manner", "language",
        "english", "easy", "understand", "everything", "summary", "doc", "docs"
    )

    private val ENUMERATION = Regex(
        """(?i)^\s*(?:please\s+)?(?:list|enumerate|name)\b|\b(?:list|name|enumerate)\s+(?:all|every|each)\b|\bwhat\s+are\s+(?:all\s+)?(?:the\s+)?(?:different\s+|various\s+|main\s+|key\s+)?[\p{L}\- ]{2,40}?s\b(?!\s+(?:of|in)\s+(?:the|this)\s+\w+\s+(?:about|for))|\b(?:all|every)\s+(?:the\s+)?[\p{L}\-]+s?\s+(?:mentioned|listed|described|discussed)\b|\b(?:types|kinds|steps|stages|phases|categories|examples)\s+of\b|\bwhat\s+steps\b|\bhow\s+many\b"""
    )

    private val TEMPORAL_CONNECTORS = Regex(
        """(?i)\b(before|after|prior\s+to|subsequent\s+to|following|earlier\s+than|later\s+than)\b"""
    )
    private val LEAD_IN = Regex(
        """(?i)^\s*(?:did|was|were|is|are|does|do|has|have|had|can|could|would|should|will)\s+"""
    )
    private val PRONOUN_SUBJECT = Regex("""(?i)^(?:he|she|they|it|we|i)\b""")
    private val NAMED_CLAUSE_SUBJECT = Regex(
        """(?i)^(?:the\s+[a-z]+|[A-Z][a-zA-Z0-9_-]+)\s+(?:first\s+|also\s+)?(?:[a-z]+ed|announced|asked|told|requested|discussed|shared|proposed|booked|consulted|sent|agreed|did|was|is)\b"""
    )

    fun route(query: String): QueryClassification {
        val q = query.trim()

        PAGE_RANGE.find(q)?.let { m ->
            val numbers = m.groupValues.drop(1).mapNotNull { it.toIntOrNull() }
            if (numbers.size >= 2 && PAGE_QUESTION.containsMatchIn(q)) {
                val (a, b) = numbers[0] to numbers[1]
                return QueryClassification(QueryIntent.PAGE_RANGE, q, pageStart = minOf(a, b), pageEnd = maxOf(a, b))
            }
        }

        PAGE_LOOKUP.find(q)?.let { m ->
            val page = m.groupValues.drop(1).firstNotNullOfOrNull { it.toIntOrNull() }
            if (page != null && PAGE_QUESTION.containsMatchIn(q)) {
                return QueryClassification(QueryIntent.PAGE_LOOKUP, q, pageStart = page, pageEnd = page)
            }
        }

        if (DOCUMENT_SUMMARY.containsMatchIn(q) || isWholeDocumentRequest(q)) {
            return QueryClassification(QueryIntent.SUMMARY, q)
        }

        COMPARISON.find(q)?.let { m ->
            val sides = m.groupValues.drop(1).filter { it.isNotBlank() }.map(::cleanEntity)
            if (sides.size >= 2 && sides[0].isNotEmpty() && sides[1].isNotEmpty() && !sides[0].equals(sides[1], ignoreCase = true)) {
                return QueryClassification(QueryIntent.COMPARISON, q, entityA = sides[0], entityB = sides[1])
            }
        }

        extractSequentialClauses(q)?.let { (sideA, sideB) ->
            return QueryClassification(QueryIntent.COMPARISON, q, entityA = sideA, entityB = sideB)
        }

        if (ENUMERATION.containsMatchIn(q)) {
            return QueryClassification(QueryIntent.ENUMERATION, q)
        }

        return QueryClassification(QueryIntent.GENERAL, q)
    }

    /**
     * True when the question asks for a summary and names no topic: "can you summerise this pdf in
     * 5 points" is a whole-document summary, "summarize the section on interest rates" is not.
     */
    private fun isWholeDocumentRequest(q: String): Boolean {
        if (!SUMMARY_WORD.containsMatchIn(q)) return false
        val remainder = TextAnalyzer.tokens(SUMMARY_WORD.replace(q, " "))
        return remainder.none { it.length > 1 && !it.all(Char::isDigit) && it !in TextAnalyzer.STOPWORDS && it !in SUMMARY_FILLER }
    }

    private fun extractSequentialClauses(q: String): Pair<String, String>? {
        val clean = LEAD_IN.replace(q.trim(), "").trimEnd('?', '.', '!', ' ')
        val allMatches = TEMPORAL_CONNECTORS.findAll(clean).toList()
        if (allMatches.isEmpty()) return null

        data class Candidate(val left: String, val right: String, val score: Double)
        val candidates = mutableListOf<Candidate>()

        for (m in allMatches) {
            val left = clean.substring(0, m.range.first).trim()
            val right = clean.substring(m.range.last + 1).trim()
            if (left.length < 12 || right.length < 12) continue
            val wordsL = left.split(Regex("""\s+""")).filter { it.isNotEmpty() }
            val wordsR = right.split(Regex("""\s+""")).filter { it.isNotEmpty() }
            if (wordsL.size < 3 || wordsR.size < 3) continue

            val balance = minOf(wordsL.size, wordsR.size).toDouble() / maxOf(wordsL.size, wordsR.size)
            var score = balance * 10.0
            if (PRONOUN_SUBJECT.containsMatchIn(right)) {
                score += 15.0
            } else if (NAMED_CLAUSE_SUBJECT.containsMatchIn(right)) {
                score += 12.0
            }
            candidates.add(Candidate(left, right, score))
        }

        if (candidates.isEmpty()) return null
        val best = candidates.maxByOrNull { it.score } ?: return null
        val sideA = cleanEntity(best.left)
        val sideB = cleanEntity(best.right)
        return if (sideA.isNotBlank() && sideB.isNotBlank() && !sideA.equals(sideB, ignoreCase = true)) {
            sideA to sideB
        } else null
    }

    private fun cleanEntity(raw: String): String =
        raw.trim()
            .replace(Regex("""[?.!,;:"]+$"""), "")
            .replace(Regex("""^(?i:the|a|an)\s+"""), "")
            .replace(Regex("""(?i)\s+(?:in|from)\s+(?:the|this)\s+(?:document|file|pdf|book|paper|report)$"""), "")
            .trim()
}
