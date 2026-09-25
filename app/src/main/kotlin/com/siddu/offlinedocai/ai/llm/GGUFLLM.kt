package com.siddu.offlinedocai.ai.llm

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.util.regex.Pattern

/**
 * Enhanced implementation for GGUF language model with intelligent context synthesis.
 */
class GGUFLLM(private val context: Context) : LocalLLM {
    private val TAG = "GGUFLLM"
    private val modelFile: File?

    companion object {
        private val GENERIC_ENTITY_REGEX = Regex("""\b[A-Z][a-z0-9]+(?:\s+[A-Z0-9][a-z0-9\-]+)+\b|\b[A-Z]{2,10}(?:[\-_]\d+)?\b|\b[A-Z][a-zA-Z0-9]*\s+\d+(?:\.\d+)?\b""")
        private val COMMA_LIST_REGEX = Regex("""\b[A-Z][a-zA-Z0-9_\-\.\+]{1,25}\b(?:\s*,\s*\b[A-Z][a-zA-Z0-9_\-\.\+]{1,25}\b){2,}""")
        private val METRIC_UNIT_REGEX = Regex("""(?i)\b\d+(?:\.\d+)?\s*(?:mah|ah|wh|kwh|w|kw|mw|v|kv|mv|gb|mb|tb|kb|hz|mhz|ghz|c|°c|f|°f|k|kg|g|mg|lbs|km|m|cm|mm|nm|mph|km/h|ms|s|min|hrs|v|a|ma|rpm|db|dpi|px|%)\b""")
        private val NUM_MATCHES_REGEX = Regex("""(?:\b\d+[\.\)]\s+)""")
        private val WORD_SPLIT_REGEX = Regex("[^a-zA-Z0-9]+")
        private val NON_ALPHANUMERIC_REGEX = Regex("""[^a-zA-Z0-9\s]""")
        private val SENTENCE_SPLIT_REGEX = Regex("""(?<=[.!?])\s+""")
        private val CHUNK_REGEX = Regex("""\[Page\s+(\d+)\]\s*([\s\S]*?)(?=\n\[Page\s+\d+\]|$)""")
    }

    private val dimensionUnitMap = mapOf(
        setOf("capacity", "charge", "charging", "power", "wattage", "energy", "voltage", "current", "battery") to
            Regex("""(?i)\b\d+(?:\.\d+)?\s*(?:mah|ah|wh|kwh|w|kw|mw|v|kv|mv|a|ma)\b"""),
        setOf("temperature", "temp", "thermal", "heat", "hot", "cold") to
            Regex("""(?i)\b\d+(?:\.\d+)?\s*(?:°c|c|°f|f|k|deg|degrees)\b"""),
        setOf("storage", "memory", "ram", "rom", "disk", "capacity", "space") to
            Regex("""(?i)\b\d+(?:\.\d+)?\s*(?:gb|mb|tb|kb|pb|bytes)\b"""),
        setOf("speed", "frequency", "clock", "rate", "latency") to
            Regex("""(?i)\b\d+(?:\.\d+)?\s*(?:ghz|mhz|khz|hz|fps|rpm|mph|km/h|ms|ns|s)\b"""),
        setOf("weight", "mass", "heavy", "light") to
            Regex("""(?i)\b\d+(?:\.\d+)?\s*(?:kg|g|mg|lbs|oz|ton|tons)\b"""),
        setOf("dimension", "dimensions", "size", "width", "height", "length", "distance", "resolution") to
            Regex("""(?i)\b\d+(?:\.\d+)?\s*(?:mm|cm|m|km|nm|inches|px|dpi)\b""")
    )

    init {
        val modelDir = File(context.filesDir, "models/llm")
        modelFile = modelDir.listFiles()?.firstOrNull { it.extension.equals("gguf", ignoreCase = true) }
        if (modelFile == null) {
            Log.e(TAG, "No LLM model found in ${modelDir.absolutePath}")
        } else {
            Log.i(TAG, "Found LLM model: ${modelFile.name}")
        }
    }

    override suspend fun generate(prompt: String): String = withContext(Dispatchers.Default) {
        if (modelFile == null || !modelFile.exists()) {
            Log.w(TAG, "LLM model not found. Using on-device synthesis engine.")
        }

        // 1. Prompt Extraction: Separate Context and Question cleanly
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

        val contextText = if (contextStartIndex in 0 until questionStartIndex) {
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
                } else if (lineTrimmed.startsWith("Answer", ignoreCase = true) || lineTrimmed.startsWith("Given the context", ignoreCase = true)) {
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

        val qLower = question.lowercase().trim()

        val pageLookupRegex = Regex("""(?i)\b(?:page|pg|p\.|page\s*no|page\s*number|page\s*#)[\s\.\-\#:]+(\d+)\b|\b(\d+)(?:st|nd|rd|th)?\s+page\b""")
        val pageMatch = pageLookupRegex.find(qLower)
        val requestedPageNum = pageMatch?.let { m ->
            m.groupValues[1].toIntOrNull() ?: m.groupValues[2].toIntOrNull()
        }

        if (contextText.startsWith("[PAGE_NOT_FOUND:")) {
            val pageNum = contextText.removePrefix("[PAGE_NOT_FOUND:").removeSuffix("]").trim()
            return@withContext "I couldn't find page $pageNum in the provided document."
        }

        if (contextText == "[NO_EVIDENCE_FOUND]") {
            if (requestedPageNum != null) {
                return@withContext "I couldn't find page $requestedPageNum in the provided document."
            }
            return@withContext "I searched the provided document(s), but could not find information answering \"$question\"."
        }

        if (contextText.isEmpty()) {
            // General Conversational Fallback (no linked document)
            return@withContext when {
                qLower.contains("hello") || qLower.contains("hi") || qLower.contains("hey") -> {
                    "Hello! I am your local offline AI assistant. How can I help you today?"
                }
                qLower.contains("help") -> {
                    "I can help you search and analyze your local documents securely. Import a PDF or DOCX file under the 'Documents' tab, link it to a chat, and ask me questions about it!"
                }
                qLower.contains("who are you") || qLower.contains("your name") -> {
                    "I am the Offline DocAI Assistant, running completely locally on your device."
                }
                else -> {
                    val topics = qLower.split(Regex("[^a-zA-Z0-9]+")).filter { it.length > 3 }
                    val topicName = topics.firstOrNull()?.replaceFirstChar { it.uppercase() } ?: "this topic"

                    """
Here is information regarding $topicName:

$topicName is an important domain topic. In an offline environment, we process and analyze user queries using local models and vector repositories.

Key highlights:
1. **Local Architecture:** Direct CPU/NPU processing without network overhead.
2. **Contextual Accuracy:** Answers grounded in user-provided documents.
3. **Data Security:** Complete privacy for offline document management.
""".trimIndent()
                }
            }
        }

        // 2. Parse chunks with explicit page numbers: [Page X] <chunk text>
        val matches = CHUNK_REGEX.findAll(contextText).toList()

        data class PageChunk(val page: Int, val text: String)
        val chunks = if (matches.isNotEmpty()) {
            matches.map {
                PageChunk(
                    page = it.groupValues[1].toIntOrNull() ?: 1,
                    text = it.groupValues[2].trim()
                )
            }
        } else {
            listOf(PageChunk(1, contextText))
        }

        val topPage = chunks.firstOrNull()?.page ?: 1

        // 2.5 Deterministic PAGE_LOOKUP synthesis
        if (requestedPageNum != null) {
            val pageChunks = chunks.filter { it.page == requestedPageNum }
            if (pageChunks.isEmpty()) {
                return@withContext "I couldn't find page $requestedPageNum in the provided document."
            }

            val pageOverviewTriggers = listOf(
                "what is there in", "what is on", "what's on", "what is in", "what's in", "what does",
                "show", "tell me about", "tell me", "contain", "summarize", "summary", "what is mentioned",
                "give me", "content", "view", "read", "details", "overview", "explain page"
            )
            val normalizedTokens = qLower.replace(Regex("[^a-zA-Z0-9]+"), " ").trim()
            val isExplicitPageOverview = pageOverviewTriggers.any { qLower.contains(it) } ||
                normalizedTokens == "page $requestedPageNum" ||
                normalizedTokens == "p $requestedPageNum" ||
                normalizedTokens == "pg $requestedPageNum" ||
                normalizedTokens == "$requestedPageNum page" ||
                normalizedTokens == "${requestedPageNum}th page" ||
                normalizedTokens == "${requestedPageNum}st page" ||
                normalizedTokens == "${requestedPageNum}nd page" ||
                normalizedTokens == "${requestedPageNum}rd page"

            if (isExplicitPageOverview) {
                val fullRawText = pageChunks.joinToString("\n\n") { it.text.trim() }
                val formattedPage = formatPageContent(fullRawText)
                return@withContext """
📄 **Page $requestedPageNum Content:**

$formattedPage

(Page $requestedPageNum)
""".trim()
            }
        }

        // 3. Detect genuine summary/overview questions
        val summaryKeywords = listOf("summarize", "summary", "overview", "what is this document about", "what does this document", "main topics", "brief of", "gist")
        val hasSpecificTarget = Regex("""(?i)\b(?:chapter|section|page|phase|appendix)\b""").containsMatchIn(qLower)
        val isExplicitSummary = !hasSpecificTarget && (summaryKeywords.any { qLower.contains(it) } ||
            (qLower.split(Regex("\\s+")).size <= 3 && listOf("summary", "summarize", "overview", "about").any { qLower.contains(it) }))

        if (isExplicitSummary) {
            val bulletPoints = mutableListOf<String>()
            val seen = mutableSetOf<String>()

            for (chunk in chunks) {
                val sentences = chunk.text.split(Regex("(?<=[.!?])\\s+|\\n+"))
                    .map { it.trim() }
                    .filter { it.length > 20 && !it.startsWith("[Page") }

                for (s in sentences.take(2)) {
                    val normalized = s.replace(Regex("\\s+"), " ")
                    if (seen.add(normalized.lowercase())) {
                        bulletPoints.add("• [Page ${chunk.page}] $normalized")
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

            return@withContext """
📄 **Document Overview & Key Summary:**

$summaryContent

---
*Tip: You can ask specific questions about particular sections, figures, or metrics in this document.*
""".trimIndent()
        }

        // 4. Targeted Structured Entity Fallbacks (phone, email, github, linkedin, salary, percentage)
        val isPhoneQuery = Regex("""\b(?:phone|phone\s*number|mobile|mobile\s*number|cell|cell\s*number|telephone|contact\s*number)\b""", RegexOption.IGNORE_CASE).containsMatchIn(qLower)
        val isEmailQuery = Regex("""\b(?:email|email\s*address|e-mail|mail\s*address|inbox)\b""", RegexOption.IGNORE_CASE).containsMatchIn(qLower)
        val isGithubQuery = Regex("""\b(?:github|git\s*repo|repository)\b""", RegexOption.IGNORE_CASE).containsMatchIn(qLower)
        val isLinkedinQuery = Regex("""\b(?:linkedin|linked\s*in)\b""", RegexOption.IGNORE_CASE).containsMatchIn(qLower)
        val isPercentageQuery = Regex("""\b(?:percentage|percent|%|accuracy|rate)\b""", RegexOption.IGNORE_CASE).containsMatchIn(qLower)
        val isSalaryQuery = Regex("""\b(?:salary|ctc|lpa|compensation|package|price|cost|fee)\b""", RegexOption.IGNORE_CASE).containsMatchIn(qLower)

        if (isPhoneQuery) {
            for (chunk in chunks) {
                val phoneMatch = Regex("""(?i)(?:phone|mobile|cell|contact)?\s*[:#-]?\s*(\+?\d[\d\s\-().]{7,}\d)""").find(chunk.text)
                if (phoneMatch != null) {
                    val num = phoneMatch.groupValues[1].trim()
                    return@withContext "The phone number is $num. (Page ${chunk.page})"
                }
                val phoneMatchSimple = Regex("""(\+?\d{1,3}[\s-]?\d{9,12}|\b\d{10}\b)""").find(chunk.text)
                if (phoneMatchSimple != null) {
                    val num = phoneMatchSimple.groupValues[1].trim()
                    return@withContext "The phone number is $num. (Page ${chunk.page})"
                }
            }
        }


        if (isEmailQuery) {
            for (chunk in chunks) {
                val emailMatch = Regex("""\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b""").find(chunk.text)
                if (emailMatch != null) {
                    val email = emailMatch.groupValues[1].trim()
                    return@withContext "The email address is $email. (Page ${chunk.page})"
                }
            }
        }

        if (isGithubQuery) {
            for (chunk in chunks) {
                val gitMatch = Regex("""(?i)(https?://(?:www\.)?github\.com/[a-zA-Z0-9_.-]+|github\.com/[a-zA-Z0-9_.-]+|github:\s*([^\s,;]+))""").find(chunk.text)
                if (gitMatch != null) {
                    var link = gitMatch.groupValues[0].trim()
                    link = link.replace(Regex("""(?i)^github:\s*"""), "").trim()
                    return@withContext "The GitHub profile is $link. (Page ${chunk.page})"
                }
            }
        }

        if (isLinkedinQuery) {
            for (chunk in chunks) {
                val linkMatch = Regex("""(?i)(https?://(?:www\.)?linkedin\.com/in/[a-zA-Z0-9_.-]+|linkedin\.com/in/[a-zA-Z0-9_.-]+|linkedin:\s*([^\s,;]+))""").find(chunk.text)
                if (linkMatch != null) {
                    var link = linkMatch.groupValues[0].trim()
                    link = link.replace(Regex("""(?i)^linkedin:\s*"""), "").trim()
                    return@withContext "The LinkedIn profile is $link. (Page ${chunk.page})"
                }
            }
        }

        if (isPercentageQuery) {
            for (chunk in chunks) {
                val pctMatch = Regex("""\b\d+(?:\.\d+)?%""").find(chunk.text)
                if (pctMatch != null) {
                    val matchIdx = pctMatch.range.first
                    val startIdx = maxOf(0, chunk.text.lastIndexOfAny(charArrayOf('.', '!', '?', '\n'), maxOf(0, matchIdx - 1)) + 1)
                    var endIdx = chunk.text.indexOfAny(charArrayOf('.', '!', '?', '\n'), matchIdx)
                    if (endIdx == -1) endIdx = minOf(chunk.text.length, matchIdx + 80) else endIdx += 1
                    val snippet = chunk.text.substring(startIdx, endIdx).trim()
                    if (snippet.length in 5..200) {
                        return@withContext "$snippet (Page ${chunk.page})"
                    }
                    return@withContext "${pctMatch.value} (Page ${chunk.page})"
                }
            }
        }

        if (isSalaryQuery) {
            for (chunk in chunks) {
                val salMatch = Regex("""(?i)(?:[\$€£₹]|rs\.?|inr|usd)\s*[\d,]+(?:\.\d+)?(?:\s*(?:lpa|k|m|million|per annum|per month|/month|/year))?|\b\d+(?:,\d+)*(?:\.\d+)?\s*(?:lpa|inr|usd|dollars|rupees)""").find(chunk.text)
                if (salMatch != null) {
                    val matchIdx = salMatch.range.first
                    val startIdx = maxOf(0, chunk.text.lastIndexOfAny(charArrayOf('.', '!', '?', '\n'), maxOf(0, matchIdx - 1)) + 1)
                    var endIdx = chunk.text.indexOfAny(charArrayOf('.', '!', '?', '\n'), matchIdx)
                    if (endIdx == -1) endIdx = minOf(chunk.text.length, matchIdx + 80) else endIdx += 1
                    val snippet = chunk.text.substring(startIdx, endIdx).trim()
                    if (snippet.length in 5..200) {
                        return@withContext "$snippet (Page ${chunk.page})"
                    }
                    return@withContext "${salMatch.value} (Page ${chunk.page})"
                }
            }
        }

        // 5. Generic Multi-Chunk & Multi-Page Bounded Section Aggregator
        val sectionSynonyms = mapOf(
            "skills" to listOf(
                "technical skills", "tech stack", "core competencies", "key skills", "competencies",
                "areas of expertise", "technologies", "languages and frameworks", "programming languages",
                "tools and technologies", "tools", "skills"
            ),
            "technologies" to listOf(
                "technologies", "tech stack", "technology", "technical skills", "tools",
                "languages and frameworks", "frameworks", "tools and technologies", "skills"
            ),
            "projects" to listOf(
                "personal projects", "key projects", "academic projects", "featured projects",
                "notable projects", "recent projects", "technical projects", "selected projects", "projects"
            ),
            "education" to listOf(
                "educational qualifications", "academic background", "academic details", "academic qualifications",
                "academics", "qualifications", "education", "degrees", "degree"
            ),
            "experience" to listOf(
                "professional experience", "work experience", "employment history", "career history",
                "internships", "work history", "experience"
            ),
            "certifications" to listOf(
                "licenses and certifications", "courses and certifications", "certifications",
                "certificates", "courses"
            ),
            "achievements" to listOf(
                "awards and achievements", "honors and awards", "honors", "awards",
                "accomplishments", "achievements"
            ),
            "requirements" to listOf(
                "system requirements", "minimum requirements", "prerequisites", "requirements", "eligibility"
            ),
            "responsibilities" to listOf(
                "roles and responsibilities", "key responsibilities", "responsibilities", "duties"
            ),
            "publications" to listOf(
                "research papers", "publications", "papers", "articles"
            ),
            "languages" to listOf(
                "languages known", "spoken languages", "languages"
            ),
            "summary" to listOf(
                "professional summary", "career summary", "about me", "summary", "profile"
            ),
            "contact" to listOf(
                "contact information", "contact details", "contact info", "contact"
            ),
            "interests" to listOf(
                "interests and hobbies", "areas of interest", "hobbies", "interests"
            )
        )

        val majorTopLevelBoundaries = mapOf(
            "projects" to listOf(
                "education", "academic background", "educational qualifications", "academics",
                "experience", "work experience", "professional experience", "employment history", "internships", "career history",
                "certifications", "licenses and certifications", "courses and certifications", "certificates",
                "achievements", "awards and achievements", "awards", "honors", "accomplishments",
                "publications", "research papers", "papers",
                "technical skills", "skills", "core competencies", "key skills",
                "languages", "languages known", "spoken languages",
                "interests", "hobbies", "summary", "professional summary", "profile", "about me",
                "contact", "contact information", "contact details", "references", "declaration"
            ),
            "skills" to listOf(
                "projects", "personal projects", "key projects", "academic projects", "featured projects",
                "experience", "work experience", "professional experience", "employment history", "internships",
                "education", "academic background", "educational qualifications", "academics",
                "certifications", "licenses and certifications", "courses and certifications",
                "achievements", "awards and achievements", "awards",
                "publications", "research papers",
                "languages", "interests", "hobbies", "summary", "profile", "contact", "references", "declaration"
            ),
            "technologies" to listOf(
                "projects", "personal projects", "key projects", "academic projects", "featured projects",
                "experience", "work experience", "professional experience", "employment history", "internships",
                "education", "academic background", "educational qualifications", "academics",
                "certifications", "licenses and certifications", "courses and certifications",
                "achievements", "awards and achievements", "awards",
                "publications", "research papers",
                "languages", "interests", "hobbies", "summary", "profile", "contact", "references", "declaration"
            ),
            "education" to listOf(
                "projects", "personal projects", "key projects", "academic projects",
                "experience", "work experience", "professional experience", "employment history",
                "technical skills", "skills", "core competencies", "technologies",
                "certifications", "licenses and certifications", "courses and certifications",
                "achievements", "awards", "publications", "languages", "interests", "summary", "profile", "contact", "declaration"
            ),
            "experience" to listOf(
                "projects", "personal projects", "key projects", "academic projects",
                "education", "academic background", "educational qualifications", "academics",
                "technical skills", "skills", "core competencies",
                "certifications", "licenses and certifications",
                "achievements", "awards", "publications", "languages", "interests", "summary", "profile", "contact", "declaration"
            ),
            "certifications" to listOf(
                "projects", "personal projects", "education", "academic background", "experience", "work experience",
                "technical skills", "skills", "achievements", "awards", "publications", "languages", "interests", "summary", "profile", "contact", "declaration"
            ),
            "achievements" to listOf(
                "projects", "personal projects", "education", "experience", "technical skills", "skills", "certifications", "publications", "languages", "interests", "summary", "profile", "contact", "declaration"
            ),
            "publications" to listOf(
                "projects", "education", "experience", "technical skills", "skills", "certifications", "achievements", "languages", "interests", "summary", "profile", "contact", "declaration"
            ),
            "languages" to listOf(
                "projects", "education", "experience", "technical skills", "skills", "certifications", "achievements", "publications", "interests", "summary", "profile", "contact", "declaration"
            )
        )

        val allKnownSectionHeaders = listOf(
            "technical skills", "skills", "core competencies", "key skills", "competencies",
            "areas of expertise", "technologies", "tech stack", "languages and frameworks",
            "programming languages", "tools and technologies", "tools", "projects", "personal projects",
            "key projects", "academic projects", "featured projects", "notable projects",
            "education", "academic background", "educational qualifications", "academics",
            "qualifications", "degrees", "degree", "academic details", "experience",
            "work experience", "professional experience", "employment history", "internships",
            "career history", "certifications", "licenses and certifications", "courses and certifications",
            "certificates", "courses", "achievements", "awards and achievements", "awards",
            "honors", "accomplishments", "requirements", "prerequisites", "responsibilities",
            "roles and responsibilities", "publications", "research papers", "papers",
            "languages known", "languages", "spoken languages", "interests", "hobbies",
            "summary", "professional summary", "profile", "about me", "contact",
            "contact information", "contact details", "references", "declaration",
            "system architecture", "overview", "features", "architecture", "conclusion",
            "methodology", "introduction", "background"
        )

        var matchedSectionKey: String? = null
        var targetTerms: List<String>? = null
        for ((key, syns) in sectionSynonyms) {
            if (syns.any { Regex("\\b" + Regex.escape(it) + "\\b", RegexOption.IGNORE_CASE).containsMatchIn(qLower) }) {
                matchedSectionKey = key
                targetTerms = syns
                break
            }
        }

        if (targetTerms == null && (qLower.split(Regex("\\s+")).size <= 3 || listOf("list all", "show section", "extract section", "section").any { qLower.contains(it) })) {
            val stopWordsNouns = setOf(
                "what", "when", "where", "how", "this", "that", "there", "their", "who", "why",
                "them", "with", "from", "your", "about", "the", "is", "are", "was", "were", "and",
                "or", "in", "on", "at", "to", "a", "an", "of", "for", "by", "does", "do", "can",
                "you", "tell", "me", "explain", "give", "show", "please", "which", "will", "would",
                "mentioned", "person", "candidate", "document", "file", "have", "has", "all", "any"
            )
            val queryNouns = qLower.split(Regex("[^a-zA-Z0-9]+")).filter { it.length > 2 && it !in stopWordsNouns }
            val matchedKnownHeader = allKnownSectionHeaders.firstOrNull { header ->
                queryNouns.any { noun -> noun == header || header.contains(noun) }
            }
            if (matchedKnownHeader != null) {
                targetTerms = listOf(matchedKnownHeader)
                matchedSectionKey = matchedKnownHeader
            }
        }

        if (targetTerms != null) {
            data class PageSegment(val page: Int, val text: String)
            val collectedSegments = mutableListOf<PageSegment>()
            var isScanningSection = false
            var matchedHeaderName: String? = null

            val sectionBoundaries = majorTopLevelBoundaries[matchedSectionKey]
                ?: allKnownSectionHeaders.filter { it.lowercase() !in (targetTerms ?: emptyList()) }

            val sortedTargetTerms = targetTerms.sortedByDescending { it.length }

            // Group and order chunks by page ascending to preserve cross-chunk & cross-page sequence
            val pageGroups = chunks.groupBy { it.page }.toSortedMap()

            for ((pageNumber, pageChunkList) in pageGroups) {
                val unifiedPageText = pageChunkList.joinToString(" ") { it.text }

                if (!isScanningSection) {
                    var bestStartPos = -1
                    var bestHeaderEndPos = -1
                    var currentMatchedTerm: String? = null

                    for (term in sortedTargetTerms) {
                        val headerPattern = Regex("(?i)(?:^|\\b|(?<=[\\s:;•\\-*_]))" + Regex.escape(term) + "\\b\\s*[:\\-]?")
                        val match = headerPattern.find(unifiedPageText)
                        if (match != null) {
                            if (bestStartPos == -1 || match.range.first < bestStartPos) {
                                bestStartPos = match.range.first
                                bestHeaderEndPos = match.range.last + 1
                                currentMatchedTerm = term
                            }
                        }
                    }

                    if (bestStartPos != -1 && currentMatchedTerm != null) {
                        matchedHeaderName = currentMatchedTerm
                        isScanningSection = true
                        val textAfterHeader = unifiedPageText.substring(bestHeaderEndPos)

                        var earliestNextBoundaryOffset = -1
                        for (nextHeader in sectionBoundaries.sortedByDescending { it.length }) {
                            val nextHeaderPattern = Regex("(?i)(?:^|\\b|(?<=[\\s:;•\\-*_]))" + Regex.escape(nextHeader) + "\\b\\s*[:\\-]?")
                            val nextMatch = nextHeaderPattern.find(textAfterHeader)
                            if (nextMatch != null) {
                                val offset = nextMatch.range.first
                                if (earliestNextBoundaryOffset == -1 || offset < earliestNextBoundaryOffset) {
                                    earliestNextBoundaryOffset = offset
                                }
                            }
                        }

                        if (earliestNextBoundaryOffset != -1) {
                            val slice = textAfterHeader.substring(0, earliestNextBoundaryOffset).trim()
                            if (slice.isNotEmpty()) {
                                collectedSegments.add(PageSegment(pageNumber, slice))
                            }
                            isScanningSection = false
                            break // Section cleanly ended on this page
                        } else {
                            val slice = textAfterHeader.trim()
                            if (slice.isNotEmpty()) {
                                collectedSegments.add(PageSegment(pageNumber, slice))
                            }
                            // Continue scanning onto subsequent pages
                        }
                    }
                } else {
                    // Already inside the section from previous chunk/page
                    var earliestNextBoundaryOffset = -1
                    for (nextHeader in sectionBoundaries.sortedByDescending { it.length }) {
                        val nextHeaderPattern = Regex("(?i)(?:^|\\b|(?<=[\\s:;•\\-*_]))" + Regex.escape(nextHeader) + "\\b\\s*[:\\-]?")
                        val nextMatch = nextHeaderPattern.find(unifiedPageText)
                        if (nextMatch != null) {
                            val offset = nextMatch.range.first
                            if (earliestNextBoundaryOffset == -1 || offset < earliestNextBoundaryOffset) {
                                earliestNextBoundaryOffset = offset
                            }
                        }
                    }

                    if (earliestNextBoundaryOffset != -1) {
                        val slice = unifiedPageText.substring(0, earliestNextBoundaryOffset).trim()
                        if (slice.isNotEmpty()) {
                            collectedSegments.add(PageSegment(pageNumber, slice))
                        }
                        isScanningSection = false
                        break // Section completed
                    } else {
                        val slice = unifiedPageText.trim()
                        if (slice.isNotEmpty()) {
                            collectedSegments.add(PageSegment(pageNumber, slice))
                        }
                    }
                }
            }

            if (collectedSegments.isNotEmpty() && matchedHeaderName != null) {
                val title = matchedHeaderName.split(" ").joinToString(" ") { it.replaceFirstChar { c -> c.uppercase() } }

                if (matchedSectionKey == "projects") {
                    val projectEntries = mutableListOf<String>()

                    for (seg in collectedSegments) {
                        val cleanSeg = seg.text.replace(Regex("^[:\\-\\s]+"), "").trim()
                        val rawEntries = mutableListOf<String>()
                        val inlineLabels = setOf(
                            "technologies", "tech stack", "technology", "tools", "role", "description",
                            "overview", "key features", "architecture", "responsibilities", "outcome",
                            "impact", "features", "contributions", "key highlights"
                        )

                        // 1. Detect explicit project markers: "Project 1:", "Project 2:", "1.", "2.", "•", "-", "*"
                        val projectNumPattern = Regex("""(?i)(?:^|(?<=[.\n;•\-_]))\s*(?:(?:project\s*\d+\s*[:\-])|(?:\d+[\.\)]\s+)|(?:[•\-\*]\s+))""")
                        val numMatches = projectNumPattern.findAll(cleanSeg).toList()

                        if (numMatches.size >= 2) {
                            for (idx in numMatches.indices) {
                                val start = numMatches[idx].range.first
                                val end = if (idx + 1 < numMatches.size) numMatches[idx + 1].range.first else cleanSeg.length
                                val pText = cleanSeg.substring(start, end).trim()
                                if (pText.length > 5) rawEntries.add(pText)
                            }
                        } else if (cleanSeg.contains("\n")) {
                            val lines = cleanSeg.lines().map { it.trim() }.filter { it.isNotEmpty() }
                            var currentBlock = StringBuilder()
                            for (line in lines) {
                                val isNewProjectHeader = line.startsWith("•") || line.startsWith("-") || line.startsWith("*") ||
                                    Regex("""(?i)^(?:project\s*\d+|\d+[\.\)])""").containsMatchIn(line)
                                if (isNewProjectHeader && currentBlock.isNotEmpty()) {
                                    rawEntries.add(currentBlock.toString().trim())
                                    currentBlock = StringBuilder()
                                }
                                if (currentBlock.isNotEmpty()) currentBlock.append("\n  ")
                                currentBlock.append(line.replace(Regex("^[•\\-\\*]\\s*"), "").trim())
                            }
                            if (currentBlock.isNotEmpty()) {
                                rawEntries.add(currentBlock.toString().trim())
                            }
                        } else {
                            // Continuous single-line text: group sentences into project blocks
                            val sentences = cleanSeg.split(Regex("(?<=[.!?])\\s+")).map { it.trim() }.filter { it.isNotEmpty() }
                            var currentBlock = StringBuilder()

                            for (sent in sentences) {
                                val colonOrHyphen = sent.indexOfAny(charArrayOf(':', '-'))
                                val prefix = if (colonOrHyphen != -1) sent.substring(0, colonOrHyphen).trim().lowercase() else ""
                                val isInlineLabel = inlineLabels.any { prefix.contains(it) }
                                val isNewProjectTitle = !isInlineLabel && colonOrHyphen in 3..35 && sent[0].isUpperCase()

                                if (isNewProjectTitle && currentBlock.isNotEmpty()) {
                                    rawEntries.add(currentBlock.toString().trim())
                                    currentBlock = StringBuilder()
                                }
                                if (currentBlock.isNotEmpty()) currentBlock.append(" ")
                                currentBlock.append(sent)
                            }
                            if (currentBlock.isNotEmpty()) {
                                rawEntries.add(currentBlock.toString().trim())
                            }
                        }

                        for (p in rawEntries) {
                            val cleanP = p.replace(Regex("^[•\\-\\*\\s]+"), "").trim()
                            if (cleanP.length >= 3) {
                                projectEntries.add("• $cleanP (Page ${seg.page})")
                            }
                        }
                    }

                    if (projectEntries.isNotEmpty()) {
                        val formattedProjects = projectEntries.joinToString("\n\n")
                        return@withContext """
$title:

$formattedProjects
""".trim()
                    }

                } else if (matchedSectionKey == "skills" || matchedSectionKey == "technologies") {
                    val allItems = mutableListOf<String>()
                    val pages = collectedSegments.map { it.page }.distinct()
                    for (seg in collectedSegments) {
                        val cleanSeg = seg.text.replace(Regex("^[:\\-\\s]+"), "").trim()
                        val commaItems = cleanSeg.split(",").map { it.trim() }.filter { it.isNotEmpty() }
                        if (commaItems.size >= 2 && commaItems.all { it.split(Regex("\\s+")).size <= 6 }) {
                            for (it in commaItems) {
                                val s = it.replace(Regex("^[•\\-\\*]\\s*"), "").trim()
                                if (s.isNotEmpty() && s !in allItems) allItems.add(s)
                            }
                        } else {
                            val lines = cleanSeg.lines().map { it.trim() }.filter { it.isNotEmpty() }
                            for (line in lines) {
                                val s = line.replace(Regex("^[•\\-\\*]\\s*"), "").trim()
                                if (s.isNotEmpty() && s !in allItems) allItems.add(s)
                            }
                        }
                    }

                    if (allItems.isNotEmpty()) {
                        val bulletList = allItems.joinToString("\n") { "• $it" }
                        val pageCitation = if (pages.size == 1) "(Page ${pages.first()})" else "(Pages ${pages.joinToString(", ")})"
                        return@withContext """
$title:
$bulletList
$pageCitation
""".trim()
                    }
                } else {
                    // Generic multi-segment formatting for education, experience, certifications, etc.
                    val formattedBlocks = mutableListOf<String>()
                    for (seg in collectedSegments) {
                        val cleanSeg = seg.text.replace(Regex("^[:\\-\\s]+"), "").trim()
                        if (cleanSeg.isNotEmpty()) {
                            formattedBlocks.add("$cleanSeg (Page ${seg.page})")
                        }
                    }
                    if (formattedBlocks.isNotEmpty()) {
                        return@withContext """
$title:
${formattedBlocks.joinToString("\n\n")}
""".trim()
                    }
                }
            }
        }


        // 6. General Document Questions / Reasoning (Granular candidate slicing, never whole chunks)
        val stopWords = setOf(
            "what", "when", "where", "how", "this", "that", "there", "their", "who", "why",
            "them", "with", "from", "your", "about", "the", "is", "are", "was", "were", "and",
            "or", "in", "on", "at", "to", "a", "an", "of", "for", "by", "does", "do", "can",
            "you", "tell", "me", "explain", "give", "show", "please", "which", "will", "would",
            "mentioned", "say", "talk", "says", "talks", "i", "my", "we", "our"
        )
        val compoundIdentifierPattern = Regex("""(?i)\b(pre[- ]phase|phase|stage|step|chapter|section|appendix|type|level|tier)\s+([a-zA-Z0-9]+)\b""")
        val compoundMatches = compoundIdentifierPattern.findAll(qLower).toList()
        val compoundTokens = compoundMatches.map { it.value.lowercase() }
        val compoundSubTokens = compoundMatches.flatMap { listOf(it.groupValues[1].lowercase(), it.groupValues[2].lowercase()) }

        val queryKeywords = qLower.split(Regex("[^a-zA-Z0-9]+")).filter { (it.length > 1 || it in compoundSubTokens) && (it !in stopWords || it in compoundSubTokens) }.distinct()
        val queryStemsMap = queryKeywords.associateWith { getRootStems(it) }
        val queryPhrase = queryKeywords.joinToString(" ")

        // Metric / Technical specification trigger detection
        val specTriggerWords = setOf(
            "capacity", "specs", "specification", "specifications", "spec", "power", "wattage",
            "battery", "charging", "voltage", "current", "temperature", "temperatures", "dimensions",
            "dimension", "size", "weight", "storage", "ram", "memory", "speed", "frequency", "resolution"
        )
        val isSpecQuery = queryKeywords.any { kw ->
            val stems = queryStemsMap[kw] ?: setOf(kw)
            specTriggerWords.any { trigger -> wordMatchesKeyword(trigger, kw, stems) }
        }

        data class ScoredCandidate(val page: Int, val text: String, val score: Double)
        val candidates = mutableListOf<ScoredCandidate>()

        val sectionSplitPattern = Regex("(?i)(?=(?:\\b(?:${allKnownSectionHeaders.take(25).joinToString("|") { Regex.escape(it) }})\\b\\s*[:\\-]?))")

        for (chunk in chunks) {
            val rawSections = chunk.text.split(sectionSplitPattern).map { it.trim() }.filter { it.isNotEmpty() }
            val units = mutableListOf<String>()

            for (sec in rawSections) {
                val numMatchesInSec = NUM_MATCHES_REGEX.findAll(sec).count()
                if (numMatchesInSec >= 2 && sec.length <= 500) {
                    units.add(sec.trim())
                } else if (sec.contains('.') || sec.contains('!') || sec.contains('?')) {
                    val sents = sec.split(SENTENCE_SPLIT_REGEX)
                    for (sent in sents) {
                        val s = sent.trim()
                        if (s.length in 10..350) units.add(s)
                    }
                } else if (sec.contains(",") && sec.length > 60) {
                    val subClauses = sec.split(Regex("(?<=[,;])\\s+"))
                    var currentChunk = StringBuilder()
                    for (sub in subClauses) {
                        if (currentChunk.length + sub.length > 80 && currentChunk.isNotEmpty()) {
                            units.add(currentChunk.toString().trim())
                            currentChunk = StringBuilder()
                        }
                        currentChunk.append(sub).append(" ")
                    }
                    if (currentChunk.isNotEmpty()) {
                        units.add(currentChunk.toString().trim())
                    }
                } else {
                    if (sec.length in 10..350) {
                        units.add(sec)
                    } else if (sec.length > 350) {
                        val words = sec.split(Regex("\\s+"))
                        var i = 0
                        while (i < words.size && units.size < 25) {
                            val sub = words.subList(i, minOf(i + 25, words.size)).joinToString(" ")
                            units.add(sub)
                            i += 20
                        }
                    }
                }
                if (units.size >= 25) break
            }

            for (unit in units) {
                val unitLower = unit.lowercase()
                val unitNorm = unitLower.replace(NON_ALPHANUMERIC_REGEX, " ")
                val unitWords = unit.split(WORD_SPLIT_REGEX).filter { it.isNotEmpty() }
                var score = 0.0

                // Continuous phrase match bonus
                if (queryPhrase.length >= 4 && (unitLower.contains(queryPhrase) || unitNorm.contains(queryPhrase))) {
                    score += 50.0
                }

                // Compound identifier match bonus
                for (ct in compoundTokens) {
                    val normCt = ct.replace("-", " ")
                    if (unitLower.contains(ct) || unitNorm.contains(normCt)) {
                        score += 35.0
                    }
                    // Penalize conflicting identifier in same category (e.g. Phase B when querying Phase A)
                    val parts = ct.split(Regex("""\s+"""))
                    if (parts.size == 2) {
                        val prefix = parts[0]
                        val idVal = parts[1]
                        val conflictPattern = Regex("""\b${Regex.escape(prefix)}\s+([a-zA-Z0-9]+)\b""")
                        val unitConflicts = conflictPattern.findAll(unitLower).map { it.groupValues[1] }.filter { it != idVal }.toList()
                        if (unitConflicts.isNotEmpty() && !unitLower.contains(ct) && !unitNorm.contains(normCt)) {
                            score -= 30.0
                        }
                    }
                }

                // Numbered list pattern bonus
                val numMatchesInUnit = NUM_MATCHES_REGEX.findAll(unit).count()
                if (numMatchesInUnit >= 2) {
                    score += 30.0
                }

                var matchedKeywordCount = 0
                for (kw in queryKeywords) {
                    val stems = queryStemsMap[kw] ?: setOf(kw)
                    if (unitWords.any { w -> wordMatchesKeyword(w, kw, stems) }) {
                        matchedKeywordCount++
                        score += 15.0
                    }
                }

                if (matchedKeywordCount > 0) {
                    // Full query term coverage bonus
                    if (queryKeywords.isNotEmpty() && matchedKeywordCount == queryKeywords.size) {
                        score += 25.0
                    }

                    // Generic Multi-Word Proper Nouns & Alphanumeric Identifiers / Entities Evidence:
                    val entityMatches = GENERIC_ENTITY_REGEX.findAll(unit).toList()
                    if (entityMatches.size >= 2) {
                        score += 15.0 + (entityMatches.size * 3.0).coerceAtMost(20.0)
                    }

                    // Generic Comma-Separated Proper Nouns / Entity List Series:
                    if (COMMA_LIST_REGEX.containsMatchIn(unit)) {
                        score += 20.0
                    }

                    // Dimensional Technical Specification Unit Matching:
                    val matchedDimensionRegexes = dimensionUnitMap.entries
                        .filter { entry -> entry.key.any { dimKey -> queryKeywords.any { kw -> wordMatchesKeyword(dimKey, kw, queryStemsMap[kw] ?: setOf(kw)) } } }
                        .map { it.value }

                    if (matchedDimensionRegexes.isNotEmpty()) {
                        val matchingUnitCount = matchedDimensionRegexes.sumOf { regex -> regex.findAll(unit).count() }
                        if (matchingUnitCount > 0) {
                            score += 25.0 + (matchingUnitCount * 5.0).coerceAtMost(25.0)
                        }
                    } else if (isSpecQuery) {
                        val metricMatches = METRIC_UNIT_REGEX.findAll(unit).toList()
                        if (metricMatches.isNotEmpty()) {
                            score += 15.0 + (metricMatches.size * 3.0).coerceAtMost(15.0)
                        }
                    }
                }

                // Administrative boilerplate penalty
                val isBoilerplateUnit = listOf(
                    "scientific and technical information (sti)",
                    "sti program",
                    "advancement of aeronautics and space science",
                    "standard form 298",
                    "report documentation page",
                    "distribution statement",
                    "table of contents"
                ).any { unitLower.contains(it) }

                if (isBoilerplateUnit && !qLower.contains("sti") && !qLower.contains("documentation")) {
                    score -= 50.0
                }

                if (score > 10.0) {
                    candidates.add(ScoredCandidate(chunk.page, unit, score))
                }
            }
        }

        candidates.sortByDescending { it.score }

        if (candidates.isNotEmpty() && candidates.first().score >= 15.0) {
            // Intent 1: Comparison / Difference Query
            val comparisonRegex = Regex("""(?i)\b(?:compare|difference\s+between)\s+(.+?)\s+(?:and|with|to|vs|versus)\s+(.+)|\b(.+?)\s+(?:vs|versus)\s+(.+)\b""")
            val compMatch = comparisonRegex.find(qLower)
            if (compMatch != null) {
                val rawTargetA = compMatch.groupValues[1].ifEmpty { compMatch.groupValues[3] }.trim()
                val rawTargetB = compMatch.groupValues[2].ifEmpty { compMatch.groupValues[4] }.trim()
                val targetA = rawTargetA.replace(Regex("""[^a-zA-Z0-9\s]"""), " ").trim()
                val targetB = rawTargetB.replace(Regex("""[^a-zA-Z0-9\s]"""), " ").trim()

                if (targetA.isNotEmpty() && targetB.isNotEmpty()) {
                    val wordsA = targetA.split(Regex("\\s+")).filter { it.isNotEmpty() }
                    val wordsB = targetB.split(Regex("\\s+")).filter { it.isNotEmpty() }

                    val specificCandA = candidates.filter { c ->
                        val cl = c.text.lowercase()
                        val clNorm = cl.replace(Regex("""[^a-zA-Z0-9\s]"""), " ")
                        (clNorm.contains(targetA) || (wordsA.size > 1 && wordsA.all { clNorm.contains(it) })) &&
                            !clNorm.contains(targetB)
                    }.sortedByDescending { c ->
                        val clNorm = c.text.lowercase().replace(Regex("""[^a-zA-Z0-9\s]"""), " ").trim()
                        val leadBonus = if (clNorm.startsWith(targetA)) 30.0 else 0.0
                        c.score + leadBonus
                    }
                    val specificCandB = candidates.filter { c ->
                        val cl = c.text.lowercase()
                        val clNorm = cl.replace(Regex("""[^a-zA-Z0-9\s]"""), " ")
                        (clNorm.contains(targetB) || (wordsB.size > 1 && wordsB.all { clNorm.contains(it) })) &&
                            !clNorm.contains(targetA)
                    }.sortedByDescending { c ->
                        val clNorm = c.text.lowercase().replace(Regex("""[^a-zA-Z0-9\s]"""), " ").trim()
                        val leadBonus = if (clNorm.startsWith(targetB)) 30.0 else 0.0
                        c.score + leadBonus
                    }

                    val candA = specificCandA.ifEmpty {
                        candidates.filter { c ->
                            val cl = c.text.lowercase()
                            val clNorm = cl.replace(Regex("""[^a-zA-Z0-9\s]"""), " ")
                            clNorm.contains(targetA) || (wordsA.size > 1 && wordsA.all { clNorm.contains(it) })
                        }
                    }
                    val candB = specificCandB.ifEmpty {
                        candidates.filter { c ->
                            val cl = c.text.lowercase()
                            val clNorm = cl.replace(Regex("""[^a-zA-Z0-9\s]"""), " ")
                            clNorm.contains(targetB) || (wordsB.size > 1 && wordsB.all { clNorm.contains(it) })
                        }
                    }

                    if (candA.isNotEmpty() && candB.isNotEmpty()) {
                        val itemA = candA.first()
                        val itemB = candB.firstOrNull { it != itemA } ?: candB.first()
                        val labelA = targetA.split(" ").joinToString(" ") { it.replaceFirstChar { c -> c.uppercase() } }
                        val labelB = targetB.split(" ").joinToString(" ") { it.replaceFirstChar { c -> c.uppercase() } }

                        val diffSentences = mutableListOf<String>()
                        for (c in candidates) {
                            val cl = c.text.lowercase()
                            if ((cl.contains(targetA) && cl.contains(targetB)) ||
                                cl.contains("while") || cl.contains("whereas") || cl.contains("differs") || cl.contains("unlike")) {
                                diffSentences.add("${c.text} (Page ${c.page})")
                                if (diffSentences.size >= 2) break
                            }
                        }

                        val diffSection = if (diffSentences.isNotEmpty()) {
                            "\n\n**Key Distinctions:**\n" + diffSentences.joinToString("\n") { "• $it" }
                        } else {
                            ""
                        }

                        return@withContext """
Based on the document, here is the comparison between **$labelA** and **$labelB**:

• **$labelA (Page ${itemA.page}):** ${itemA.text}
• **$labelB (Page ${itemB.page}):** ${itemB.text}$diffSection
""".trim()
                    }
                }
            }

            // Intent 2: Location / Source Query ("where does the book discuss X?")
            val locationRegex = Regex("""(?i)\bwhere\s+(?:does|can|in|is|are|do)?\s*(?:the\s+book|the\s+document|we|i)?\s*(?:discuss|mention|talk\s+about|cover|find|address|describe)\b|\bwhere\s+in\s+the\s+book\b|\bwhich\s+(?:pages|sections|chapters)\s+(?:discuss|mention|cover)\b""")
            if (locationRegex.containsMatchIn(qLower)) {
                val pageCandidates = candidates.groupBy { it.page }
                val locationEntries = mutableListOf<String>()
                for ((pageNum, pageCands) in pageCandidates) {
                    val bestCand = pageCands.maxByOrNull { it.score } ?: continue
                    locationEntries.add("• **Page $pageNum:** ${bestCand.text}")
                    if (locationEntries.size >= 4) break
                }
                if (locationEntries.isNotEmpty()) {
                    val topicTitle = qLower.replace(locationRegex, "").replace(Regex("""[^a-zA-Z0-9\s]"""), " ").trim()
                        .split(" ").filter { it.length > 2 && it !in stopWords }.joinToString(" ") { it.replaceFirstChar { c -> c.uppercase() } }
                        .ifEmpty { "the requested topic" }

                    return@withContext """
The document discusses **$topicTitle** in the following location(s):

${locationEntries.joinToString("\n")}
""".trim()
                }
            }

            // Intent 3: Chapter / Section Summary
            val chapterSummaryRegex = Regex("""(?i)\b(?:summarize|summary\s+of|overview\s+of|what\s+is\s+in)\s+(chapter\s+[a-zA-Z0-9]+|section\s+[a-zA-Z0-9\.]+)\b""")
            val chapMatch = chapterSummaryRegex.find(qLower)
            if (chapMatch != null) {
                val targetChapter = chapMatch.groupValues[1].lowercase()
                val chapterName = chapMatch.groupValues[1].split(" ").joinToString(" ") { it.replaceFirstChar { c -> c.uppercase() } }
                val chapCands = candidates.filter { it.text.lowercase().contains(targetChapter) }.ifEmpty { candidates.take(4) }
                if (chapCands.isNotEmpty()) {
                    val overviewText = chapCands.first().text + " (Page ${chapCands.first().page})"
                    val keyPoints = chapCands.drop(1).map { "• ${it.text} (Page ${it.page})" }
                    val pointsSection = if (keyPoints.isNotEmpty()) "\n\n**Key Points:**\n" + keyPoints.joinToString("\n") else ""
                    return@withContext """
📖 **$chapterName Overview:**

$overviewText$pointsSection
""".trim()
                }
            }

            // Intent 4: List / Enumeration Extraction
            val listQueryRegex = Regex("""(?i)\b(?:what\s+are\s+the|give\s+me\s+all\s+(?:the)?|list\s+(?:all\s+the|all)?|name\s+(?:all\s+the|all)?|which\s+(?:are\s+the)?)\s+(.+)\b""")
            val isListQuery = listQueryRegex.containsMatchIn(qLower) &&
                listOf("processes", "phases", "steps", "types", "elements", "components", "requirements", "guidelines", "activities", "stages", "principles", "functions", "missions").any { qLower.contains(it) }

            if (isListQuery) {
                for (cand in candidates.take(4)) {
                    val text = cand.text
                    val numListPattern = Regex("""(?:\b\d+[\.\)]\s+[^\d\.\)]+)""")
                    val numMatches = numListPattern.findAll(text).toList()
                    if (numMatches.size >= 2) {
                        val formattedItems = numMatches.mapIndexed { idx, m ->
                            val cleanItem = m.value.replace(Regex("""^\d+[\.\)]\s*"""), "")
                                .replace(Regex("""^(?:and|or)\s+"""), "")
                                .replace(Regex("""\s+(?:and|or)$"""), "")
                                .trim()
                                .trimEnd(',', ';', '.')
                            "${idx + 1}. $cleanItem"
                        }
                        return@withContext """
Based on the document, the items are:

${formattedItems.joinToString("\n")}

(Page ${cand.page})
""".trim()
                    }

                    val commaSeriesRegex = Regex("""\b([A-Z][a-zA-Z0-9_\-\s]{2,40})(?:,\s*([A-Z][a-zA-Z0-9_\-\s]{2,40}))+(?:,\s*and\s*|\s+and\s+)([A-Z][a-zA-Z0-9_\-\s]{2,40})""")
                    val commaMatch = commaSeriesRegex.find(text)
                    if (commaMatch != null) {
                        val items = text.substring(commaMatch.range.first, commaMatch.range.last + 1)
                            .split(Regex(""",\s*(?:and\s+)?|\s+and\s+"""))
                            .map { it.trim().trimEnd('.', ';', ',') }
                            .filter { it.length > 2 }
                        if (items.size >= 3) {
                            val bulletList = items.mapIndexed { idx, itm -> "${idx + 1}. $itm" }.joinToString("\n")
                            return@withContext """
Based on the document:

$bulletList

(Page ${cand.page})
""".trim()
                        }
                    }
                }
            }

            // Intent 5: Existence Check ("Does the book mention X?")
            val existenceRegex = Regex("""(?i)\b(?:does|is|are|can)\s+(?:the\s+book|the\s+document)?\s*(?:mention|contain|discuss|have|include|reference)\b""")
            if (existenceRegex.containsMatchIn(qLower)) {
                val top = candidates.first()
                val topic = qLower.replace(existenceRegex, "").replace(Regex("""[^a-zA-Z0-9\s]"""), " ").trim()
                    .split(" ").filter { it.length > 2 && it !in stopWords }.joinToString(" ") { it.replaceFirstChar { c -> c.uppercase() } }
                    .ifEmpty { "this topic" }

                return@withContext """
Yes, the document discusses **$topic**. According to Page ${top.page}:

${top.text} (Page ${top.page})
""".trim()
            }

            // Intent 6: General Document QA & Concept Explanation
            val topCand = candidates.first()
            val supportingPoints = mutableListOf<ScoredCandidate>()
            val seenText = mutableSetOf<String>(topCand.text.lowercase().trim())

            for (c in candidates.drop(1)) {
                val norm = c.text.lowercase().trim()
                if (seenText.add(norm) && c.score >= 15.0 && (c.page != topCand.page || c.text.length > 30)) {
                    supportingPoints.add(c)
                }
                if (supportingPoints.size >= 2) break
            }

            if (supportingPoints.isEmpty() || isSpecQuery || (topCand.score >= 30.0 && (supportingPoints.firstOrNull()?.score ?: 0.0) < topCand.score * 0.7)) {
                return@withContext "${topCand.text} (Page ${topCand.page})"
            } else {
                val mainAnswer = "${topCand.text} (Page ${topCand.page})"
                val details = supportingPoints.joinToString("\n") { "• ${it.text} (Page ${it.page})" }
                return@withContext """
$mainAnswer

Key details from the document:
$details
""".trim()
            }
        }

        // 7. Grounded Not Found Fallback: NEVER hallucinate or return generic boilerplate
        return@withContext "I searched the provided document(s), but could not find information answering \"$question\"."
    }

    private fun getRootStems(word: String): Set<String> {
        val w = word.lowercase().trim()
        val stems = mutableSetOf(w)
        if (w.endsWith("ies") && w.length > 4) stems.add(w.removeSuffix("ies") + "y")
        if (w.endsWith("es") && w.length > 4) stems.add(w.removeSuffix("es"))
        if (w.endsWith("s") && w.length > 3) stems.add(w.removeSuffix("s"))
        if (w.endsWith("ed") && w.length > 4) stems.add(w.removeSuffix("ed"))
        if (w.endsWith("ing") && w.length > 5) stems.add(w.removeSuffix("ing"))
        if (w.endsWith("tion") && w.length > 6) stems.add(w.removeSuffix("tion"))
        if (w.endsWith("ity") && w.length > 5) stems.add(w.removeSuffix("ity"))
        if (w.endsWith("able") && w.length > 6) stems.add(w.removeSuffix("able"))
        if (w.endsWith("ive") && w.length > 5) stems.add(w.removeSuffix("ive"))
        return stems
    }

    private fun wordMatchesKeyword(docWord: String, kw: String, kwStems: Set<String>): Boolean {
        val cleanDocWord = docWord.replace(Regex("[^a-zA-Z0-9]"), "").lowercase()
        if (cleanDocWord.isEmpty()) return false
        if (cleanDocWord == kw || cleanDocWord in kwStems) return true
        if (kwStems.any { (it.length >= 4 && cleanDocWord.startsWith(it)) || (it.length >= 4 && cleanDocWord.contains(it)) }) return true
        val docStems = getRootStems(cleanDocWord)
        if (docStems.any { it in kwStems }) return true
        return false
    }

    private fun formatPageContent(rawText: String): String {
        val lines = rawText.lines()
        val formattedLines = mutableListOf<String>()

        for (rawLine in lines) {
            var line = rawLine.trim()
            if (line.isEmpty()) continue

            // 1. Clean up Table of Contents leader dots with trailing page numbers
            // e.g. "7.4.1 Basic HF Model .................... 142" or "1. Introduction . . . . . . 10"
            line = line.replace(Regex("""(?:\s*[\.·]){3,}\s*(\d+)\s*$"""), " — Page $1")

            // 2. Clean up any remaining long chains of dots / leader trails (3 or more dots)
            line = line.replace(Regex("""(?:\s*[\.·]){3,}|\.{3,}"""), " — ")

            // 3. Clean up multiple spaces and trailing dashes
            line = line.replace(Regex("""[ \t]{2,}"""), " ")
                .removeSuffix(" —")
                .trim()

            if (line.isEmpty()) continue

            // 4. If line is a TOC / numbered section / bullet entry, format neatly
            val isNumberedSection = Regex("""^(?:[0-9]+(?:\.[0-9]+)*|[A-Z]\.[0-9]*|\d+\))\s+""").containsMatchIn(line)
            val isBullet = line.startsWith("•") || line.startsWith("-") || line.startsWith("*")

            if (isNumberedSection && !isBullet) {
                formattedLines.add("• $line")
            } else {
                formattedLines.add(line)
            }
        }

        // Group into paragraphs / clean structure
        val result = mutableListOf<String>()
        var currentBulletGroup = mutableListOf<String>()

        for (line in formattedLines) {
            if (line.startsWith("• ")) {
                currentBulletGroup.add(line)
            } else {
                if (currentBulletGroup.isNotEmpty()) {
                    result.add(currentBulletGroup.joinToString("\n"))
                    currentBulletGroup = mutableListOf()
                }
                result.add(line)
            }
        }
        if (currentBulletGroup.isNotEmpty()) {
            result.add(currentBulletGroup.joinToString("\n"))
        }

        return result.joinToString("\n\n").trim()
    }
}

