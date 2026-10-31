package com.offlinedocai.ai.retrieval

import java.text.Normalizer

/**
 * Lexical analysis shared by indexing and querying for BM25.
 *
 * lowercase -> strip accents -> split on anything that is not a letter or digit -> drop stopwords
 * -> Porter stem. Terms match only when their stems are *equal*; there is no substring matching,
 * so "rate" never matches "accurate" and "data" never matches "database".
 */
object TextAnalyzer {

    private val SPLIT = Regex("[^\\p{L}\\p{N}]+")
    private val COMBINING_MARKS = Regex("\\p{Mn}+")

    /** Standard English function words (Lucene/Snowball-style). Content words are never removed. */
    val STOPWORDS: Set<String> = setOf(
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
        "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
        "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further",
        "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
        "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself",
        "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves",
        "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
        "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
        "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who",
        "whom", "why", "will", "with", "would", "you", "your", "yours", "yourself", "yourselves",
        // Question scaffolding that carries no topical meaning.
        "please", "tell", "explain", "describe", "give", "show", "list", "mention", "mentioned", "document",
        "documents", "file", "pdf"
    )

    fun normalize(text: String): String =
        Normalizer.normalize(text.lowercase(), Normalizer.Form.NFD).replace(COMBINING_MARKS, "")

    /** Raw lowercase tokens (no stopword removal, no stemming). */
    fun tokens(text: String): List<String> = normalize(text).split(SPLIT).filter { it.isNotEmpty() }

    /** Index/query terms: stopwords removed, stemmed. Single characters are kept only if numeric. */
    fun terms(text: String): List<String> =
        tokens(text).asSequence()
            .filter { it !in STOPWORDS && (it.length > 1 || it[0].isDigit()) }
            .map { PorterStemmer.stem(it) }
            .toList()

    /**
     * True when stemmed term [a] is a likely misspelling of [b] ("reciev" / "receiv", "summeris" /
     * "summaris"): same first letter, at least 5 letters, and at most 1 edit (2 for 8+ letters), where
     * swapping two adjacent letters counts as one edit. Numbers never match fuzzily.
     */
    fun isTypoOf(a: String, b: String): Boolean {
        if (a == b) return true
        val shorter = minOf(a.length, b.length)
        if (shorter < 5 || a[0] != b[0] || a.any(Char::isDigit) || b.any(Char::isDigit)) return false
        val allowed = if (shorter >= 8) 2 else 1
        if (kotlin.math.abs(a.length - b.length) > allowed) return false
        return editDistance(a, b, allowed) <= allowed
    }

    /**
     * Maps each query term that never occurs in the indexed text to its closest typo match in
     * [docFreq] (most frequent wins on ties). Terms that do occur, or have no close match, are kept.
     */
    fun correctTerms(queryTerms: List<String>, docFreq: Map<String, Int>): List<String> =
        queryTerms.map { term ->
            if (term in docFreq || term.length < 5) term
            else docFreq.entries
                .filter { (candidate, _) -> isTypoOf(term, candidate) }
                .minWithOrNull(compareBy<Map.Entry<String, Int>> { editDistance(term, it.key, 2) }.thenByDescending { it.value })
                ?.key ?: term
        }.distinct()

    /** Optimal-string-alignment distance (adjacent transposition = 1), stopping early above [limit]. */
    private fun editDistance(a: String, b: String, limit: Int): Int {
        var prevPrev = IntArray(b.length + 1)
        var prev = IntArray(b.length + 1) { it }
        for (i in 1..a.length) {
            val cur = IntArray(b.length + 1)
            cur[0] = i
            var rowMin = cur[0]
            for (j in 1..b.length) {
                val cost = if (a[i - 1] == b[j - 1]) 0 else 1
                var d = minOf(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
                if (i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1]) d = minOf(d, prevPrev[j - 2] + 1)
                cur[j] = d
                rowMin = minOf(rowMin, d)
            }
            if (rowMin > limit) return limit + 1
            prevPrev = prev
            prev = cur
        }
        return prev[b.length]
    }
}

/**
 * Porter (1980) stemming algorithm, faithful to the reference implementation.
 * Non-ASCII-letter words (numbers, other scripts) are returned unchanged.
 */
object PorterStemmer {

    fun stem(word: String): String {
        if (word.length <= 2 || word.any { it !in 'a'..'z' }) return word
        val s = Stemmer(word.toCharArray())
        s.step1ab(); s.step1c(); s.step2(); s.step3(); s.step4(); s.step5()
        return String(s.b, 0, s.k + 1)
    }

    private class Stemmer(val b: CharArray) {
        var k = b.size - 1
        var j = 0

        private fun cons(i: Int): Boolean = when (b[i]) {
            'a', 'e', 'i', 'o', 'u' -> false
            'y' -> if (i == 0) true else !cons(i - 1)
            else -> true
        }

        /** Number of VC sequences in b[0..j]. */
        private fun m(): Int {
            var n = 0
            var i = 0
            while (true) {
                if (i > j) return n
                if (!cons(i)) break
                i++
            }
            i++
            while (true) {
                while (true) {
                    if (i > j) return n
                    if (cons(i)) break
                    i++
                }
                i++
                n++
                while (true) {
                    if (i > j) return n
                    if (!cons(i)) break
                    i++
                }
                i++
            }
        }

        private fun vowelInStem(): Boolean = (0..j).any { !cons(it) }

        private fun doubleC(i: Int): Boolean = i >= 1 && b[i] == b[i - 1] && cons(i)

        private fun cvc(i: Int): Boolean {
            if (i < 2 || !cons(i) || cons(i - 1) || !cons(i - 2)) return false
            return b[i] !in "wxy"
        }

        private fun ends(s: String): Boolean {
            val len = s.length
            if (len > k + 1) return false
            for (x in 0 until len) if (b[k - len + 1 + x] != s[x]) return false
            j = k - len
            return true
        }

        private fun setTo(s: String) {
            for (x in s.indices) b[j + 1 + x] = s[x]
            k = j + s.length
        }

        private fun r(s: String) { if (m() > 0) setTo(s) }

        fun step1ab() {
            if (b[k] == 's') {
                when {
                    ends("sses") -> k -= 2
                    ends("ies") -> setTo("i")
                    k >= 1 && b[k - 1] != 's' -> k--
                }
            }
            if (ends("eed")) {
                if (m() > 0) k--
            } else if ((ends("ed") || ends("ing")) && vowelInStem()) {
                k = j
                when {
                    ends("at") -> setTo("ate")
                    ends("bl") -> setTo("ble")
                    ends("iz") -> setTo("ize")
                    doubleC(k) -> {
                        val ch = b[k]
                        if (ch != 'l' && ch != 's' && ch != 'z') k--
                    }
                    else -> { j = k; if (m() == 1 && cvc(k)) setTo("e") }
                }
            }
        }

        fun step1c() {
            if (ends("y") && vowelInStem()) b[k] = 'i'
        }

        fun step2() {
            if (k < 1) return
            when (b[k - 1]) {
                'a' -> { if (ends("ational")) r("ate") else if (ends("tional")) r("tion") }
                'c' -> { if (ends("enci")) r("ence") else if (ends("anci")) r("ance") }
                'e' -> { if (ends("izer")) r("ize") }
                'l' -> {
                    if (ends("bli")) r("ble") else if (ends("alli")) r("al") else if (ends("entli")) r("ent")
                    else if (ends("eli")) r("e") else if (ends("ousli")) r("ous")
                }
                'o' -> { if (ends("ization")) r("ize") else if (ends("ation")) r("ate") else if (ends("ator")) r("ate") }
                's' -> {
                    if (ends("alism")) r("al") else if (ends("iveness")) r("ive") else if (ends("fulness")) r("ful")
                    else if (ends("ousness")) r("ous")
                }
                't' -> { if (ends("aliti")) r("al") else if (ends("iviti")) r("ive") else if (ends("biliti")) r("ble") }
                'g' -> { if (ends("logi")) r("log") }
            }
        }

        fun step3() {
            when (b[k]) {
                'e' -> { if (ends("icate")) r("ic") else if (ends("ative")) r("") else if (ends("alize")) r("al") }
                'i' -> { if (ends("iciti")) r("ic") }
                'l' -> { if (ends("ical")) r("ic") else if (ends("ful")) r("") }
                's' -> { if (ends("ness")) r("") }
            }
        }

        fun step4() {
            if (k < 1) return
            val matched = when (b[k - 1]) {
                'a' -> ends("al")
                'c' -> ends("ance") || ends("ence")
                'e' -> ends("er")
                'i' -> ends("ic")
                'l' -> ends("able") || ends("ible")
                'n' -> ends("ant") || ends("ement") || ends("ment") || ends("ent")
                'o' -> (ends("ion") && j >= 0 && (b[j] == 's' || b[j] == 't')) || ends("ou")
                's' -> ends("ism")
                't' -> ends("ate") || ends("iti")
                'u' -> ends("ous")
                'v' -> ends("ive")
                'z' -> ends("ize")
                else -> false
            }
            if (matched && m() > 1) k = j
        }

        fun step5() {
            j = k
            if (b[k] == 'e') {
                val a = m()
                if (a > 1 || (a == 1 && !cvc(k - 1))) k--
            }
            if (b[k] == 'l' && doubleC(k) && m() > 1) k--
        }
    }
}
