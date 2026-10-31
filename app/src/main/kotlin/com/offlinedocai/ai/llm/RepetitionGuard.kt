package com.offlinedocai.ai.llm

/**
 * Small models sometimes fall into a loop, writing the same sentence or list item again and again
 * until the token budget runs out. [isLooping] spots that while the answer streams so generation can
 * stop early, and [clean] removes the repeated parts from the final answer.
 */
object RepetitionGuard {

    /** A sentence this long (normalised) that appears this many times means the model is looping. */
    private const val MIN_SENTENCE_CHARS = 30
    private const val LOOP_SENTENCE_REPEATS = 3
    /** Shorter fragments are never treated as duplicates (e.g. "Yes.", "See above."). */
    private const val MIN_DUPLICATE_CHARS = 20

    private val LIST_MARKER = Regex("""^\s*(?:[-*•‣◦]|\d{1,3}[.)])\s+""")
    private val NUMBERED = Regex("""^(\s*)(\d{1,3})([.)])(\s+)""")
    /** "Data loss: ..." / "**Step two**: ..." — a short label before the actual content. */
    private val LEADING_LABEL = Regex("""^[^:.!?\n]{1,60}:\s+""")
    private val SENTENCE_END = Regex("""(?<=[.!?])\s+(?=\S)""")
    private val NON_WORD = Regex("""[^\p{L}\p{N}]+""")
    private val WHITESPACE = Regex("""\s+""")

    /** Comparable form of a sentence or list item: no list marker, no leading label, no punctuation. */
    internal fun normalize(text: String): String {
        val withoutMarker = text.replace(LIST_MARKER, "")
        val body = LEADING_LABEL.find(withoutMarker)?.let { m ->
            withoutMarker.substring(m.range.last + 1).takeIf { it.length >= MIN_DUPLICATE_CHARS }
        } ?: withoutMarker
        return body.lowercase().replace(NON_WORD, " ").trim()
    }

    private fun sentences(text: String): List<String> =
        text.lines().flatMap { line -> line.split(SENTENCE_END) }.map { it.trim() }.filter { it.isNotEmpty() }

    /** True when [text] has started repeating itself (same sentence 3+ times, or a repeating word cycle at the end). */
    fun isLooping(text: String): Boolean {
        val counts = HashMap<String, Int>()
        for (s in sentences(text)) {
            val key = normalize(s)
            if (key.length < MIN_SENTENCE_CHARS) continue
            if (counts.merge(key, 1, Int::plus)!! >= LOOP_SENTENCE_REPEATS) return true
        }
        return endsInWordCycle(text)
    }

    /** "... the app and the app and the app and" — the last words are one short phrase repeated. */
    private fun endsInWordCycle(text: String): Boolean {
        val words = text.takeLast(1500).lowercase().split(WHITESPACE).filter { it.isNotEmpty() }
        for (period in 1..30) {
            val repeats = if (period < 3) 8 else 4
            if (words.size < period * repeats) break
            val tail = words.takeLast(period * repeats)
            val unit = tail.take(period)
            if ((1 until repeats).all { r -> tail.subList(r * period, (r + 1) * period) == unit }) return true
        }
        return false
    }

    /**
     * Removes sentences and list items that repeat an earlier one (ignoring list numbers and short
     * leading labels), then renumbers numbered lists so no gaps remain.
     */
    fun clean(text: String): String {
        if (text.isBlank()) return text
        val seen = HashSet<String>()
        val kept = mutableListOf<String>()
        for (line in text.lines()) {
            if (line.isBlank()) { kept.add(""); continue }
            // The list marker is kept aside so "2. Text." is not split into "2." and "Text.".
            val marker = LIST_MARKER.find(line)?.value.orEmpty()
            val content = line.substring(marker.length)
            val whole = normalize(content)
            if (marker.isNotEmpty() && whole.length >= MIN_DUPLICATE_CHARS && whole in seen) continue
            val keptParts = content.split(SENTENCE_END).filter { part ->
                val key = normalize(part)
                key.length < MIN_DUPLICATE_CHARS || seen.add(key)
            }
            if (marker.isNotEmpty() && whole.length >= MIN_DUPLICATE_CHARS) seen.add(whole)
            if (keptParts.isEmpty()) continue
            kept.add(marker + keptParts.joinToString(" "))
        }
        return renumber(kept).joinToString("\n").replace(Regex("""\n{3,}"""), "\n\n").trim()
    }

    /** Drops a final line that stops mid-sentence (left behind when generation is stopped early). */
    fun dropUnfinishedTail(text: String): String {
        val trimmed = text.trimEnd()
        val lastBreak = trimmed.lastIndexOf('\n')
        if (lastBreak < 0) return trimmed
        val last = trimmed.substring(lastBreak + 1).trim()
        val finished = last.endsWith(".") || last.endsWith("!") || last.endsWith("?") || last.endsWith("]") || last.endsWith(")")
        return if (finished) trimmed else trimmed.substring(0, lastBreak).trimEnd()
    }

    private fun renumber(lines: List<String>): List<String> {
        var next = 0
        return lines.map { line ->
            val m = NUMBERED.find(line)
            when {
                m != null -> {
                    next = if (next == 0) m.groupValues[2].toInt() else next + 1
                    m.groupValues[1] + next + m.groupValues[3] + m.groupValues[4] + line.substring(m.range.last + 1)
                }
                line.isBlank() || line.startsWith(" ") || line.startsWith("\t") -> line // blank lines and sub-lines keep the list going
                else -> { next = 0; line }
            }
        }
    }
}
