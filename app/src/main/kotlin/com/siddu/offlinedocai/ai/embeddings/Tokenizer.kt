package com.siddu.offlinedocai.ai.embeddings

import java.io.File
import java.io.InputStream
import java.text.Normalizer

/**
 * WordPiece tokenizer for uncased BERT-family models (all-MiniLM-L6-v2).
 *
 * Mirrors HuggingFace `BertTokenizer(do_lower_case=True)`:
 * clean text -> split CJK characters -> lowercase -> strip accents (NFD, drop combining marks)
 * -> split on whitespace and any Unicode punctuation -> greedy longest-match WordPiece.
 * A word that cannot be fully decomposed becomes a single [UNK], as in the reference implementation.
 */
class WordPieceTokenizer(vocabStream: InputStream) {
    private val vocab = HashMap<String, Int>(32000)

    val clsTokenId: Int
    val sepTokenId: Int
    val unkTokenId: Int
    val padTokenId: Int

    init {
        vocabStream.bufferedReader(Charsets.UTF_8).useLines { lines ->
            lines.forEachIndexed { index, line ->
                val word = line.trimEnd('\r', '\n')
                if (word.isNotEmpty()) vocab[word] = index
            }
        }
        clsTokenId = vocab["[CLS]"] ?: 101
        sepTokenId = vocab["[SEP]"] ?: 102
        unkTokenId = vocab["[UNK]"] ?: 100
        padTokenId = vocab["[PAD]"] ?: 0
    }

    constructor(vocabFile: File) : this(vocabFile.inputStream())

    /** WordPiece ids for [text] without [CLS]/[SEP]. */
    fun tokenizeToIds(text: String): List<Int> {
        val ids = ArrayList<Int>()
        for (word in basicTokenize(text)) {
            wordPiece(word, ids)
        }
        return ids
    }

    /** Number of WordPiece tokens in [text], excluding [CLS]/[SEP]. */
    fun countTokens(text: String): Int = tokenizeToIds(text).size

    /**
     * Full model input: [CLS] + ids + [SEP], truncated so the total never exceeds [maxLength]
     * and the trailing [SEP] is always kept.
     */
    fun encode(text: String, maxLength: Int): IntArray {
        val body = tokenizeToIds(text)
        val bodyLength = minOf(body.size, maxLength - 2)
        val out = IntArray(bodyLength + 2)
        out[0] = clsTokenId
        for (i in 0 until bodyLength) out[i + 1] = body[i]
        out[bodyLength + 1] = sepTokenId
        return out
    }

    /** Backward-compatible API: [CLS] + ids + [SEP] without truncation. */
    fun tokenize(text: String): List<Int> = listOf(clsTokenId) + tokenizeToIds(text) + sepTokenId

    private fun basicTokenize(text: String): List<String> {
        val cleaned = StringBuilder(text.length + 16)
        for (ch in text) {
            val code = ch.code
            when {
                code == 0 || code == 0xFFFD || (Character.isISOControl(ch) && !ch.isWhitespace()) -> Unit
                isCjk(code) -> cleaned.append(' ').append(ch).append(' ')
                ch.isWhitespace() -> cleaned.append(' ')
                else -> cleaned.append(ch)
            }
        }
        val lowered = cleaned.toString().lowercase()
        val stripped = Normalizer.normalize(lowered, Normalizer.Form.NFD)
            .filter { Character.getType(it) != Character.NON_SPACING_MARK.toInt() }

        val words = ArrayList<String>()
        val current = StringBuilder()
        for (ch in stripped) {
            when {
                ch.isWhitespace() -> flush(current, words)
                isPunctuation(ch) -> {
                    flush(current, words)
                    words.add(ch.toString())
                }
                else -> current.append(ch)
            }
        }
        flush(current, words)
        return words
    }

    private fun flush(current: StringBuilder, words: MutableList<String>) {
        if (current.isNotEmpty()) {
            words.add(current.toString())
            current.setLength(0)
        }
    }

    private fun wordPiece(word: String, out: MutableList<Int>) {
        if (word.length > MAX_CHARS_PER_WORD) {
            out.add(unkTokenId)
            return
        }
        val pieces = ArrayList<Int>()
        var start = 0
        while (start < word.length) {
            var end = word.length
            var found: Int? = null
            while (start < end) {
                val sub = if (start > 0) "##" + word.substring(start, end) else word.substring(start, end)
                val id = vocab[sub]
                if (id != null) {
                    found = id
                    break
                }
                end--
            }
            if (found == null) {
                out.add(unkTokenId)
                return
            }
            pieces.add(found)
            start = end
        }
        out.addAll(pieces)
    }

    private fun isPunctuation(ch: Char): Boolean {
        val code = ch.code
        if ((code in 33..47) || (code in 58..64) || (code in 91..96) || (code in 123..126)) return true
        return when (Character.getType(ch).toByte()) {
            Character.CONNECTOR_PUNCTUATION, Character.DASH_PUNCTUATION, Character.START_PUNCTUATION,
            Character.END_PUNCTUATION, Character.INITIAL_QUOTE_PUNCTUATION, Character.FINAL_QUOTE_PUNCTUATION,
            Character.OTHER_PUNCTUATION -> true
            else -> false
        }
    }

    private fun isCjk(code: Int): Boolean =
        (code in 0x4E00..0x9FFF) || (code in 0x3400..0x4DBF) || (code in 0x20000..0x2A6DF) ||
            (code in 0x2A700..0x2B73F) || (code in 0x2B740..0x2B81F) || (code in 0x2B820..0x2CEAF) ||
            (code in 0xF900..0xFAFF) || (code in 0x2F800..0x2FA1F)

    companion object {
        private const val MAX_CHARS_PER_WORD = 100
    }
}
