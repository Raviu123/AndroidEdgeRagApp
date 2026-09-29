package com.siddu.offlinedocai.ai.embeddings

import java.io.File
import java.io.InputStream
import java.util.Locale

/**
 * WordPiece tokenizer for BERT-family models (such as all-MiniLM-L6-v2).
 * Reads standard HuggingFace vocab.txt and produces input_ids and attention_mask.
 */
class WordPieceTokenizer(vocabStream: InputStream) {
    private val vocab = HashMap<String, Int>(32000)
    private val invVocab = HashMap<Int, String>(32000)

    val clsTokenId: Int
    val sepTokenId: Int
    val unkTokenId: Int

    init {
        vocabStream.bufferedReader(Charsets.UTF_8).useLines { lines ->
            lines.forEachIndexed { index, line ->
                val word = line.trim()
                if (word.isNotEmpty()) {
                    vocab[word] = index
                    invVocab[index] = word
                }
            }
        }
        clsTokenId = vocab["[CLS]"] ?: 101
        sepTokenId = vocab["[SEP]"] ?: 102
        unkTokenId = vocab["[UNK]"] ?: 100
    }

    constructor(vocabFile: File) : this(vocabFile.inputStream())

    fun tokenize(text: String): List<Int> {
        val tokens = ArrayList<Int>()
        tokens.add(clsTokenId)

        // Pre-tokenization: lowercase, split on whitespace and punctuation boundaries
        val cleanText = text.lowercase(Locale.ROOT)
        val words = cleanText.split(Regex("(?<=\\p{Punct})|(?=\\p{Punct})|\\s+"))
            .filter { it.isNotBlank() }

        for (word in words) {
            val pieces = wordPieceTokenize(word)
            for (p in pieces) {
                tokens.add(vocab[p] ?: unkTokenId)
            }
        }

        tokens.add(sepTokenId)
        return tokens
    }

    private fun wordPieceTokenize(word: String): List<String> {
        if (vocab.containsKey(word)) {
            return listOf(word)
        }

        val result = ArrayList<String>()
        var start = 0
        val len = word.length

        while (start < len) {
            var end = len
            var curPiece: String? = null

            while (start < end) {
                var substr = word.substring(start, end)
                if (start > 0) {
                    substr = "##$substr"
                }

                if (vocab.containsKey(substr)) {
                    curPiece = substr
                    break
                }
                end--
            }

            if (curPiece == null) {
                result.add("[UNK]")
                break
            }

            result.add(curPiece)
            start = end
        }

        return result
    }

    fun padOrTruncate(tokens: List<Int>, maxLength: Int): Pair<LongArray, LongArray> {
        val inputIds = LongArray(maxLength) { 0L }
        val attentionMask = LongArray(maxLength) { 0L }

        val actualLength = minOf(tokens.size, maxLength)
        for (i in 0 until actualLength) {
            inputIds[i] = tokens[i].toLong()
            attentionMask[i] = 1L
        }

        return Pair(inputIds, attentionMask)
    }
}
