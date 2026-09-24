package com.example.offlinedocai.ai.embeddings

import java.io.File
import java.io.InputStream
import java.util.Locale

class WordPieceTokenizer(vocabStream: InputStream) {
    private val vocab = mutableMapOf<String, Int>()
    private val invVocab = mutableMapOf<Int, String>()

    init {
        vocabStream.bufferedReader().useLines { lines ->
            lines.forEachIndexed { index, line ->
                val word = line.trim()
                vocab[word] = index
                invVocab[index] = word
            }
        }
    }

    constructor(vocabFile: File) : this(vocabFile.inputStream())

    fun tokenize(text: String): List<Int> {
        val tokens = mutableListOf<Int>()
        tokens.add(vocab["[CLS]"] ?: 101) // Standard BERT CLS token

        // Simple pre-tokenization: lowercase, split punctuation
        val cleanText = text.lowercase(Locale.ROOT)
        val words = cleanText.split(Regex("(?<=\\p{Punct})|(?=\\p{Punct})|\\s+"))
            .filter { it.isNotBlank() }

        for (word in words) {
            val wordPieces = wordPieceTokenize(word)
            for (piece in wordPieces) {
                tokens.add(vocab[piece] ?: (vocab["[UNK]"] ?: 100))
            }
        }

        tokens.add(vocab["[SEP]"] ?: 102) // Standard BERT SEP token
        return tokens
    }

    private fun wordPieceTokenize(word: String): List<String> {
        if (vocab.containsKey(word)) {
            return listOf(word)
        }

        val result = mutableListOf<String>()
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
        val inputIds = LongArray(maxLength) { 0 }
        val attentionMask = LongArray(maxLength) { 0 }

        val actualLength = minOf(tokens.size, maxLength)
        for (i in 0 until actualLength) {
            inputIds[i] = tokens[i].toLong()
            attentionMask[i] = 1L
        }

        return Pair(inputIds, attentionMask)
    }
}
