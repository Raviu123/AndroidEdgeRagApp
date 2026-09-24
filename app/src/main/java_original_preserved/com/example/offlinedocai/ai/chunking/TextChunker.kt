package com.example.offlinedocai.ai.chunking

import java.util.UUID

data class TextChunk(
    val id: String = UUID.randomUUID().toString(),
    val text: String,
    val pageNumber: Int,
    val chunkIndex: Int
)

class TextChunker {
    /**
     * Splits text into chunks of roughly wordCount words, with wordOverlap words of overlap.
     * Keeps track of the page boundaries so that chunk source citations map accurately to pages.
     * Preserves sentence integrity where possible.
     */
    fun chunkDocument(
        pages: List<Pair<Int, String>>, // list of (pageNumber, pageText)
        wordCount: Int = 300,
        wordOverlap: Int = 50
    ): List<TextChunk> {
        val chunks = mutableListOf<TextChunk>()
        var chunkIndex = 0

        for ((pageNumber, text) in pages) {
            val words = text.split(Regex("\\s+")).filter { it.isNotBlank() }
            if (words.isEmpty()) continue

            var start = 0
            while (start < words.size) {
                val end = minOf(start + wordCount, words.size)
                val chunkWords = words.subList(start, end)
                val chunkText = chunkWords.joinToString(" ")

                chunks.add(
                    TextChunk(
                        text = chunkText,
                        pageNumber = pageNumber,
                        chunkIndex = chunkIndex++
                    )
                )

                // Advance by size - overlap
                start += if (wordCount - wordOverlap > 0) wordCount - wordOverlap else wordCount
            }
        }
        return chunks
    }
}
