package com.siddu.offlinedocai.ai.chunking

/**
 * Utility object to split a large text into overlapping word chunks.
 * The default size is 400 words with an 80‑word overlap.
 */
object TextChunker {
    data class Chunk(
        val index: Int,
        val pageNumber: Int,
        val text: String
    )

    /**
     * Split the given [text] into chunks.
     * @param pageNumber page number the text belongs to (used for citation).
     * @param chunkSize number of words per chunk (must be > 0).
     * @param overlap number of words that overlap between consecutive chunks (must be >= 0 and < chunkSize).
     * @return list of [Chunk] objects ordered by index.
     */
    fun chunkText(
        text: String,
        pageNumber: Int,
        chunkSize: Int = 400,
        overlap: Int = 80
    ): List<Chunk> {
        require(chunkSize > 0) { "chunkSize must be > 0" }
        require(overlap >= 0 && overlap < chunkSize) { "overlap must be >=0 and < chunkSize" }
        val words = text.trim().split(Regex("\\s+"))
        if (words.isEmpty()) return emptyList()
        val chunks = mutableListOf<Chunk>()
        var start = 0
        var index = 0
        while (start < words.size) {
            val end = (start + chunkSize).coerceAtMost(words.size)
            val chunkWords = words.subList(start, end)
            val chunkText = chunkWords.joinToString(" ")
            chunks.add(Chunk(index = index, pageNumber = pageNumber, text = chunkText))
            if (end == words.size) break
            start = end - overlap
            index++
        }
        return chunks
    }
}
