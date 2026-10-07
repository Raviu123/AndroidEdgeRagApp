package com.siddu.offlinedocai.rag

import com.siddu.offlinedocai.ai.chunking.TextChunker
import com.siddu.offlinedocai.ai.embeddings.ONNXEmbeddingEngine
import com.siddu.offlinedocai.domain.rag.ExtractedPage
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class TextChunkerTest {

    private val count: (String) -> Int = TestTokenizer.instance::countTokens

    private fun sentence(i: Int) =
        "Sentence number $i explains a distinct observation about warehouse logistics, delivery schedules and inventory accuracy."

    private fun paragraph(from: Int, n: Int) = (from until from + n).joinToString(" ") { sentence(it) }

    @Test
    fun noChunkExceedsTheEmbeddingModelLimit() {
        val pages = (1..6).map { p -> ExtractedPage(p, paragraph(p * 100, 25) + "\n\n" + paragraph(p * 100 + 50, 3)) }
        val chunks = TextChunker.chunkDocument(pages, count)
        assertTrue(chunks.size > 10)
        for (c in chunks) {
            assertTrue("chunk ${c.index} has ${count(c.text)} tokens", count(c.text) <= TextChunker.DEFAULT_MAX_TOKENS)
            assertTrue(count(c.text) <= ONNXEmbeddingEngine.MAX_CONTENT_TOKENS)
        }
    }

    @Test
    fun consecutiveChunksInASectionShareOverlap() {
        val chunks = TextChunker.chunkDocument(listOf(ExtractedPage(1, paragraph(0, 30))), count)
        assertTrue(chunks.size >= 3)
        for (i in 1 until chunks.size) {
            val firstSentenceOfNext = TextChunker.splitIntoSentences(chunks[i].text).first()
            assertTrue("chunk $i should start with a sentence from chunk ${i - 1}", chunks[i - 1].text.contains(firstSentenceOfNext))
        }
    }

    @Test
    fun everySentenceIsKeptAndNoWordIsSplit() {
        val text = paragraph(0, 40)
        val chunks = TextChunker.chunkDocument(listOf(ExtractedPage(1, text)), count)
        val joined = chunks.joinToString(" ") { it.text }
        for (i in 0 until 40) assertTrue("sentence $i missing", joined.contains(sentence(i)))
    }

    @Test
    fun headingsStartNewChunksAndAreRecordedAsSections() {
        val text = """
            Introduction

            ${paragraph(0, 3)}

            2.1 Shipping Rules

            ${paragraph(10, 3)}
        """.trimIndent()
        val chunks = TextChunker.chunkDocument(listOf(ExtractedPage(1, text)), count)
        assertEquals(listOf("Introduction", "2.1 Shipping Rules"), chunks.map { it.sectionTitle })
        assertTrue(chunks[1].text.startsWith("2.1 Shipping Rules"))
        assertFalse("no overlap across sections", chunks[1].text.contains(sentence(2)))
    }

    @Test
    fun sentencesThatMerelyStartWithANumberAreNotHeadings() {
        val text = "3.5 million parcels were delivered in March, which was a record for the region.\nThe team expects similar volume next year."
        val chunks = TextChunker.chunkDocument(listOf(ExtractedPage(1, text)), count)
        assertEquals(1, chunks.size)
        assertEquals(null, chunks[0].sectionTitle)
    }

    @Test
    fun pagesAndCrossPageParagraphsAreTracked() {
        val pages = listOf(
            ExtractedPage(4, "The committee reviewed the proposal and decided that the"),
            ExtractedPage(5, "budget would be increased by ten percent next year.")
        )
        val chunks = TextChunker.chunkDocument(pages, count)
        assertEquals(1, chunks.size)
        assertEquals("4-5", chunks[0].pageSpan)
        assertTrue(chunks[0].text.contains("decided that the budget would be increased"))
    }

    @Test
    fun tablesAndListsKeepTheirLines() {
        val text = """
            Fees:
            | Plan | Monthly | Annual |
            | Basic | 10 | 100 |
            | Pro | 25 | 250 |

            - Free shipping on orders over 50
            - Returns accepted within 30 days
        """.trimIndent()
        val chunk = TextChunker.chunkDocument(listOf(ExtractedPage(1, text)), count).single()
        assertTrue(chunk.text.contains("| Basic | 10 | 100 |\n| Pro | 25 | 250 |"))
        assertTrue(chunk.text.contains("- Free shipping on orders over 50\n\n- Returns accepted within 30 days"))
    }

    @Test
    fun runOnTextWithoutPunctuationIsSplitAtWordBoundaries() {
        val words = (0 until 900).joinToString(" ") { "word$it" }
        val chunks = TextChunker.chunkDocument(listOf(ExtractedPage(1, words)), count)
        assertTrue(chunks.size > 3)
        assertTrue(chunks.all { count(it.text) <= TextChunker.DEFAULT_MAX_TOKENS })
        assertTrue(chunks.joinToString(" ") { it.text }.contains("word899"))
    }
}
