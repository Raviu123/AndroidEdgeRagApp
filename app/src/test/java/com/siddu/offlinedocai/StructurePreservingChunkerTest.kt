package com.siddu.offlinedocai

import com.siddu.offlinedocai.ai.chunking.TextChunker
import com.siddu.offlinedocai.domain.rag.ExtractedPage
import org.junit.Assert.*
import org.junit.Test

class StructurePreservingChunkerTest {

    @Test
    fun test1_ExactTechnicalSkillsAndProjects_LinesPreserved() {
        val rawInput = """
TECHNICAL SKILLS

Languages:
Kotlin
Java
Python

Projects:
Project A
Project B
        """.trimIndent()

        val chunks = TextChunker.chunkText(rawInput, pageNumber = 1)

        assertFalse("Must generate at least one chunk", chunks.isEmpty())
        val chunk = chunks.first()

        println("=== TEST 1: Exact Technical Skills & Projects Output ===")
        println(chunk.text)
        println("========================================================")

        // 1. Verify exact line breaks between items
        assertTrue("Must preserve Languages followed by Kotlin on next line", chunk.text.contains("Languages:\nKotlin"))
        assertTrue("Must preserve Kotlin followed by Java on next line", chunk.text.contains("Kotlin\nJava"))
        assertTrue("Must preserve Java followed by Python on next line", chunk.text.contains("Java\nPython"))
        assertTrue("Must preserve Projects followed by Project A on next line", chunk.text.contains("Projects:\nProject A"))
        assertTrue("Must preserve Project A followed by Project B on next line", chunk.text.contains("Project A\nProject B"))

        // 2. Explicitly verify it is NOT flattened into single-line run-on words
        assertFalse(
            "Must NOT convert individual lines into flat 'Kotlin Java Python'",
            chunk.text.contains("Kotlin Java Python")
        )
        assertFalse(
            "Must NOT convert individual lines into flat 'Project A Project B'",
            chunk.text.contains("Project A Project B")
        )

        // 3. Verify metadata
        assertEquals("TECHNICAL SKILLS", chunk.sectionTitle)
        assertEquals(1, chunk.pageNumber)
        assertEquals("1", chunk.pageSpan)
    }

    @Test
    fun test2_Item1Item2Item3_LinesPreserved() {
        val rawInput = """
Item 1
Item 2
Item 3
        """.trimIndent()

        val chunks = TextChunker.chunkText(rawInput, pageNumber = 1)
        assertFalse(chunks.isEmpty())
        val chunk = chunks.first()

        println("=== TEST 2: Item 1, Item 2, Item 3 Output ===")
        println(chunk.text)
        println("=============================================")

        assertTrue("Item 1 must be followed by Item 2 on separate line", chunk.text.contains("Item 1\nItem 2"))
        assertTrue("Item 2 must be followed by Item 3 on separate line", chunk.text.contains("Item 2\nItem 3"))
        assertFalse("Must not flatten Item 1 Item 2 Item 3 into one line", chunk.text.contains("Item 1 Item 2 Item 3"))
    }

    @Test
    fun test3_SimpleTablePreserved() {
        val rawInput = """
| Name | Role |
| Alice | Engineer |
| Bob | Architect |
        """.trimIndent()

        val chunks = TextChunker.chunkText(rawInput, pageNumber = 2)
        assertFalse(chunks.isEmpty())
        val chunk = chunks.first()

        println("=== TEST 3: Simple Table Output ===")
        println(chunk.text)
        println("===================================")

        assertTrue(chunk.text.contains("| Name | Role |"))
        assertTrue(chunk.text.contains("| Alice | Engineer |"))
        assertTrue(chunk.text.contains("| Bob | Architect |"))
        assertTrue("Table rows must be separated by newlines", chunk.text.contains("| Name | Role |\n| Alice | Engineer |\n| Bob | Architect |"))
        assertEquals(2, chunk.pageNumber)
        assertEquals("2", chunk.pageSpan)
    }

    @Test
    fun test4_CrossPageContinuation_PageSpanMetadata() {
        val page10 = ExtractedPage(
            pageNumber = 10,
            text = """
Chapter 4: Cryptographic Key Management

The encryption subsystem establishes secure communication channels using quantum-resistant
            """.trimIndent()
        )

        val page11 = ExtractedPage(
            pageNumber = 11,
            text = """
lattice-based algorithms that provide forward secrecy and authentication across all nodes in the distributed network.
            """.trimIndent()
        )

        val chunks = TextChunker.chunkDocument(listOf(page10, page11))
        assertFalse("Must generate chunks", chunks.isEmpty())

        println("=== TEST 4: Cross-Page Continuation Output ===")
        for (c in chunks) {
            println("Chunk #${c.index} | Page=${c.pageNumber} | PageSpan=${c.pageSpan} | Section=${c.sectionTitle}")
            println(c.text)
            println("---")
        }
        println("===============================================")

        val bridgedChunk = chunks.firstOrNull { it.pageSpan == "10-11" }
        assertNotNull("Should create a bridged chunk with pageSpan = 10-11", bridgedChunk)
        assertEquals(10, bridgedChunk!!.pageNumber)
        assertEquals("10-11", bridgedChunk.pageSpan)
        assertTrue("Must join the wrapped sentence smoothly", bridgedChunk.text.contains("quantum-resistant lattice-based"))
        assertEquals("Chapter 4: Cryptographic Key Management", bridgedChunk.sectionTitle)
    }

    @Test
    fun test5_StableSequentialOrdering() {
        val pages = (1..5).map { pageNum ->
            ExtractedPage(
                pageNumber = pageNum,
                text = """
Section $pageNum: Topic $pageNum

This is content for page $pageNum paragraph 1.

This is content for page $pageNum paragraph 2.
                """.trimIndent()
            )
        }

        val chunks = TextChunker.chunkDocument(pages)
        assertFalse(chunks.isEmpty())

        println("=== TEST 5: Global Sequential Indexing ===")
        for (c in chunks) {
            println("Chunk #${c.index} | Page=${c.pageNumber} | Span=${c.pageSpan}")
        }
        println("==========================================")

        for (i in chunks.indices) {
            assertEquals("chunkIndex must be sequential", i, chunks[i].index)
        }
        for (i in 1 until chunks.size) {
            assertTrue("Page numbers must be non-decreasing", chunks[i].pageNumber >= chunks[i - 1].pageNumber)
        }
    }
}
