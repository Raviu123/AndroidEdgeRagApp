package com.siddu.offlinedocai

import android.content.Context
import com.siddu.offlinedocai.ai.llm.GGUFLLM
import com.siddu.offlinedocai.ai.retrieval.VectorStore
import com.siddu.offlinedocai.data.dao.ChunkDao
import com.siddu.offlinedocai.data.entities.ChunkEntity
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test
import java.io.File

class RAGRetrievalRelevanceTest {

    // 1. Realistic NASA STI Document Chunks (The exact observed user scenario)
    private val chunk1_Boilerplate = ChunkEntity(
        id = 101,
        documentId = 1,
        pageNumber = 1,
        text = "National Aeronautics and Space Administration (NASA). Since its inception, the Space Administration (NASA) has been dedicated to the advancement of aeronautics and space science. The NASA Scientific and Technical Information (STI) Program plays a key part in NASA's mission to contribute to the advancement of scientific knowledge and aerospace systems.",
        embedding = null
    )

    private val chunk2_ActualMissions = ChunkEntity(
        id = 102,
        documentId = 1,
        pageNumber = 4,
        text = "Major NASA Space Exploration Missions: The Apollo 11 lunar landing, Voyager 1 and 2 interstellar probes, Hubble Space Telescope deployment, and the Artemis lunar exploration program represent historic milestones in deep space missions.",
        embedding = null
    )

    private val chunk3_GeneralAerospace = ChunkEntity(
        id = 103,
        documentId = 1,
        pageNumber = 2,
        text = "Aerospace technology and aerodynamics research report on supersonic flow dynamics and thermal insulation systems for suborbital atmospheric reentry vehicles.",
        embedding = null
    )

    // 2. Hardware Spec Document Chunks (Related-but-wrong vs Exact Answer)
    private val chunk4_BatterySafetyGuidelines = ChunkEntity(
        id = 201,
        documentId = 2,
        pageNumber = 2,
        text = "Battery Safety and Handling Guidelines: Do not expose battery to high temperatures above 60C. Always use certified chargers to prevent battery degradation and thermal runaway.",
        embedding = null
    )

    private val chunk5_BatteryExactSpecs = ChunkEntity(
        id = 202,
        documentId = 2,
        pageNumber = 5,
        text = "Hardware Specifications: The device features a 5000 mAh lithium-polymer battery with 45W fast charging support and 15W Qi wireless charging capability.",
        embedding = null
    )

    // 3. Software Engineering Document Chunks
    private val chunk6_CompanyPreamble = ChunkEntity(
        id = 301,
        documentId = 3,
        pageNumber = 1,
        text = "Acme Global Solutions Annual Engineering Report. Table of contents and legal disclaimers: All rights reserved. Published by Acme Engineering Division. Our corporate mission is empowering global teams through scalable enterprise software architectures.",
        embedding = null
    )

    private val chunk7_ProjectsList = ChunkEntity(
        id = 302,
        documentId = 3,
        pageNumber = 3,
        text = "Key Engineering Projects: 1. Smart Traffic Optimizer: Implemented real-time traffic signal regulation using computer vision and edge AI. 2. OfflineDocAI: Developed on-device private document processing engine with local LLM and vector search.",
        embedding = null
    )

    private val chunk8_CertificationsList = ChunkEntity(
        id = 303,
        documentId = 3,
        pageNumber = 6,
        text = "Professional Certifications: AWS Certified Solutions Architect Associate, Google Cloud Professional Data Engineer, and Certified Kubernetes Administrator (CKA).",
        embedding = null
    )

    private val chunk9_AuthMechanism = ChunkEntity(
        id = 304,
        documentId = 3,
        pageNumber = 4,
        text = "Authentication Architecture: The API uses stateless JSON Web Tokens (JWT) signed with RS256 private keys, with automatic token rotation every 15 minutes and refresh tokens stored in encrypted HTTP-only cookies.",
        embedding = null
    )

    private val chunk10_SkillsList = ChunkEntity(
        id = 305,
        documentId = 3,
        pageNumber = 2,
        text = "Technical Skills: Kotlin, Java, Python, C++, Android NDK, PyTorch Mobile, Vector Search, SQLite, Room DB, Docker.",
        embedding = null
    )

    // 4. Multi-Page Document Chunks (The exact observed user failure case: Page 10 vs Page 328 & 341)
    private val chunk11_Page10_Content = ChunkEntity(
        id = 401,
        documentId = 4,
        pageNumber = 10,
        text = "Quantum Encryption Protocols: Page 10 outlines post-quantum lattice-based cryptography, detailing Kyber key encapsulation and Dilithium digital signatures for next-generation network communication security.",
        embedding = null
    )

    private val chunk12_Page328_Generic = ChunkEntity(
        id = 402,
        documentId = 4,
        pageNumber = 328,
        text = "Index and Cross-References: For additional security considerations and encryption parameters, refer to page 10 and Appendix B on secure key management.",
        embedding = null
    )

    private val chunk13_Page341_Generic = ChunkEntity(
        id = 403,
        documentId = 4,
        pageNumber = 341,
        text = "Document Bibliography and Historical Citations: Page 10 of NIST Special Publication 800-56A discusses deterministic random number generators.",
        embedding = null
    )

    private fun createMockContext(): Context {
        val tempDir = File(System.getProperty("java.io.tmpdir"), "offlinedocai_rag_test")
        tempDir.mkdirs()
        return object : android.content.ContextWrapper(null) {
            override fun getFilesDir(): File = tempDir
        }
    }

    private fun createMockChunkDao(chunkList: List<ChunkEntity>): ChunkDao {
        return object : ChunkDao {
            override suspend fun insertAll(chunks: List<ChunkEntity>) {}
            override fun getChunksForDocument(docId: Long): Flow<List<ChunkEntity>> = flowOf(chunkList.filter { it.documentId == docId })
            override suspend fun getAllChunksForDocumentIds(docIds: List<Long>): List<ChunkEntity> = chunkList.filter { it.documentId in docIds }
        }
    }

    @Test
    fun test1_WhichSpaceMissions_ActualMissionsOutranksSTI_Boilerplate() = runBlocking {
        val dao = createMockChunkDao(listOf(chunk1_Boilerplate, chunk2_ActualMissions, chunk3_GeneralAerospace))
        val vectorStore = VectorStore(dao)

        val retrieved = vectorStore.retrieve(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(1L),
            topK = 5,
            query = "which space missions"
        )

        // 1. Verify that Chunk 2 (Actual Missions) is ranked #1 and Chunk 1 (STI boilerplate) is NOT ranked #1
        assertTrue("Retrieved list must not be empty", retrieved.isNotEmpty())
        assertEquals("Chunk 2 (Page 4 actual missions) must be ranked #1", 102L, retrieved.first().id)

        // 2. Synthesize with GGUFLLM
        val llm = GGUFLLM(createMockContext())
        val contextText = retrieved.joinToString("\n\n") { "[Page ${it.pageNumber}] ${it.text}" }
        val prompt = """
            You are an offline AI assistant. Use the following extracted document context to answer the user's question.

            Context:
            $contextText

            Question: which space missions

            Answer concisely, referencing the page numbers when appropriate.
        """.trimIndent()

        val answer = llm.generate(prompt)
        println("=== TEST 1: which space missions ===")
        println("RETRIEVED TOP CHUNK: ID=${retrieved.first().id}, Page=${retrieved.first().pageNumber}")
        println("ANSWER:\n$answer\n")

        assertTrue("Answer must contain Apollo", answer.contains("Apollo"))
        assertTrue("Answer must contain Voyager", answer.contains("Voyager"))
        assertTrue("Answer must contain Artemis", answer.contains("Artemis"))
        assertTrue("Answer must cite Page 4", answer.contains("(Page 4)"))
        assertFalse("Answer must NOT contain STI boilerplate", answer.contains("Scientific and Technical Information") || answer.contains("STI Program"))
    }

    @Test
    fun test2_ExactAnswerExists_HardwareBatterySpecs() = runBlocking {
        val dao = createMockChunkDao(listOf(chunk4_BatterySafetyGuidelines, chunk5_BatteryExactSpecs))
        val vectorStore = VectorStore(dao)

        val retrieved = vectorStore.retrieve(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(2L),
            topK = 5,
            query = "What is the battery capacity?"
        )

        assertTrue(retrieved.isNotEmpty())
        assertEquals("Exact battery specs chunk (Page 5) must be ranked #1", 202L, retrieved.first().id)

        val llm = GGUFLLM(createMockContext())
        val contextText = retrieved.joinToString("\n\n") { "[Page ${it.pageNumber}] ${it.text}" }
        val prompt = """
            Context:
            $contextText

            Question: What is the battery capacity?
        """.trimIndent()

        val answer = llm.generate(prompt)
        println("=== TEST 2: Battery Capacity ===")
        println("ANSWER:\n$answer\n")
        assertTrue("Answer must contain 5000 mAh", answer.contains("5000 mAh"))
        assertTrue("Answer must cite Page 5", answer.contains("(Page 5)"))
        assertFalse("Must not include safety warning text", answer.contains("thermal runaway"))
    }

    @Test
    fun test3_RelatedButWrongChunk_SafetyVsSpecs() = runBlocking {
        val dao = createMockChunkDao(listOf(chunk4_BatterySafetyGuidelines, chunk5_BatteryExactSpecs))
        val vectorStore = VectorStore(dao)

        val retrieved = vectorStore.retrieve(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(2L),
            topK = 5,
            query = "What temperature damages the battery?"
        )

        assertTrue(retrieved.isNotEmpty())
        assertEquals("Safety guideline chunk (Page 2) must be ranked #1", 201L, retrieved.first().id)

        val llm = GGUFLLM(createMockContext())
        val contextText = retrieved.joinToString("\n\n") { "[Page ${it.pageNumber}] ${it.text}" }
        val prompt = """
            Context:
            $contextText

            Question: What temperature damages the battery?
        """.trimIndent()

        val answer = llm.generate(prompt)
        println("=== TEST 3: Battery Temperature ===")
        println("ANSWER:\n$answer\n")
        assertTrue("Answer must contain 60C", answer.contains("60C") || answer.contains("60"))
        assertTrue("Answer must cite Page 2", answer.contains("(Page 2)"))
    }

    @Test
    fun test4_UnrelatedQuery_ChocolateCakeRecipe() = runBlocking {
        val dao = createMockChunkDao(listOf(chunk1_Boilerplate, chunk2_ActualMissions, chunk4_BatterySafetyGuidelines))
        val vectorStore = VectorStore(dao)

        val retrieved = vectorStore.retrieve(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(1L, 2L),
            topK = 5,
            query = "What is the recipe for chocolate cake?"
        )

        val llm = GGUFLLM(createMockContext())
        val contextText = if (retrieved.isNotEmpty()) {
            retrieved.joinToString("\n\n") { "[Page ${it.pageNumber}] ${it.text}" }
        } else {
            "[NO_EVIDENCE_FOUND]"
        }

        val prompt = """
            Context:
            $contextText

            Question: What is the recipe for chocolate cake?
        """.trimIndent()

        val answer = llm.generate(prompt)
        println("=== TEST 4: Unrelated Query ===")
        println("ANSWER:\n$answer\n")
        assertTrue("Must return grounded not found response", answer.contains("could not find information") || answer.contains("I searched the provided document"))
        assertFalse("Must NOT dump NASA or battery chunks", answer.contains("NASA") || answer.contains("5000 mAh"))
    }

    @Test
    fun test5_QueryWithNoAnswerInDocuments_SpaceX_CEO() = runBlocking {
        val dao = createMockChunkDao(listOf(chunk4_BatterySafetyGuidelines, chunk5_BatteryExactSpecs))
        val vectorStore = VectorStore(dao)

        val retrieved = vectorStore.retrieve(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(2L),
            topK = 5,
            query = "Who is the CEO of SpaceX?"
        )

        val llm = GGUFLLM(createMockContext())
        val contextText = if (retrieved.isNotEmpty()) {
            retrieved.joinToString("\n\n") { "[Page ${it.pageNumber}] ${it.text}" }
        } else {
            "[NO_EVIDENCE_FOUND]"
        }

        val prompt = """
            Context:
            $contextText

            Question: Who is the CEO of SpaceX?
        """.trimIndent()

        val answer = llm.generate(prompt)
        println("=== TEST 5: No Answer In Documents ===")
        println("ANSWER:\n$answer\n")
        assertTrue("Must return grounded not found response", answer.contains("could not find information") || answer.contains("I searched the provided document"))
    }

    @Test
    fun test6_Generalization_WhichProjects() = runBlocking {
        val dao = createMockChunkDao(listOf(chunk6_CompanyPreamble, chunk7_ProjectsList, chunk8_CertificationsList))
        val vectorStore = VectorStore(dao)

        val retrieved = vectorStore.retrieve(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(3L),
            topK = 5,
            query = "which projects"
        )

        assertTrue("Must retrieve projects chunk", retrieved.isNotEmpty())
        assertEquals("Projects chunk (Page 3) must be ranked #1", 302L, retrieved.first().id)

        val llm = GGUFLLM(createMockContext())
        val contextText = retrieved.joinToString("\n\n") { "[Page ${it.pageNumber}] ${it.text}" }
        val prompt = """
            Context:
            $contextText

            Question: which projects
        """.trimIndent()

        val answer = llm.generate(prompt)
        println("=== TEST 6: Which Projects ===")
        println("ANSWER:\n$answer\n")
        assertTrue("Answer must contain Smart Traffic Optimizer", answer.contains("Smart Traffic Optimizer"))
        assertTrue("Answer must contain OfflineDocAI", answer.contains("OfflineDocAI"))
        assertFalse("Answer must not contain company legal preamble", answer.contains("Table of contents") || answer.contains("All rights reserved"))
    }

    @Test
    fun test7_Generalization_WhichCertifications() = runBlocking {
        val dao = createMockChunkDao(listOf(chunk6_CompanyPreamble, chunk8_CertificationsList))
        val vectorStore = VectorStore(dao)

        val retrieved = vectorStore.retrieve(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(3L),
            topK = 5,
            query = "which certifications"
        )

        assertTrue(retrieved.isNotEmpty())
        assertEquals("Certifications chunk (Page 6) must be ranked #1", 303L, retrieved.first().id)

        val llm = GGUFLLM(createMockContext())
        val contextText = retrieved.joinToString("\n\n") { "[Page ${it.pageNumber}] ${it.text}" }
        val prompt = """
            Context:
            $contextText

            Question: which certifications
        """.trimIndent()

        val answer = llm.generate(prompt)
        println("=== TEST 7: Which Certifications ===")
        println("ANSWER:\n$answer\n")
        assertTrue("Must contain AWS Certified", answer.contains("AWS Certified") || answer.contains("Solutions Architect"))
        assertTrue("Must contain Kubernetes", answer.contains("Kubernetes") || answer.contains("CKA"))
    }

    @Test
    fun test8_Generalization_WhatAreTheSkills() = runBlocking {
        val dao = createMockChunkDao(listOf(chunk6_CompanyPreamble, chunk10_SkillsList))
        val vectorStore = VectorStore(dao)

        val retrieved = vectorStore.retrieve(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(3L),
            topK = 5,
            query = "what are the skills"
        )

        assertTrue(retrieved.isNotEmpty())
        assertEquals("Skills chunk (Page 2) must be ranked #1", 305L, retrieved.first().id)

        val llm = GGUFLLM(createMockContext())
        val contextText = retrieved.joinToString("\n\n") { "[Page ${it.pageNumber}] ${it.text}" }
        val prompt = """
            Context:
            $contextText

            Question: what are the skills
        """.trimIndent()

        val answer = llm.generate(prompt)
        println("=== TEST 8: What Are The Skills ===")
        println("ANSWER:\n$answer\n")
        assertTrue("Must contain Kotlin", answer.contains("Kotlin"))
        assertTrue("Must contain Python", answer.contains("Python"))
        assertTrue("Must cite Page 2", answer.contains("(Page 2)"))
    }

    @Test
    fun test9_Generalization_GenericFactualAuth() = runBlocking {
        val dao = createMockChunkDao(listOf(chunk6_CompanyPreamble, chunk9_AuthMechanism))
        val vectorStore = VectorStore(dao)

        val retrieved = vectorStore.retrieve(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(3L),
            topK = 5,
            query = "How does token authentication work?"
        )

        assertTrue(retrieved.isNotEmpty())
        assertEquals("Auth mechanism chunk (Page 4) must be ranked #1", 304L, retrieved.first().id)

        val llm = GGUFLLM(createMockContext())
        val contextText = retrieved.joinToString("\n\n") { "[Page ${it.pageNumber}] ${it.text}" }
        val prompt = """
            Context:
            $contextText

            Question: How does token authentication work?
        """.trimIndent()

        val answer = llm.generate(prompt)
        println("=== TEST 9: Factual Auth Mechanism ===")
        println("ANSWER:\n$answer\n")
        assertTrue("Must contain JWT / RS256", answer.contains("JWT") || answer.contains("JSON Web Tokens") || answer.contains("RS256"))
        assertTrue("Must cite Page 4", answer.contains("(Page 4)"))
    }

    @Test
    fun test10_PageLookup_WhatIsThereInPage10_ReturnsOnlyPage10() = runBlocking {
        // Reproduce exact user scenario: document has Page 10, Page 328, and Page 341.
        val dao = createMockChunkDao(listOf(chunk11_Page10_Content, chunk12_Page328_Generic, chunk13_Page341_Generic))
        val vectorStore = VectorStore(dao)

        val retrieved = vectorStore.retrieve(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(4L),
            topK = 5,
            query = "what is there in page 10"
        )

        // 1. Verify that ONLY Page 10 is retrieved (No Page 328, No Page 341)
        assertTrue("Must retrieve Page 10 chunks", retrieved.isNotEmpty())
        assertTrue("All retrieved chunks must strictly be Page 10", retrieved.all { it.pageNumber == 10 })
        assertEquals("Chunk 11 (Page 10) must be retrieved", 401L, retrieved.first().id)
        assertFalse("Must NOT retrieve Page 328", retrieved.any { it.pageNumber == 328 })
        assertFalse("Must NOT retrieve Page 341", retrieved.any { it.pageNumber == 341 })

        // 2. Synthesize with GGUFLLM
        val llm = GGUFLLM(createMockContext())
        val contextText = retrieved.joinToString("\n\n") { "[Page ${it.pageNumber}] ${it.text}" }
        val prompt = """
            Context:
            $contextText

            Question: what is there in page 10
        """.trimIndent()

        val answer = llm.generate(prompt)
        println("=== TEST 10: what is there in page 10 ===")
        println("RETRIEVED PAGES: ${retrieved.map { it.pageNumber }}")
        println("ANSWER:\n$answer\n")

        assertTrue("Answer must mention Quantum Encryption Protocols / Kyber", answer.contains("Quantum Encryption") || answer.contains("Kyber"))
        assertTrue("Answer must cite Page 10", answer.contains("(Page 10)") || answer.contains("Page 10"))
        assertFalse("Answer must NOT mention Page 328 or Page 341 content", answer.contains("Appendix B") || answer.contains("NIST Special Publication"))
    }

    @Test
    fun test11_PageLookup_WhatIsOnPage10_ReturnsOnlyPage10() = runBlocking {
        val dao = createMockChunkDao(listOf(chunk11_Page10_Content, chunk12_Page328_Generic, chunk13_Page341_Generic))
        val vectorStore = VectorStore(dao)

        val retrieved = vectorStore.retrieve(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(4L),
            topK = 5,
            query = "what is on page 10"
        )

        assertTrue(retrieved.isNotEmpty())
        assertTrue("All retrieved chunks must strictly be Page 10", retrieved.all { it.pageNumber == 10 })

        val llm = GGUFLLM(createMockContext())
        val contextText = retrieved.joinToString("\n\n") { "[Page ${it.pageNumber}] ${it.text}" }
        val prompt = """
            Context:
            $contextText

            Question: what is on page 10
        """.trimIndent()

        val answer = llm.generate(prompt)
        println("=== TEST 11: what is on page 10 ===")
        println("ANSWER:\n$answer\n")
        assertTrue("Must contain Kyber / Dilithium", answer.contains("Kyber") || answer.contains("Dilithium"))
        assertTrue("Must cite Page 10", answer.contains("Page 10"))
    }

    @Test
    fun test12_PageLookup_SummarizePage10_ReturnsOnlyPage10() = runBlocking {
        val dao = createMockChunkDao(listOf(chunk11_Page10_Content, chunk12_Page328_Generic, chunk13_Page341_Generic))
        val vectorStore = VectorStore(dao)

        val retrieved = vectorStore.retrieve(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(4L),
            topK = 5,
            query = "summarize page 10"
        )

        assertTrue(retrieved.isNotEmpty())
        assertTrue("All retrieved chunks must strictly be Page 10", retrieved.all { it.pageNumber == 10 })

        val llm = GGUFLLM(createMockContext())
        val contextText = retrieved.joinToString("\n\n") { "[Page ${it.pageNumber}] ${it.text}" }
        val prompt = """
            Context:
            $contextText

            Question: summarize page 10
        """.trimIndent()

        val answer = llm.generate(prompt)
        println("=== TEST 12: summarize page 10 ===")
        println("ANSWER:\n$answer\n")
        assertTrue("Must contain page 10 content", answer.contains("Quantum Encryption") || answer.contains("Kyber"))
        assertTrue("Must cite Page 10", answer.contains("Page 10"))
    }

    @Test
    fun test13_PageLookup_PageDoesNotExist_ReturnsGroundedNotFound() = runBlocking {
        val dao = createMockChunkDao(listOf(chunk11_Page10_Content, chunk12_Page328_Generic))
        val vectorStore = VectorStore(dao)

        val retrieved = vectorStore.retrieve(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(4L),
            topK = 5,
            query = "what is on page 999"
        )

        // Must be empty because page 999 does not exist
        assertTrue("Retrieved list must be empty for non-existent page", retrieved.isEmpty())

        val llm = GGUFLLM(createMockContext())
        val contextText = if (retrieved.isNotEmpty()) {
            retrieved.joinToString("\n\n") { "[Page ${it.pageNumber}] ${it.text}" }
        } else {
            "[PAGE_NOT_FOUND:999]"
        }

        val prompt = """
            Context:
            $contextText

            Question: what is on page 999
        """.trimIndent()

        val answer = llm.generate(prompt)
        println("=== TEST 13: Page 999 Does Not Exist ===")
        println("ANSWER:\n$answer\n")
        assertTrue("Must return grounded not found message", answer.contains("couldn't find page 999") || answer.contains("could not find page 999"))
        assertFalse("Must NOT dump Page 10 or Page 328", answer.contains("Quantum Encryption") || answer.contains("Appendix B"))
    }

    @Test
    fun test14_SemanticQueryWithNumber_DoesNotTriggerPageLookup() = runBlocking {
        val dao = createMockChunkDao(listOf(chunk1_Boilerplate, chunk2_ActualMissions, chunk5_BatteryExactSpecs))
        val vectorStore = VectorStore(dao)

        // Query with a number, but without "page" keyword -> must run semantic search
        val retrieved = vectorStore.retrieve(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(1L, 2L),
            topK = 5,
            query = "Tell me about Apollo 11"
        )

        assertTrue("Must retrieve relevant Apollo 11 chunk", retrieved.isNotEmpty())
        assertEquals("Apollo 11 chunk (Page 4) must be retrieved", 102L, retrieved.first().id)
    }
}
