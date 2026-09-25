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

    @Test
    fun test15_PageLookup_NASA_SE_Handbook_TOC_Synthesis_CleansDottedLeaders() = runBlocking {
        val chunkTOC = ChunkEntity(
            id = 501,
            documentId = 5,
            pageNumber = 10,
            text = """
NASA Systems Engineering Handbook
Table of Contents
7.4.1 Basic HF Model .................... 142
7.4.2 HF Analysis and Evaluation Techniques .................. 145
7.4.3 Human Factors Engineering (HFE) in Design ................ 148
            """.trimIndent(),
            embedding = null
        )
        val dao = createMockChunkDao(listOf(chunkTOC))
        val vectorStore = VectorStore(dao)

        val retrieved = vectorStore.retrieve(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(5L),
            topK = 5,
            query = "what is there in 10th page"
        )

        assertTrue(retrieved.isNotEmpty())
        assertEquals(10, retrieved.first().pageNumber)

        val llm = GGUFLLM(createMockContext())
        val contextText = retrieved.joinToString("\n\n") { "[Page ${it.pageNumber}] ${it.text}" }
        val prompt = """
            Context:
            $contextText

            Question: what is there in 10th page
        """.trimIndent()

        val answer = llm.generate(prompt)
        println("=== TEST 15: NASA SE Handbook TOC Synthesis ===")
        println("ANSWER:\n$answer\n")

        // 1. Must be grounded in Page 10
        assertTrue("Answer must cite Page 10", answer.contains("Page 10"))
        assertTrue("Must contain Basic HF Model", answer.contains("Basic HF Model"))
        assertTrue("Must contain HF Analysis", answer.contains("HF Analysis and Evaluation Techniques"))

        // 2. Dotted leader artifacts must be completely eliminated
        assertFalse("Must NOT contain dotted leader sequences", answer.contains("......"))
        assertFalse("Must NOT contain 3 or more consecutive dots", Regex("""\.{3,}""").containsMatchIn(answer))

        // 3. Must be structured cleanly
        assertTrue("Must format TOC sections cleanly", answer.contains("• 7.4.1 Basic HF Model — Page 142") || answer.contains("7.4.1 Basic HF Model"))
    }

    // =========================================================================
    // GENERAL-PURPOSE BOOK RAG REGRESSION SUITE (CATEGORIES A THROUGH L)
    // =========================================================================

    // Realistic NASA Systems Engineering Handbook Chunks
    private val se_p3_SE_Engine = ChunkEntity(
        id = 601, documentId = 10, pageNumber = 3,
        text = "The Systems Engineering (SE) engine is a cyclical, iterative process consisting of 17 processes categorized into System Design Processes, Product Realization Processes, and Technical Management Processes. It is applied recursively throughout the system structure from the top-level mission down to subsystem components.",
        embedding = null
    )

    private val se_p15_Chap2_Intro = ChunkEntity(
        id = 602, documentId = 10, pageNumber = 15,
        text = "Chapter 2: Fundamentals of Systems Engineering. This chapter describes the systems engineering discipline, the role of systems engineers, and the core SE engine. Systems engineering provides the technical foundation for the successful realization of complex aerospace systems.",
        embedding = null
    )

    private val se_p16_Chap2_Concepts = ChunkEntity(
        id = 603, documentId = 10, pageNumber = 16,
        text = "Chapter 2 emphasizes the relationship between project management and systems engineering, iterative decision-making, stakeholder satisfaction, and rigorous trade studies to manage system complexity.",
        embedding = null
    )

    private val se_p18_LifeCycle = ChunkEntity(
        id = 604, documentId = 10, pageNumber = 18,
        text = "The systems engineering life cycle spans seven distinct project phases, progressing from concept studies (Pre-Phase A) through concept development (Phase A), preliminary design (Phase B), final design (Phase C), fabrication and integration (Phase D), operations (Phase E), to closeout (Phase F).",
        embedding = null
    )

    private val se_p20_PrePhaseA = ChunkEntity(
        id = 605, documentId = 10, pageNumber = 20,
        text = "The primary purpose of Pre-Phase A (Concept Studies) is to formulate broad mission objectives, assess feasibility, explore alternative concept architectures, and identify initial technical risks and required technology investments.",
        embedding = null
    )

    private val se_p21_PhaseA_1 = ChunkEntity(
        id = 606, documentId = 10, pageNumber = 21,
        text = "Phase A (Concept and Technology Development) determines the feasible system concept, defines mission architecture, establishes top-level system requirements, and develops initial risk management plans.",
        embedding = null
    )

    private val se_p22_PhaseA_2 = ChunkEntity(
        id = 607, documentId = 10, pageNumber = 22,
        text = "During Phase A, formal trade studies evaluate alternative designs and the System Requirements Review (SRR) is conducted to ensure requirements baseline readiness before preliminary design.",
        embedding = null
    )

    private val se_p23_Page23 = ChunkEntity(
        id = 608, documentId = 10, pageNumber = 23,
        text = "Section 3.1: System Requirements Review (SRR) Criteria and Entrance Guidelines. The SRR ensures functional and performance requirements are mature.",
        embedding = null
    )

    private val se_p25_PhaseB = ChunkEntity(
        id = 609, documentId = 10, pageNumber = 25,
        text = "Phase B (Preliminary Design and Technology Completion) establishes the preliminary design, freezes the baseline requirements, demonstrates critical technology maturity at TRL 6, and culminates in the Preliminary Design Review (PDR).",
        embedding = null
    )

    private val se_p40_RequirementsDef = ChunkEntity(
        id = 610, documentId = 10, pageNumber = 40,
        text = "Technical Requirements Definition Process transforms stakeholder expectations into clear, unambiguous, verifiable, and allocated technical specifications across system hierarchies.",
        embedding = null
    )

    private val se_p88_Verification = ChunkEntity(
        id = 611, documentId = 10, pageNumber = 88,
        text = "Section 4.3: System Verification Process provides evidence that the delivered system complies with its specified requirements through test, analysis, inspection, or demonstration.",
        embedding = null
    )

    private val se_p104_Validation = ChunkEntity(
        id = 612, documentId = 10, pageNumber = 104,
        text = "Section 5.4: System Validation Process demonstrates that the realized system functions as intended in its operational environment to fulfill stakeholder expectations and mission objectives.",
        embedding = null
    )

    private val se_p120_TechMgmtProcesses = ChunkEntity(
        id = 613, documentId = 10, pageNumber = 120,
        text = "Technical Management Processes include: 1. Technical Planning, 2. Requirements Management, 3. Interface Management, 4. Technical Risk Management, 5. Configuration Management, 6. Technical Data Management, 7. Technical Assessment, and 8. Decision Analysis.",
        embedding = null
    )

    private val se_p135_ReqMgmt_Config = ChunkEntity(
        id = 614, documentId = 10, pageNumber = 135,
        text = "Requirements Management and Configuration Management ensure baseline control, trace requirements from top-level objectives to component tests, and track changes throughout the system life cycle.",
        embedding = null
    )

    private val allSEHandbookChunks = listOf(
        se_p3_SE_Engine, se_p15_Chap2_Intro, se_p16_Chap2_Concepts, se_p18_LifeCycle,
        se_p20_PrePhaseA, se_p21_PhaseA_1, se_p22_PhaseA_2, se_p23_Page23,
        se_p25_PhaseB, se_p40_RequirementsDef, se_p88_Verification, se_p104_Validation,
        se_p120_TechMgmtProcesses, se_p135_ReqMgmt_Config
    )

    @Test
    fun testA_DirectFactualQuestion_PrePhaseA_Purpose() = runBlocking {
        val dao = createMockChunkDao(allSEHandbookChunks)
        val vectorStore = VectorStore(dao)

        val retrieved = vectorStore.retrieve(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(10L),
            topK = 5,
            query = "What is the purpose of Pre-Phase A?"
        )

        assertTrue(retrieved.isNotEmpty())
        assertEquals(20, retrieved.first().pageNumber)

        val llm = GGUFLLM(createMockContext())
        val contextText = retrieved.joinToString("\n\n") { "[Page ${it.pageNumber}] ${it.text}" }
        val prompt = """
            Context:
            $contextText

            Question: What is the purpose of Pre-Phase A?
        """.trimIndent()

        val answer = llm.generate(prompt)
        println("=== TEST A: Direct Factual (Pre-Phase A) ===")
        println("ANSWER:\n$answer\n")

        assertTrue("Answer must mention concept studies / feasibility", answer.contains("Concept Studies") || answer.contains("feasibility"))
        assertTrue("Answer must cite Page 20", answer.contains("(Page 20)"))
    }

    @Test
    fun testB_ConceptExplanation_SystemsEngineeringLifeCycle() = runBlocking {
        val dao = createMockChunkDao(allSEHandbookChunks)
        val vectorStore = VectorStore(dao)

        val retrieved = vectorStore.retrieve(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(10L),
            topK = 5,
            query = "Explain the systems engineering life cycle."
        )

        assertTrue(retrieved.isNotEmpty())
        val llm = GGUFLLM(createMockContext())
        val contextText = retrieved.joinToString("\n\n") { "[Page ${it.pageNumber}] ${it.text}" }
        val prompt = """
            Context:
            $contextText

            Question: Explain the systems engineering life cycle.
        """.trimIndent()

        val answer = llm.generate(prompt)
        println("=== TEST B: Concept Explanation (Life Cycle) ===")
        println("ANSWER:\n$answer\n")

        assertTrue("Answer must mention phases", answer.contains("phases") || answer.contains("Phase A") || answer.contains("life cycle"))
        assertTrue("Answer must cite Page 18", answer.contains("(Page 18)"))
    }

    @Test
    fun testC_Definition_WhatIsTheSEEngine() = runBlocking {
        val dao = createMockChunkDao(allSEHandbookChunks)
        val vectorStore = VectorStore(dao)

        val retrieved = vectorStore.retrieve(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(10L),
            topK = 5,
            query = "What is the SE engine?"
        )

        assertTrue(retrieved.isNotEmpty())
        assertEquals(3, retrieved.first().pageNumber)

        val llm = GGUFLLM(createMockContext())
        val contextText = retrieved.joinToString("\n\n") { "[Page ${it.pageNumber}] ${it.text}" }
        val prompt = """
            Context:
            $contextText

            Question: What is the SE engine?
        """.trimIndent()

        val answer = llm.generate(prompt)
        println("=== TEST C: Definition (SE Engine) ===")
        println("ANSWER:\n$answer\n")

        assertTrue("Answer must define SE engine", answer.contains("SE") || answer.contains("Systems Engineering"))
        assertTrue("Answer must mention processes", answer.contains("processes") || answer.contains("System Design"))
        assertTrue("Answer must cite Page 3", answer.contains("(Page 3)"))
    }

    @Test
    fun testD_MultiPageQuestion_WhatDoesTheBookSayAboutRequirements() = runBlocking {
        val dao = createMockChunkDao(allSEHandbookChunks)
        val vectorStore = VectorStore(dao)

        val retrieved = vectorStore.retrieve(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(10L),
            topK = 5,
            query = "What does the book say about requirements?"
        )

        assertTrue(retrieved.isNotEmpty())
        val retrievedPages = retrieved.map { it.pageNumber }.distinct()
        assertTrue("Must retrieve multiple pages addressing requirements", retrievedPages.size >= 2)

        val llm = GGUFLLM(createMockContext())
        val contextText = retrieved.joinToString("\n\n") { "[Page ${it.pageNumber}] ${it.text}" }
        val prompt = """
            Context:
            $contextText

            Question: What does the book say about requirements?
        """.trimIndent()

        val answer = llm.generate(prompt)
        println("=== TEST D: Multi-Page (Requirements) ===")
        println("ANSWER:\n$answer\n")

        assertTrue("Answer must mention requirements", answer.contains("Requirements"))
        assertTrue("Answer must cite at least one of the requirements pages", answer.contains("(Page 40)") || answer.contains("(Page 135)") || answer.contains("(Page 23)"))
    }

    @Test
    fun testE_Comparison_ComparePhaseAAndPhaseB() = runBlocking {
        val dao = createMockChunkDao(allSEHandbookChunks)
        val vectorStore = VectorStore(dao)

        val retrieved = vectorStore.retrieve(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(10L),
            topK = 5,
            query = "Compare Phase A and Phase B."
        )

        assertTrue(retrieved.isNotEmpty())
        val pages = retrieved.map { it.pageNumber }.distinct()
        assertTrue("Must retrieve Phase A (Page 21 or 22)", pages.contains(21) || pages.contains(22))
        assertTrue("Must retrieve Phase B (Page 25)", pages.contains(25))

        val llm = GGUFLLM(createMockContext())
        val contextText = retrieved.joinToString("\n\n") { "[Page ${it.pageNumber}] ${it.text}" }
        val prompt = """
            Context:
            $contextText

            Question: Compare Phase A and Phase B.
        """.trimIndent()

        val answer = llm.generate(prompt)
        println("=== TEST E: Comparison (Phase A vs Phase B) ===")
        println("ANSWER:\n$answer\n")

        assertTrue("Answer must mention Phase A", answer.contains("Phase A"))
        assertTrue("Answer must mention Phase B", answer.contains("Phase B"))
        assertTrue("Answer must cite Page 21 or Page 22", answer.contains("(Page 21)") || answer.contains("(Page 22)"))
        assertTrue("Answer must cite Page 25", answer.contains("(Page 25)"))
    }

    @Test
    fun testF_WhereInTheBook_VerificationAndValidation() = runBlocking {
        val dao = createMockChunkDao(allSEHandbookChunks)
        val vectorStore = VectorStore(dao)

        val retrieved = vectorStore.retrieve(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(10L),
            topK = 5,
            query = "Where does the book discuss verification and validation?"
        )

        assertTrue(retrieved.isNotEmpty())
        val pages = retrieved.map { it.pageNumber }.distinct()
        assertTrue("Must retrieve verification (Page 88)", pages.contains(88))
        assertTrue("Must retrieve validation (Page 104)", pages.contains(104))

        val llm = GGUFLLM(createMockContext())
        val contextText = retrieved.joinToString("\n\n") { "[Page ${it.pageNumber}] ${it.text}" }
        val prompt = """
            Context:
            $contextText

            Question: Where does the book discuss verification and validation?
        """.trimIndent()

        val answer = llm.generate(prompt)
        println("=== TEST F: Where In The Book (Verification & Validation) ===")
        println("ANSWER:\n$answer\n")

        assertTrue("Answer must cite Page 88", answer.contains("Page 88"))
        assertTrue("Answer must cite Page 104", answer.contains("Page 104"))
        assertTrue("Answer must mention verification", answer.contains("Verification") || answer.contains("verification"))
        assertTrue("Answer must mention validation", answer.contains("Validation") || answer.contains("validation"))
    }

    @Test
    fun testG_PageSpecificQuestion_WhatIsOnPage23() = runBlocking {
        val dao = createMockChunkDao(allSEHandbookChunks)
        val vectorStore = VectorStore(dao)

        val retrieved = vectorStore.retrieve(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(10L),
            topK = 5,
            query = "What is on page 23?"
        )

        assertTrue(retrieved.isNotEmpty())
        assertTrue("All retrieved chunks must strictly be Page 23", retrieved.all { it.pageNumber == 23 })

        val llm = GGUFLLM(createMockContext())
        val contextText = retrieved.joinToString("\n\n") { "[Page ${it.pageNumber}] ${it.text}" }
        val prompt = """
            Context:
            $contextText

            Question: What is on page 23?
        """.trimIndent()

        val answer = llm.generate(prompt)
        println("=== TEST G: Page-Specific (Page 23) ===")
        println("ANSWER:\n$answer\n")

        assertTrue("Answer must cite Page 23", answer.contains("Page 23"))
        assertTrue("Answer must mention SRR", answer.contains("SRR") || answer.contains("System Requirements Review"))
    }

    @Test
    fun testH_ChapterLevelSummary_SummarizeChapter2() = runBlocking {
        val dao = createMockChunkDao(allSEHandbookChunks)
        val vectorStore = VectorStore(dao)

        val retrieved = vectorStore.retrieve(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(10L),
            topK = 5,
            query = "Summarize Chapter 2."
        )

        assertTrue(retrieved.isNotEmpty())
        val pages = retrieved.map { it.pageNumber }.distinct()
        assertTrue("Must retrieve Chapter 2 chunks (Page 15 or 16)", pages.contains(15) || pages.contains(16))

        val llm = GGUFLLM(createMockContext())
        val contextText = retrieved.joinToString("\n\n") { "[Page ${it.pageNumber}] ${it.text}" }
        val prompt = """
            Context:
            $contextText

            Question: Summarize Chapter 2.
        """.trimIndent()

        val answer = llm.generate(prompt)
        println("=== TEST H: Chapter Summary (Chapter 2) ===")
        println("ANSWER:\n$answer\n")

        assertTrue("Answer must cite Chapter 2", answer.contains("Chapter 2") || answer.contains("Fundamentals"))
        assertTrue("Answer must cite Page 15 or 16", answer.contains("(Page 15)") || answer.contains("(Page 16)"))
    }

    @Test
    fun testI_ListExtraction_TechnicalManagementProcesses() = runBlocking {
        val dao = createMockChunkDao(allSEHandbookChunks)
        val vectorStore = VectorStore(dao)

        val retrieved = vectorStore.retrieve(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(10L),
            topK = 5,
            query = "What are the technical management processes?"
        )

        assertTrue(retrieved.isNotEmpty())
        assertEquals(120, retrieved.first().pageNumber)

        val llm = GGUFLLM(createMockContext())
        val contextText = retrieved.joinToString("\n\n") { "[Page ${it.pageNumber}] ${it.text}" }
        val prompt = """
            Context:
            $contextText

            Question: What are the technical management processes?
        """.trimIndent()

        val answer = llm.generate(prompt)
        println("=== TEST I: List Extraction (Tech Mgmt Processes) ===")
        println("ANSWER:\n$answer\n")

        assertTrue("Answer must contain Technical Planning", answer.contains("Technical Planning"))
        assertTrue("Answer must contain Configuration Management", answer.contains("Configuration Management"))
        assertTrue("Answer must contain Decision Analysis", answer.contains("Decision Analysis"))
        assertTrue("Answer must cite Page 120", answer.contains("(Page 120)"))
    }

    @Test
    fun testJ_QuestionWhoseAnswerDoesNotExist_SpaceX_CEO() = runBlocking {
        val dao = createMockChunkDao(allSEHandbookChunks)
        val vectorStore = VectorStore(dao)

        val retrieved = vectorStore.retrieve(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(10L),
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
        println("=== TEST J: Non-Existent Answer ===")
        println("ANSWER:\n$answer\n")

        assertTrue("Must state information not found", answer.contains("could not find information") || answer.contains("I searched the provided document"))
    }

    @Test
    fun testK_UnrelatedQuestion_ChocolateCakeRecipe() = runBlocking {
        val dao = createMockChunkDao(allSEHandbookChunks)
        val vectorStore = VectorStore(dao)

        val retrieved = vectorStore.retrieve(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(10L),
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
        println("=== TEST K: Unrelated Question ===")
        println("ANSWER:\n$answer\n")

        assertTrue("Must state information not found", answer.contains("could not find information") || answer.contains("I searched the provided document"))
    }

    @Test
    fun testL_QuestionRequiringMultipleRetrievedChunks_WhatHappensInPhaseA() = runBlocking {
        val dao = createMockChunkDao(allSEHandbookChunks)
        val vectorStore = VectorStore(dao)

        val retrieved = vectorStore.retrieve(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(10L),
            topK = 5,
            query = "What happens in Phase A?"
        )

        assertTrue(retrieved.isNotEmpty())
        val retrievedIds = retrieved.map { it.id }
        assertTrue("Must retrieve Phase A chunk 606 (Page 21)", retrievedIds.contains(606L))
        assertTrue("Must retrieve Phase A chunk 607 (Page 22)", retrievedIds.contains(607L))

        val llm = GGUFLLM(createMockContext())
        val contextText = retrieved.joinToString("\n\n") { "[Page ${it.pageNumber}] ${it.text}" }
        val prompt = """
            Context:
            $contextText

            Question: What happens in Phase A?
        """.trimIndent()

        val answer = llm.generate(prompt)
        println("=== TEST L: Multiple Retrieved Chunks (Phase A) ===")
        println("ANSWER:\n$answer\n")

        assertTrue("Answer must mention Phase A", answer.contains("Phase A"))
        assertTrue("Answer must cite Page 21", answer.contains("(Page 21)"))
        assertTrue("Answer must cite Page 22", answer.contains("(Page 22)"))
        assertTrue("Answer must mention trade studies or SRR", answer.contains("trade studies") || answer.contains("SRR") || answer.contains("System Requirements Review"))
    }
}
