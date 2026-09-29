package com.siddu.offlinedocai

import com.siddu.offlinedocai.ai.retrieval.QueryIntent
import com.siddu.offlinedocai.ai.retrieval.RetrievalSource
import com.siddu.offlinedocai.ai.retrieval.VectorStore
import com.siddu.offlinedocai.data.dao.ChunkDao
import com.siddu.offlinedocai.data.entities.ChunkEntity
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test

class VectorStoreIntentRetrievalTest {

    private fun createMockDao(chunkList: List<ChunkEntity>): ChunkDao {
        return object : ChunkDao {
            override suspend fun insertAll(chunks: List<ChunkEntity>) {}
            override fun getChunksForDocument(docId: Long): Flow<List<ChunkEntity>> =
                flowOf(chunkList.filter { it.documentId == docId })

            override suspend fun getAllChunksForDocumentIds(docIds: List<Long>): List<ChunkEntity> =
                chunkList.filter { it.documentId in docIds }

            override suspend fun getChunksForPage(docIds: List<Long>, pageNumber: Int): List<ChunkEntity> =
                chunkList.filter { it.documentId in docIds && it.pageNumber == pageNumber }

            override suspend fun getChunksForPageRange(docIds: List<Long>, startPage: Int, endPage: Int): List<ChunkEntity> =
                chunkList.filter { it.documentId in docIds && it.pageNumber in startPage..endPage }
        }
    }

    // A. PAGE_LOOKUP
    @Test
    fun testA_PageLookup_RetrievesExactPageOnly() = runBlocking {
        val c1 = ChunkEntity(id = 1, documentId = 1, pageNumber = 5, text = "Page 5 content", chunkIndex = 0)
        val c2 = ChunkEntity(id = 2, documentId = 1, pageNumber = 10, text = "Page 10 section A", chunkIndex = 1)
        val c3 = ChunkEntity(id = 3, documentId = 1, pageNumber = 10, text = "Page 10 section B", chunkIndex = 2)
        val c4 = ChunkEntity(id = 4, documentId = 1, pageNumber = 15, text = "Page 15 content", chunkIndex = 3)

        val store = VectorStore(createMockDao(listOf(c1, c2, c3, c4)))
        val results = store.retrieve(FloatArray(0), listOf(1L), topK = 5, query = "What is on page 10?")

        assertEquals(2, results.size)
        assertTrue(results.all { it.pageNumber == 10 })
        assertEquals(2L, results[0].id)
        assertEquals(3L, results[1].id)
    }

    // B. PAGE_RANGE
    @Test
    fun testB_PageRange_RetrievesPagesInRangeInOrder() = runBlocking {
        val c1 = ChunkEntity(id = 1, documentId = 1, pageNumber = 8, text = "Page 8", chunkIndex = 0)
        val c2 = ChunkEntity(id = 2, documentId = 1, pageNumber = 10, text = "Page 10", chunkIndex = 1)
        val c3 = ChunkEntity(id = 3, documentId = 1, pageNumber = 12, text = "Page 12", chunkIndex = 2)
        val c4 = ChunkEntity(id = 4, documentId = 1, pageNumber = 15, text = "Page 15", chunkIndex = 3)
        val c5 = ChunkEntity(id = 5, documentId = 1, pageNumber = 20, text = "Page 20", chunkIndex = 4)

        val store = VectorStore(createMockDao(listOf(c1, c2, c3, c4, c5)))
        val results = store.retrieve(FloatArray(0), listOf(1L), topK = 5, query = "What is on pages 10 to 15?")

        assertEquals(3, results.size)
        assertEquals(listOf(10, 12, 15), results.map { it.pageNumber })
    }

    // C. EXACT_FACT
    @Test
    fun testC_ExactFact_PhoneEmailUrlId_FavorsExactEvidence() = runBlocking {
        val c1 = ChunkEntity(id = 1, documentId = 1, pageNumber = 1, text = "Company contact details and general overview.", chunkIndex = 0)
        val cPhone = ChunkEntity(id = 2, documentId = 1, pageNumber = 2, text = "Customer support phone: +1-800-555-0199 for direct assistance.", chunkIndex = 1)
        val cEmail = ChunkEntity(id = 3, documentId = 1, pageNumber = 2, text = "Official contact email: support@offlinedocai.com for inquiries.", chunkIndex = 2)
        val cUrl = ChunkEntity(id = 4, documentId = 1, pageNumber = 3, text = "Project repository: https://github.com/siddharthabhat/OfflineDocAI for source code.", chunkIndex = 3)
        val cId = ChunkEntity(id = 5, documentId = 1, pageNumber = 4, text = "System operator ID: EMP-94827 with administrative clearance.", chunkIndex = 4)

        val store = VectorStore(createMockDao(listOf(c1, cPhone, cEmail, cUrl, cId)))

        val phoneResults = store.retrieve(FloatArray(0), listOf(1L), topK = 5, query = "What is the phone number?")
        assertTrue(phoneResults.isNotEmpty())
        assertEquals(cPhone.id, phoneResults.first().id)

        val emailResults = store.retrieve(FloatArray(0), listOf(1L), topK = 5, query = "What is the email address?")
        assertTrue(emailResults.isNotEmpty())
        assertEquals(cEmail.id, emailResults.first().id)

        val urlResults = store.retrieve(FloatArray(0), listOf(1L), topK = 5, query = "What is the website URL?")
        assertTrue(urlResults.isNotEmpty())
        assertEquals(cUrl.id, urlResults.first().id)

        val idResults = store.retrieve(FloatArray(0), listOf(1L), topK = 5, query = "What is the employee ID?")
        assertTrue(idResults.isNotEmpty())
        assertEquals(cId.id, idResults.first().id)
    }

    // D. ENTITY
    @Test
    fun testD_Entity_MultiPageRetrieval() = runBlocking {
        val c1 = ChunkEntity(id = 1, documentId = 1, pageNumber = 2, text = "Project Alpha is an autonomous robotics initiative launched in 2023.", chunkIndex = 1)
        val c2 = ChunkEntity(id = 2, documentId = 1, pageNumber = 8, text = "Project Alpha navigation subsystem utilizes LiDAR and SLAM algorithms.", chunkIndex = 5)
        val c3 = ChunkEntity(id = 3, documentId = 1, pageNumber = 14, text = "Project Alpha safety protocols ensure automatic fail-safe shutdown.", chunkIndex = 10)
        val c4 = ChunkEntity(id = 4, documentId = 1, pageNumber = 5, text = "Unrelated database schema and storage indexing specifications.", chunkIndex = 3)

        val store = VectorStore(createMockDao(listOf(c1, c2, c3, c4)))
        val results = store.retrieve(FloatArray(0), listOf(1L), topK = 5, query = "What is Project Alpha?")

        assertTrue(results.any { it.pageNumber == 2 })
        assertTrue(results.any { it.pageNumber == 8 })
        assertTrue(results.any { it.pageNumber == 14 })
    }

    // E. EXHAUSTIVE
    @Test
    fun testE_ExhaustiveList_RetrievesAllItemsAcrossDocument() = runBlocking {
        val cPreamble = ChunkEntity(id = 100, documentId = 1, pageNumber = 1, text = "Legal disclaimer and company preamble.", chunkIndex = 0)
        val cProjA = ChunkEntity(id = 101, documentId = 1, pageNumber = 2, text = "Key Engineering Projects: Project A: Smart Traffic Optimizer using edge AI.", chunkIndex = 1)
        val cProjB = ChunkEntity(id = 102, documentId = 1, pageNumber = 10, text = "Major Projects: Project B: DueGuard - Intelligent offline payment tracker.", chunkIndex = 5)
        val cProjC = ChunkEntity(id = 103, documentId = 1, pageNumber = 25, text = "Technical Projects: Project C: KisanMitra - AI-powered agricultural advisory platform.", chunkIndex = 12)
        val cProjD = ChunkEntity(id = 104, documentId = 1, pageNumber = 40, text = "Core Projects: Project D: OfflineDocAI - Complete On-Device Document Assistant.", chunkIndex = 20)
        val cBenefits = ChunkEntity(id = 105, documentId = 1, pageNumber = 15, text = "General employee benefits and office locations.", chunkIndex = 8)

        val store = VectorStore(createMockDao(listOf(cPreamble, cProjA, cProjB, cProjC, cProjD, cBenefits)))
        // Call with low topK to verify exhaustive scan does NOT truncate at ordinary topK
        val results = store.retrieve(FloatArray(0), listOf(1L), topK = 2, query = "List all projects.")

        assertEquals("Must retrieve all 4 project chunks across the entire document", 4, results.size)
        val retrievedIds = results.map { it.id }
        assertTrue(retrievedIds.contains(101L))
        assertTrue(retrievedIds.contains(102L))
        assertTrue(retrievedIds.contains(103L))
        assertTrue(retrievedIds.contains(104L))
        // Verify document sequential order
        assertEquals(listOf(2, 10, 25, 40), results.map { it.pageNumber })
    }

    // F. FIND_ALL
    @Test
    fun testF_FindAll_RetrievesAllOccurrencesAcrossDocument() = runBlocking {
        val cPreamble = ChunkEntity(id = 1, documentId = 1, pageNumber = 1, text = "Introduction and overview.", chunkIndex = 0)
        val cAuth1 = ChunkEntity(id = 2, documentId = 1, pageNumber = 3, text = "Authentication Overview: System uses tokens.", chunkIndex = 2)
        val cAuth2 = ChunkEntity(id = 3, documentId = 1, pageNumber = 12, text = "Biometric authentication is supported on mobile devices.", chunkIndex = 7)
        val cAuth3 = ChunkEntity(id = 4, documentId = 1, pageNumber = 30, text = "Two-factor authentication (2FA) is mandatory for administrators.", chunkIndex = 15)
        val cAuth4 = ChunkEntity(id = 5, documentId = 1, pageNumber = 55, text = "Authentication logs are archived every 30 days.", chunkIndex = 28)
        val cNetwork = ChunkEntity(id = 6, documentId = 1, pageNumber = 20, text = "Network topology diagram and router configuration.", chunkIndex = 10)

        val store = VectorStore(createMockDao(listOf(cPreamble, cAuth1, cAuth2, cAuth3, cAuth4, cNetwork)))
        val results = store.retrieve(FloatArray(0), listOf(1L), topK = 2, query = "Find every mention of authentication.")

        assertEquals("Must retrieve all 4 occurrences across document", 4, results.size)
        assertEquals(listOf(3, 12, 30, 55), results.map { it.pageNumber })
    }

    // G. COMPARISON
    @Test
    fun testG_Comparison_BalancedDualEntityRepresentation() = runBlocking {
        val cKot1 = ChunkEntity(id = 1, documentId = 1, pageNumber = 1, text = "Kotlin features first-class null safety and coroutines.", chunkIndex = 0)
        val cKot2 = ChunkEntity(id = 2, documentId = 1, pageNumber = 2, text = "Kotlin compiles to JVM bytecode and supports multiplatform.", chunkIndex = 1)
        val cJav1 = ChunkEntity(id = 3, documentId = 1, pageNumber = 3, text = "Java uses checked exceptions and verbose class definitions.", chunkIndex = 2)
        val cJav2 = ChunkEntity(id = 4, documentId = 1, pageNumber = 4, text = "Java memory model manages objects using garbage collection.", chunkIndex = 3)
        val cJoint = ChunkEntity(id = 5, documentId = 1, pageNumber = 5, text = "Comparison between Kotlin and Java shows Kotlin reduces boilerplate by 40%.", chunkIndex = 4)

        val store = VectorStore(createMockDao(listOf(cKot1, cKot2, cJav1, cJav2, cJoint)))
        val results = store.retrieve(FloatArray(0), listOf(1L), topK = 4, query = "Compare Kotlin and Java.")

        assertTrue("Must include joint evidence", results.any { it.id == 5L })
        assertTrue("Must include Kotlin evidence", results.any { it.text.contains("Kotlin") })
        assertTrue("Must include Java evidence", results.any { it.text.contains("Java") })
    }

    // H. PARENT EXPANSION
    @Test
    fun testH_ParentNeighborExpansion() = runBlocking {
        val cPrev = ChunkEntity(id = 14, documentId = 1, pageNumber = 3, text = "Prior to login, client establishes secure TLS session.", chunkIndex = 14, sectionTitle = "Security")
        val cTarget = ChunkEntity(id = 15, documentId = 1, pageNumber = 3, text = "Authentication: User passes cryptographic challenge signed with RSA private key.", chunkIndex = 15, sectionTitle = "Security")
        val cNext = ChunkEntity(id = 16, documentId = 1, pageNumber = 3, text = "After successful verification, server returns short-lived JWT token.", chunkIndex = 16, sectionTitle = "Security")
        val cOtherSection = ChunkEntity(id = 17, documentId = 1, pageNumber = 4, text = "Billing invoices are generated on the 1st of every month.", chunkIndex = 17, sectionTitle = "Billing")

        val store = VectorStore(createMockDao(listOf(cPrev, cTarget, cNext, cOtherSection)))
        val results = store.retrieveDetailed(FloatArray(0), listOf(1L), topK = 2, query = "Explain the cryptographic challenge authentication.")

        // Primary match is chunk 15
        assertTrue(results.any { it.chunk.id == 15L })
        // Neighbor expansion brought chunk 14 and 16
        assertTrue("Neighbor 14 must be expanded", results.any { it.chunk.id == 14L })
        assertTrue("Neighbor 16 must be expanded", results.any { it.chunk.id == 16L })
        // Unrelated section chunk 17 must NOT be expanded
        assertFalse("Unrelated section 17 must NOT be expanded", results.any { it.chunk.id == 17L })
    }

    // I. ORDER METADATA
    @Test
    fun testI_OrderMetadata_RetainsFullMetadata() = runBlocking {
        val c1 = ChunkEntity(
            id = 42,
            documentId = 7,
            pageNumber = 12,
            text = "Architecture design details and message broker topology.",
            chunkIndex = 18,
            pageSpan = "12",
            sectionTitle = "Architecture",
            paragraphIndex = 3
        )

        val store = VectorStore(createMockDao(listOf(c1)))
        val results = store.retrieve(FloatArray(0), listOf(7L), topK = 5, query = "Explain the architecture design.")

        assertTrue(results.isNotEmpty())
        val chunk = results.first()
        assertEquals(12, chunk.pageNumber)
        assertEquals("12", chunk.pageSpan)
        assertEquals(18, chunk.chunkIndex)
        assertEquals("Architecture", chunk.sectionTitle)
        assertEquals(7L, chunk.documentId)
    }

    // J. NEGATIVE
    @Test
    fun testJ_Negative_AbsentFactDoesNotReturnHallucinatedEvidence() = runBlocking {
        val c1 = ChunkEntity(id = 1, documentId = 1, pageNumber = 1, text = "Operating Systems overview: Process scheduling and virtual memory paging.", chunkIndex = 0)
        val c2 = ChunkEntity(id = 2, documentId = 1, pageNumber = 2, text = "Data structures: Binary search trees and hash tables.", chunkIndex = 1)

        val store = VectorStore(createMockDao(listOf(c1, c2)))
        // User asks for a phone number on a document with NO phone numbers
        val results = store.retrieve(FloatArray(0), listOf(1L), topK = 5, query = "What is the phone number?")

        assertTrue("Absent fact query must return empty list to prevent hallucination", results.isEmpty())
    }
}
