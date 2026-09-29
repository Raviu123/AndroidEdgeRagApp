package com.siddu.offlinedocai

import android.content.Context
import com.siddu.offlinedocai.ai.llm.GGUFLLM
import com.siddu.offlinedocai.ai.llm.ParsedSourceBlock
import com.siddu.offlinedocai.ai.retrieval.VectorStore
import com.siddu.offlinedocai.data.dao.ChunkDao
import com.siddu.offlinedocai.data.entities.ChunkEntity
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test
import java.io.File

/**
 * Comprehensive verification test suite for Phase 5: GGUFLLM Grounded Generic Synthesis Redesign.
 *
 * Verifies:
 * A. CLOSED_WORLD (Absent facts return grounded not-found response, zero hallucination)
 * B. EXACT_FACT (Exact values like phone, email, URLs, metrics, IDs preserved verbatim)
 * C. SHORT_FACT (Short factual sentences preserved without arbitrary length thresholds)
 * D. MULTI_SOURCE (Synthesis synthesizes and cites across multiple evidence blocks)
 * E. DOCUMENT_ORDER (Context structure and sequential page order respected)
 * F. EXHAUSTIVE_COMPLETE (All items represented when context is unconstrained)
 * G. EXHAUSTIVE_BUDGET_LIMITED (Disclaimer added, no false completeness claim on budget-limited context)
 * H. FIND_ALL (Distinct occurrences reported across pages)
 * I. COMPARISON (Balanced side-by-side comparison of two entities)
 * J. GLOBAL_SUMMARY (Concise document-level overview with representative evidence citations)
 * K. INVALID_SOURCE_ID (Fabricated source IDs and page numbers stripped by sanitizer)
 * L. RAW_FALLBACK (Concise grounded fallback instead of raw context dump on empty retrieval)
 * M. PROMPT_BUDGET (Safe runtime context size and token budgets verified)
 *
 * Real Document Tests: 7 end-to-end queries on realistic NASA Systems Engineering corpus.
 */
class GenericGroundedSynthesisTest {

    private fun createMockContext(): Context {
        val tempDir = File(System.getProperty("java.io.tmpdir"), "offlinedocai_grounded_test")
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
            override suspend fun getChunksForPage(docIds: List<Long>, pageNumber: Int): List<ChunkEntity> =
                chunkList.filter { it.documentId in docIds && it.pageNumber == pageNumber }
            override suspend fun getChunksForPageRange(docIds: List<Long>, startPage: Int, endPage: Int): List<ChunkEntity> =
                chunkList.filter { it.documentId in docIds && it.pageNumber in startPage..endPage }
        }
    }

    private fun buildPrompt(contextText: String, question: String): String {
        return """
            Context:
            $contextText

            Question: $question
        """.trimIndent()
    }

    // =========================================================================
    // TEST A: CLOSED_WORLD
    // =========================================================================
    @Test
    fun testA_ClosedWorld_AbsentFactReturnsNotFound() = runBlocking {
        val llm = GGUFLLM(createMockContext())
        val context = """
            [Page 1] Solar Photovoltaic Energy Principles: Silicon solar cells convert incident solar photons into direct current via the photoelectric effect with 22% typical efficiency.
            [Page 2] Inverter Systems: Inverters transform DC power into AC power at 240V for grid synchronization.
        """.trimIndent()

        val prompt = buildPrompt(context, "Who is the CEO of SpaceX?")
        val answer = llm.generate(prompt)

        println("=== TEST A: CLOSED_WORLD ===")
        println("ANSWER:\n$answer\n")

        assertTrue("Must return grounded not-found response",
            answer.contains("could not find information") || answer.contains("I searched the provided document(s)"))
        assertFalse("Must NOT hallucinate Elon Musk", answer.contains("Elon Musk"))
    }

    // =========================================================================
    // TEST B: EXACT_FACT
    // =========================================================================
    @Test
    fun testB_ExactFact_PreservedVerbatim() = runBlocking {
        val llm = GGUFLLM(createMockContext())
        val context = """
            [SOURCE_ID=101] [PAGE=1] [SECTION=Contact] Technical Support Helpline: +1-800-555-0199. Support Email: support@offlinedocai.com. Code Repository: https://github.com/example/engine.
            [SOURCE_ID=102] [PAGE=5] [SECTION=Battery] Device Power: The hardware is equipped with a 5000 mAh lithium-polymer battery operating at 3.7V nominal voltage.
        """.trimIndent()

        // Test Phone
        val phoneAns = llm.generate(buildPrompt(context, "What is the phone number?"))
        assertTrue("Phone must be verbatim", phoneAns.contains("+1-800-555-0199"))
        assertTrue("Phone must cite Page 1", phoneAns.contains("(Page 1)"))

        // Test Email
        val emailAns = llm.generate(buildPrompt(context, "What is the support email?"))
        assertTrue("Email must be verbatim", emailAns.contains("support@offlinedocai.com"))
        assertTrue("Email must cite Page 1", emailAns.contains("(Page 1)"))

        // Test URL
        val urlAns = llm.generate(buildPrompt(context, "What is the repository URL?"))
        assertTrue("URL must be verbatim", urlAns.contains("https://github.com/example/engine"))
        assertTrue("URL must cite Page 1", urlAns.contains("(Page 1)"))

        // Test Metric
        val metricAns = llm.generate(buildPrompt(context, "What is the battery capacity?"))
        assertTrue("Battery metric must be verbatim", metricAns.contains("5000 mAh"))
        assertTrue("Battery metric must cite Page 5", metricAns.contains("(Page 5)"))
    }

    // =========================================================================
    // TEST C: SHORT_FACT
    // =========================================================================
    @Test
    fun testC_ShortFact_NotDiscardedByLengthThreshold() = runBlocking {
        val llm = GGUFLLM(createMockContext())
        val context = """
            [Page 2] Meeting Location. Room: 402B.
            [Page 3] System status. Release: Beta.
        """.trimIndent()

        val prompt = buildPrompt(context, "What room is the meeting in?")
        val answer = llm.generate(prompt)

        println("=== TEST C: SHORT_FACT ===")
        println("ANSWER:\n$answer\n")

        assertTrue("Short fact 402B must be preserved", answer.contains("402B"))
        assertTrue("Must cite Page 2", answer.contains("(Page 2)"))
    }

    // =========================================================================
    // TEST D: MULTI_SOURCE
    // =========================================================================
    @Test
    fun testD_MultiSource_SynthesisCombinesMultipleEvidenceBlocks() = runBlocking {
        val llm = GGUFLLM(createMockContext())
        val context = """
            [SOURCE_ID=201] [PAGE=3] [SECTION=Safety] Battery safety operational limit: Battery temperature must not exceed 45C during continuous discharge.
            [SOURCE_ID=202] [PAGE=5] [SECTION=Power] Battery fast charging specs: Fast charging delivers 45W and reaches 80% charge in 30 minutes.
            [SOURCE_ID=203] [PAGE=9] [SECTION=Maintenance] Battery lifecycle maintenance: Replace battery cell every 2 years or after 800 full charge cycles.
        """.trimIndent()

        val prompt = buildPrompt(context, "Where in the document is the battery discussed?")
        val answer = llm.generate(prompt)

        println("=== TEST D: MULTI_SOURCE ===")
        println("ANSWER:\n$answer\n")

        assertTrue("Must cite Page 3", answer.contains("Page 3"))
        assertTrue("Must cite Page 5", answer.contains("Page 5"))
        assertTrue("Must cite Page 9", answer.contains("Page 9"))
    }

    // =========================================================================
    // TEST E: DOCUMENT_ORDER
    // =========================================================================
    @Test
    fun testE_DocumentOrder_PreservesSequentialPageOrder() = runBlocking {
        val llm = GGUFLLM(createMockContext())
        val context = """
            [Page 1] Initialization Step: Power on the device and verify status LED is green.
            [Page 2] Configuration Step: Connect device to local network via USB ethernet interface.
            [Page 3] Verification Step: Run automated diagnostic script and confirm zero packet drop.
        """.trimIndent()

        val prompt = buildPrompt(context, "Summarize pages 1 to 3.")
        val answer = llm.generate(prompt)

        println("=== TEST E: DOCUMENT_ORDER ===")
        println("ANSWER:\n$answer\n")

        val p1Idx = answer.indexOf("Page 1")
        val p2Idx = answer.indexOf("Page 2")
        val p3Idx = answer.indexOf("Page 3")

        assertTrue("Page 1 must appear", p1Idx != -1)
        assertTrue("Page 2 must appear", p2Idx != -1)
        assertTrue("Page 3 must appear", p3Idx != -1)
        assertTrue("Page 1 must appear before Page 2", p1Idx < p2Idx)
        assertTrue("Page 2 must appear before Page 3", p2Idx < p3Idx)
    }

    // =========================================================================
    // TEST F: EXHAUSTIVE_COMPLETE
    // =========================================================================
    @Test
    fun testF_ExhaustiveComplete_AllItemsRepresentedWithoutDisclaimer() = runBlocking {
        val llm = GGUFLLM(createMockContext())
        val context = """
            [Page 4] Subsystem Components: 1. Optical Sensor, 2. Infrared Camera, 3. LiDAR Scanner, and 4. Telemetry Transmitter.
        """.trimIndent()

        val prompt = buildPrompt(context, "List all components.")
        val answer = llm.generate(prompt)

        println("=== TEST F: EXHAUSTIVE_COMPLETE ===")
        println("ANSWER:\n$answer\n")

        assertTrue("Optical Sensor must be present", answer.contains("Optical Sensor"))
        assertTrue("Infrared Camera must be present", answer.contains("Infrared Camera"))
        assertTrue("LiDAR Scanner must be present", answer.contains("LiDAR Scanner"))
        assertTrue("Telemetry Transmitter must be present", answer.contains("Telemetry Transmitter"))
        assertFalse("Must NOT contain budget limited disclaimer", answer.contains("budget-limited"))
    }

    // =========================================================================
    // TEST G: EXHAUSTIVE_BUDGET_LIMITED
    // =========================================================================
    @Test
    fun testG_ExhaustiveBudgetLimited_IncludesDisclaimer() = runBlocking {
        val llm = GGUFLLM(createMockContext())
        val context = """
            [Page 4] Subsystem Components: 1. Optical Sensor, 2. Infrared Camera, 3. LiDAR Scanner.
            isBudgetLimited=true
        """.trimIndent()

        val prompt = buildPrompt(context, "List all components.")
        val answer = llm.generate(prompt)

        println("=== TEST G: EXHAUSTIVE_BUDGET_LIMITED ===")
        println("ANSWER:\n$answer\n")

        assertTrue("Must contain Optical Sensor", answer.contains("Optical Sensor"))
        assertTrue("Must include budget-limited notice", answer.contains("budget-limited") || answer.contains("budget"))
        assertFalse("Must not claim absolute completeness", answer.contains("complete list of all"))
    }

    // =========================================================================
    // TEST H: FIND_ALL
    // =========================================================================
    @Test
    fun testH_FindAll_ReportsOccurrencesAcrossPages() = runBlocking {
        val llm = GGUFLLM(createMockContext())
        val context = """
            [Page 2] Data at rest uses AES-256 encryption.
            [Page 6] Transport layer uses TLS 1.3 encryption with ECDHE key exchange.
            [Page 11] Backup archives require hardware token encryption keys.
        """.trimIndent()

        val prompt = buildPrompt(context, "Find every mention of encryption.")
        val answer = llm.generate(prompt)

        println("=== TEST H: FIND_ALL ===")
        println("ANSWER:\n$answer\n")

        assertTrue("Must cite Page 2", answer.contains("Page 2"))
        assertTrue("Must cite Page 6", answer.contains("Page 6"))
        assertTrue("Must cite Page 11", answer.contains("Page 11"))
        assertTrue("Must mention AES-256", answer.contains("AES-256"))
        assertTrue("Must mention TLS 1.3", answer.contains("TLS 1.3"))
    }

    // =========================================================================
    // TEST I: COMPARISON
    // =========================================================================
    @Test
    fun testI_Comparison_BalancedDualEntityRepresentation() = runBlocking {
        val llm = GGUFLLM(createMockContext())
        val context = """
            [Page 8] Lithium-ion chemistry provides 250 Wh/kg energy density with flammable liquid electrolyte requiring active thermal management.
            [Page 12] Solid-state chemistry replaces liquid electrolyte with non-flammable ceramic separators, enabling 400 Wh/kg density and wider operating temperatures.
        """.trimIndent()

        val prompt = buildPrompt(context, "Compare Lithium-ion and Solid-state batteries.")
        val answer = llm.generate(prompt)

        println("=== TEST I: COMPARISON ===")
        println("ANSWER:\n$answer\n")

        assertTrue("Must mention Lithium-ion", answer.contains("Lithium-ion") || answer.contains("lithium-ion"))
        assertTrue("Must mention Solid-state", answer.contains("Solid-state") || answer.contains("solid-state"))
        assertTrue("Must cite Page 8", answer.contains("Page 8"))
        assertTrue("Must cite Page 12", answer.contains("Page 12"))
    }

    // =========================================================================
    // TEST J: GLOBAL_SUMMARY
    // =========================================================================
    @Test
    fun testJ_GlobalSummary_ConciseOverviewFromEvidence() = runBlocking {
        val llm = GGUFLLM(createMockContext())
        val context = """
            [Page 1] System Architecture Specification. This document outlines the on-device document processing engine.
            [Page 3] Document parser handles PDF, DOCX, and TXT inputs while preserving layout structure.
            [Page 7] Vector retrieval combines dense ONNX embeddings with sparse BM25 indexing.
        """.trimIndent()

        val prompt = buildPrompt(context, "What is this document about?")
        val answer = llm.generate(prompt)

        println("=== TEST J: GLOBAL_SUMMARY ===")
        println("ANSWER:\n$answer\n")

        assertTrue("Must contain overview header", answer.contains("Document Overview") || answer.contains("Summary"))
        assertTrue("Must cite Page 1 or 3", answer.contains("(Page 1)") || answer.contains("(Page 3)"))
    }

    // =========================================================================
    // TEST K: INVALID_SOURCE_ID
    // =========================================================================
    @Test
    fun testK_InvalidSourceId_SanitizerRejectsFabricatedCitations() {
        val llm = GGUFLLM(createMockContext())
        val validBlocks = listOf(
            ParsedSourceBlock(sourceId = 101L, page = 3, pageSpan = "3", section = "Sec1", chunkIndex = 0, text = "Valid chunk text")
        )

        val fabricatedText = "The result is 42. [Source 999, Page 88] (Page 88)"
        val sanitized = llm.sanitizeCitations(fabricatedText, validBlocks)

        println("=== TEST K: INVALID_SOURCE_ID ===")
        println("ORIGINAL: $fabricatedText")
        println("SANITIZED: $sanitized\n")

        assertFalse("Must strip fabricated Source 999", sanitized.contains("999"))
        assertFalse("Must strip fabricated Page 88", sanitized.contains("88"))
        assertTrue("Must keep original factual answer", sanitized.contains("The result is 42."))
    }

    // =========================================================================
    // TEST L: RAW_FALLBACK
    // =========================================================================
    @Test
    fun testL_RawFallback_ConciseGroundedResponseOnEmptyEvidence() = runBlocking {
        val llm = GGUFLLM(createMockContext())
        val emptyPrompt = buildPrompt("[NO_EVIDENCE_FOUND]", "What is the cooling system?")
        val answer = llm.generate(emptyPrompt)

        println("=== TEST L: RAW_FALLBACK ===")
        println("ANSWER:\n$answer\n")

        assertTrue("Must return concise not-found response",
            answer.contains("could not find information") || answer.contains("I searched the provided document(s)"))
        assertFalse("Must not dump [NO_EVIDENCE_FOUND] verbatim", answer.contains("[NO_EVIDENCE_FOUND]"))
    }

    // =========================================================================
    // TEST M: PROMPT_BUDGET
    // =========================================================================
    @Test
    fun testM_PromptBudget_SafeExecutionParameters() {
        assertEquals("Safe context size must be 2048", 2048, GGUFLLM.SAFE_CONTEXT_SIZE)
        assertEquals("Max generation tokens must be 384", 384, GGUFLLM.MAX_GENERATION_TOKENS)
        assertEquals("Default temperature must be 0.1f", 0.1f, GGUFLLM.DEFAULT_TEMPERATURE, 0.001f)
    }

    // =========================================================================
    // REAL DOCUMENT TESTS: 7 QUERIES ON REALISTIC NASA / ENGINEERING CORPUS
    // =========================================================================

    private val seHandbookChunks = listOf(
        ChunkEntity(id = 601, documentId = 10, pageNumber = 3,
            text = "The Systems Engineering (SE) engine is a cyclical, iterative process consisting of 17 processes categorized into System Design Processes, Product Realization Processes, and Technical Management Processes.",
            embedding = null),
        ChunkEntity(id = 602, documentId = 10, pageNumber = 10,
            text = "Quantum Encryption Protocols: Page 10 outlines post-quantum lattice-based cryptography, detailing Kyber key encapsulation and Dilithium digital signatures for next-generation aerospace telemetry.",
            embedding = null),
        ChunkEntity(id = 603, documentId = 10, pageNumber = 15,
            text = "Chapter 2: Fundamentals of Systems Engineering. Systems engineering provides the technical foundation for the successful realization of complex aerospace systems.",
            embedding = null),
        ChunkEntity(id = 604, documentId = 10, pageNumber = 16,
            text = "Systems engineering emphasizes the relationship between project management and systems engineering, iterative decision-making, stakeholder satisfaction, and rigorous trade studies to manage system complexity.",
            embedding = null),
        ChunkEntity(id = 605, documentId = 10, pageNumber = 18,
            text = "The systems engineering life cycle spans seven distinct project phases, progressing from concept studies (Pre-Phase A) through concept development (Phase A), preliminary design (Phase B), final design (Phase C), fabrication and integration (Phase D), operations (Phase E), to closeout (Phase F).",
            embedding = null),
        ChunkEntity(id = 606, documentId = 10, pageNumber = 21,
            text = "Phase A (Concept and Technology Development) determines the feasible system concept, defines mission architecture, establishes top-level system requirements, and develops initial risk management plans.",
            embedding = null),
        ChunkEntity(id = 607, documentId = 10, pageNumber = 25,
            text = "Phase B (Preliminary Design and Technology Completion) establishes the preliminary design, freezes the baseline requirements, demonstrates critical technology maturity at TRL 6, and culminates in the Preliminary Design Review (PDR).",
            embedding = null),
        ChunkEntity(id = 608, documentId = 10, pageNumber = 88,
            text = "Section 4.3: System Verification Process provides evidence that the delivered system complies with its specified requirements through test, analysis, inspection, or demonstration.",
            embedding = null),
        ChunkEntity(id = 609, documentId = 10, pageNumber = 104,
            text = "Section 5.4: System Validation Process demonstrates that the realized system functions as intended in its operational environment to fulfill stakeholder expectations and mission objectives.",
            embedding = null),
        ChunkEntity(id = 610, documentId = 10, pageNumber = 120,
            text = "Technical Management Processes include: 1. Technical Planning, 2. Requirements Management, 3. Interface Management, 4. Technical Risk Management, 5. Configuration Management, 6. Technical Data Management, 7. Technical Assessment, and 8. Decision Analysis.",
            embedding = null)
    )

    private fun runPipelineQuery(query: String): Triple<String, String, String> = runBlocking {
        val dao = createMockChunkDao(seHandbookChunks)
        val vectorStore = VectorStore(dao)
        val retrieved = vectorStore.retrieve(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(10L),
            topK = 5,
            query = query
        )

        val contextText = if (retrieved.isNotEmpty()) {
            retrieved.joinToString("\n\n") { "[Page ${it.pageNumber}] ${it.text}" }
        } else {
            "[NO_EVIDENCE_FOUND]"
        }

        val prompt = buildPrompt(contextText, query)
        val llm = GGUFLLM(createMockContext())
        val answer = llm.generate(prompt)

        println("==================================================")
        println("USER QUERY: $query")
        println("RETRIEVED BLOCKS:\n$contextText")
        println("GENERATED ANSWER:\n$answer")
        println("==================================================\n")

        Triple(query, contextText, answer)
    }

    @Test
    fun testRealDoc1_Definition_SE_Engine() {
        val (_, _, answer) = runPipelineQuery("What is the Systems Engineering engine?")
        assertTrue("Must contain cyclical iterative definition", answer.contains("cyclical") || answer.contains("iterative"))
        assertTrue("Must mention 17 processes", answer.contains("17 processes"))
        assertTrue("Must cite Page 3", answer.contains("(Page 3)"))
    }

    @Test
    fun testRealDoc2_List_TechnicalManagementProcesses() {
        val (_, _, answer) = runPipelineQuery("What are the technical management processes?")
        assertTrue("Must contain Technical Planning", answer.contains("Technical Planning"))
        assertTrue("Must contain Configuration Management", answer.contains("Configuration Management"))
        assertTrue("Must contain Decision Analysis", answer.contains("Decision Analysis"))
        assertTrue("Must cite Page 120", answer.contains("(Page 120)"))
    }

    @Test
    fun testRealDoc3_Comparison_PhaseA_vs_PhaseB() {
        val (_, _, answer) = runPipelineQuery("Compare Phase A and Phase B.")
        assertTrue("Must mention Phase A", answer.contains("Phase A"))
        assertTrue("Must mention Phase B", answer.contains("Phase B"))
        assertTrue("Must cite Page 21", answer.contains("(Page 21)"))
        assertTrue("Must cite Page 25", answer.contains("(Page 25)"))
    }

    @Test
    fun testRealDoc4_PageLookup_Page10() {
        val (_, _, answer) = runPipelineQuery("What is on page 10?")
        assertTrue("Must contain Page 10 header", answer.contains("Page 10 Content"))
        assertTrue("Must contain Quantum Encryption Protocols", answer.contains("Quantum Encryption Protocols"))
        assertTrue("Must cite Page 10", answer.contains("(Page 10)"))
    }

    @Test
    fun testRealDoc5_PageRange_Pages15To16() {
        val (_, _, answer) = runPipelineQuery("Summarize pages 15 to 16.")
        assertTrue("Must cite Page 15", answer.contains("Page 15"))
        assertTrue("Must cite Page 16", answer.contains("Page 16"))
        assertTrue("Must contain systems engineering content", answer.contains("Systems Engineering") || answer.contains("systems engineering"))
    }

    @Test
    fun testRealDoc6_MultiSource_VerificationAndValidation() {
        val (_, _, answer) = runPipelineQuery("Where does the book discuss verification and validation?")
        assertTrue("Must cite Page 88", answer.contains("Page 88"))
        assertTrue("Must cite Page 104", answer.contains("Page 104"))
        assertTrue("Must mention verification", answer.contains("Verification") || answer.contains("verification"))
        assertTrue("Must mention validation", answer.contains("Validation") || answer.contains("validation"))
    }

    @Test
    fun testRealDoc7_AbsentFact_SpaceX_CEO() {
        val (_, _, answer) = runPipelineQuery("Who is the CEO of SpaceX?")
        assertTrue("Must return grounded not-found response",
            answer.contains("could not find information") || answer.contains("I searched the provided document(s)"))
        assertFalse("Must NOT hallucinate Elon Musk", answer.contains("Elon Musk"))
    }
}
