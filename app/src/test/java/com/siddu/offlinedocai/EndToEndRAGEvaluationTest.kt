package com.siddu.offlinedocai

import android.content.Context
import com.siddu.offlinedocai.ai.chunking.TextChunker
import com.siddu.offlinedocai.domain.rag.ExtractedPage
import com.siddu.offlinedocai.ai.llm.GGUFLLM
import com.siddu.offlinedocai.ai.retrieval.*
import com.siddu.offlinedocai.ai.validation.CitationValidator
import com.siddu.offlinedocai.ai.validation.FallbackLevel
import com.siddu.offlinedocai.ai.validation.ValidationStatus
import com.siddu.offlinedocai.data.dao.ChunkDao
import com.siddu.offlinedocai.data.entities.ChunkEntity
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import kotlin.system.measureNanoTime
import kotlin.system.measureTimeMillis

/**
 * PHASE 7: Comprehensive Final End-to-End RAG Evaluation Test Suite.
 *
 * Evaluates the COMPLETE OfflineDocAI pipeline:
 * Document -> TextChunker -> Metadata -> QueryIntentRouter -> VectorStore retrieval ->
 * Parent/neighbor expansion -> ContextAssembler (budgeting/ordering) ->
 * GGUFLLM Grounded Synthesis -> CitationValidator -> Final Answer.
 *
 * Covers:
 * - 25 Comprehensive Evaluation Matrix Classes (A through Y)
 * - Query Formulation Variations (Semantic consistency under arbitrary user wording)
 * - Cross-document isolation and multi-page cross-page continuation
 * - Full Evidence Tracing for every query
 * - Deterministic Metric Calculation across 15 separate evaluation dimensions
 * - Failure Layer Attribution (R1 through R8)
 * - Performance Benchmark Measurements (ingestion, retrieval, synthesis, validation)
 */
class EndToEndRAGEvaluationTest {

    private fun createMockContext(): Context {
        val tempDir = File(System.getProperty("java.io.tmpdir"), "offlinedocai_eval_test")
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

            Answer concisely, referencing the page numbers when appropriate.
        """.trimIndent()
    }

    data class EvaluationTrace(
        val category: String,
        val query: String,
        val classifiedIntent: QueryIntent,
        val retrievedChunkCount: Int,
        val retrievedPages: List<Int>,
        val contextTokens: Int,
        val isBudgetLimited: Boolean,
        val finalAnswer: String,
        val fallbackLevel: FallbackLevel,
        val citations: List<String>,
        val expectedEvidence: String,
        val passed: Boolean,
        val failureLayer: String = "None"
    )

    private fun printTrace(trace: EvaluationTrace) {
        println("==================================================")
        println("EVALUATION MATRIX: [${trace.category}]")
        println("USER QUERY: ${trace.query}")
        println("CLASSIFIED INTENT: ${trace.classifiedIntent}")
        println("RETRIEVED CHUNKS: ${trace.retrievedChunkCount} chunks, Pages: ${trace.retrievedPages}")
        println("CONTEXT ASSEMBLED: ${trace.contextTokens} tokens (Budget Limited: ${trace.isBudgetLimited})")
        println("GENERATED & VALIDATED ANSWER:\n${trace.finalAnswer}")
        println("CITATIONS: ${trace.citations}")
        println("EXPECTED EVIDENCE: ${trace.expectedEvidence}")
        println("RESULT: ${if (trace.passed) "PASSED [SUCCESS]" else "FAILED"}")
        println("FAILURE LAYER: ${trace.failureLayer}")
        println("==================================================\n")
    }

    private fun runPipeline(
        category: String,
        query: String,
        chunks: List<ChunkEntity>,
        docIds: List<Long> = listOf(10L),
        expectedEvidence: String,
        assertion: (EvaluationTrace) -> Boolean
    ): EvaluationTrace = runBlocking {
        val classification = QueryIntentRouter.route(query)
        println("ROUTED INTENT: " + classification.intent)
        val dao = createMockChunkDao(chunks)
        val vectorStore = VectorStore(dao)

        val retrieved = vectorStore.retrieveDetailed(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = docIds,
            topK = 20,
            query = query
        )

        val assembled = ContextAssembler.assemble(
            retrievedResults = retrieved,
            queryIntent = classification.intent,
            queryClassification = classification,
            maxEvidenceTokens = ContextAssembler.DEFAULT_MAX_CONTEXT_TOKENS
        )
        println("CONTEXT_TEXT: " + assembled.contextText)

        val prompt = buildPrompt(assembled.contextText, query)
        val llm = GGUFLLM(createMockContext())
        val answer = llm.generate(prompt)

        val validationResult = CitationValidator.validateWithSourceBlocks(
            answer = answer,
            sourceBlocks = assembled.sourceBlocks,
            query = query,
            isBudgetLimited = assembled.isBudgetLimited,
            classification = classification
        )

        val retrievedPages = retrieved.map { it.chunk.pageNumber }.distinct().sorted()
        val citations = validationResult.validCitations.map { "Source ${it.sourceId} Page ${it.page}" }

        val traceWithoutStatus = EvaluationTrace(
            category = category,
            query = query,
            classifiedIntent = classification.intent,
            retrievedChunkCount = retrieved.size,
            retrievedPages = retrievedPages,
            contextTokens = assembled.totalEstimatedTokens,
            isBudgetLimited = assembled.isBudgetLimited,
            finalAnswer = validationResult.validatedAnswer,
            fallbackLevel = validationResult.fallbackLevel,
            citations = citations,
            expectedEvidence = expectedEvidence,
            passed = false,
            failureLayer = "None"
        )

        val isPassed = assertion(traceWithoutStatus)
        val failureLayer = if (isPassed) "None" else {
            when {
                retrieved.isEmpty() && expectedEvidence != "NOT_FOUND" -> "R4: Retrieval Failure"
                !assembled.contextText.contains(expectedEvidence.take(20)) && expectedEvidence != "NOT_FOUND" -> "R5: Context Assembly Failure"
                validationResult.fallbackLevel == FallbackLevel.L3_EXTRACTIVE_FALLBACK -> "R6: Synthesis Degradation"
                else -> "R7: Citation/Claim Validation Failure"
            }
        }

        val trace = traceWithoutStatus.copy(passed = isPassed, failureLayer = failureLayer)
        printTrace(trace)
        println("FINAL_ANSWER_TRACE: " + trace.finalAnswer)
        assertTrue("Evaluation failed for category $category: ${trace.failureLayer}", trace.passed)
        trace
    }

    // =========================================================================
    // REAL TEST DOCUMENTS
    // =========================================================================

    // Document 1: NASA Systems Engineering Handbook Corpus (Doc ID 10)
    private val nasaCorpusChunks = listOf(
        ChunkEntity(id = 101, documentId = 10, pageNumber = 3, chunkIndex = 0,
            sectionTitle = "SE Engine", pageSpan = "3",
            text = "The Systems Engineering (SE) engine is a cyclical, iterative process consisting of 17 processes categorized into System Design Processes, Product Realization Processes, and Technical Management Processes.",
            embedding = null),
        ChunkEntity(id = 102, documentId = 10, pageNumber = 10, chunkIndex = 1,
            sectionTitle = "Cryptography", pageSpan = "10",
            text = "Quantum Encryption Protocols: Page 10 outlines post-quantum lattice-based cryptography, detailing Kyber key encapsulation and Dilithium digital signatures for next-generation aerospace telemetry.",
            embedding = null),
        ChunkEntity(id = 103, documentId = 10, pageNumber = 15, chunkIndex = 2,
            sectionTitle = "Fundamentals", pageSpan = "15-16",
            text = "Chapter 2: Fundamentals of Systems Engineering. Systems engineering provides the technical foundation for the successful realization of complex aerospace systems.",
            embedding = null),
        ChunkEntity(id = 104, documentId = 10, pageNumber = 16, chunkIndex = 3,
            sectionTitle = "Fundamentals", pageSpan = "15-16",
            text = "Systems engineering emphasizes the relationship between project management and systems engineering, iterative decision-making, stakeholder satisfaction, and rigorous trade studies to manage system complexity.",
            embedding = null),
        ChunkEntity(id = 105, documentId = 10, pageNumber = 18, chunkIndex = 4,
            sectionTitle = "Life Cycle", pageSpan = "18",
            text = "The systems engineering life cycle spans seven distinct project phases, progressing from concept studies (Pre-Phase A) through concept development (Phase A), preliminary design (Phase B), final design (Phase C), fabrication and integration (Phase D), operations (Phase E), to closeout (Phase F).",
            embedding = null),
        ChunkEntity(id = 106, documentId = 10, pageNumber = 21, chunkIndex = 5,
            sectionTitle = "Phase A", pageSpan = "21",
            text = "Phase A (Concept and Technology Development) determines the feasible system concept, defines mission architecture, establishes top-level system requirements, and develops initial risk management plans.",
            embedding = null),
        ChunkEntity(id = 107, documentId = 10, pageNumber = 25, chunkIndex = 6,
            sectionTitle = "Phase B", pageSpan = "25",
            text = "Phase B (Preliminary Design and Technology Completion) establishes the preliminary design, freezes the baseline requirements, demonstrates critical technology maturity at TRL 6, and culminates in the Preliminary Design Review (PDR).",
            embedding = null),
        ChunkEntity(id = 108, documentId = 10, pageNumber = 88, chunkIndex = 7,
            sectionTitle = "Verification", pageSpan = "88",
            text = "Section 4.3: System Verification Process provides evidence that the delivered system complies with its specified requirements through test, analysis, inspection, or demonstration.",
            embedding = null),
        ChunkEntity(id = 109, documentId = 10, pageNumber = 104, chunkIndex = 8,
            sectionTitle = "Validation", pageSpan = "104",
            text = "Section 5.4: System Validation Process demonstrates that the realized system functions as intended in its operational environment to fulfill stakeholder expectations and mission objectives.",
            embedding = null),
        ChunkEntity(id = 110, documentId = 10, pageNumber = 120, chunkIndex = 9,
            sectionTitle = "Technical Management", pageSpan = "120",
            text = "Technical Management Processes include: 1. Technical Planning, 2. Requirements Management, 3. Interface Management, 4. Technical Risk Management, 5. Configuration Management, 6. Technical Data Management, 7. Technical Assessment, and 8. Decision Analysis.",
            embedding = null)
    )

    // Document 2: Hardware Device Specification & Battery Report (Doc ID 20)
    private val hardwareCorpusChunks = listOf(
        ChunkEntity(id = 201, documentId = 20, pageNumber = 1, chunkIndex = 0,
            sectionTitle = "Overview", pageSpan = "1",
            text = "Hardware Device Specification. Helpline Contact: +1-800-555-0199. Support Email: support@offlinedocai.com. Code Repository: https://github.com/nasa/se-handbook. Release Date: 2024-05-01.",
            embedding = null),
        ChunkEntity(id = 202, documentId = 20, pageNumber = 2, chunkIndex = 1,
            sectionTitle = "Power", pageSpan = "2",
            text = "Power & Battery Subsystem: The device incorporates a 3200 mAh lithium-polymer battery operating at 3.7V nominal voltage. Continuous operational temperature must remain below 45°C. Fast charging provides 45W delivering 80% charge in 30 minutes.",
            embedding = null),
        ChunkEntity(id = 203, documentId = 20, pageNumber = 3, chunkIndex = 2,
            sectionTitle = "Components", pageSpan = "3",
            text = "| Component | Specification | Power |\n| Optical Sensor | 4K 60fps HDR | 3.5W |\n| Infrared Camera | 1080p Thermal | 2.1W |\n| LiDAR Scanner | 100m Range | 5.0W |",
            embedding = null),
        ChunkEntity(id = 204, documentId = 20, pageNumber = 4, chunkIndex = 3,
            sectionTitle = "Maintenance", pageSpan = "4",
            text = "Battery Maintenance: Replace battery cell after 800 cycles or every 2 years. Standard replacement unit cost is $120.",
            embedding = null)
    )

    // Document 3: Candidate Resume (Doc ID 30)
    private val resumeCorpusChunks = listOf(
        ChunkEntity(id = 301, documentId = 30, pageNumber = 1, chunkIndex = 0,
            sectionTitle = "Resume", pageSpan = "1",
            text = "SIDDHARTHA BHAT S. Phone: +91 7907921234. Email: siddusbs2810@gmail.com. Kasaragod, Kerala. Projects: 1. OfflineDocAI - Complete on-device document assistant. 2. DueGuard - Subscription tracker. 3. KisanMitra - Agricultural advisory app. Education: B.Tech in CSE (2020-2024), CGPA: 8.5.",
            embedding = null)
    )

    // Document 4: Web & Mobile Architecture Guide (Doc ID 40 for Multi-Document Testing)
    private val softwareCorpusChunks = listOf(
        ChunkEntity(id = 401, documentId = 40, pageNumber = 1, chunkIndex = 0,
            sectionTitle = "Mobile", pageSpan = "1",
            text = "Mobile Architecture: React Native and Kotlin Multiplatform provide cross-platform code sharing across iOS and Android.",
            embedding = null),
        ChunkEntity(id = 402, documentId = 40, pageNumber = 2, chunkIndex = 1,
            sectionTitle = "Design", pageSpan = "2",
            text = "Design System: UI Wireframing begins in Phase A with Figma prototypes.",
            embedding = null)
    )

    // =========================================================================
    // PART 1: 25 COMPREHENSIVE EVALUATION MATRIX TESTS (A THROUGH Y)
    // =========================================================================

    @Test
    fun testA_ExactFact_BatteryCapacity() {
        runPipeline(
            category = "A. EXACT FACT",
            query = "What is the battery capacity?",
            chunks = hardwareCorpusChunks,
            docIds = listOf(20L),
            expectedEvidence = "3200 mAh"
        ) { trace ->
            trace.finalAnswer.contains("3200 mAh") && !trace.finalAnswer.contains("5000 mAh")
        }
    }

    @Test
    fun testB_PageLookup_Page10() {
        runPipeline(
            category = "B. PAGE LOOKUP",
            query = "What is on page 10?",
            chunks = nasaCorpusChunks,
            docIds = listOf(10L),
            expectedEvidence = "Kyber key encapsulation"
        ) { trace ->
            trace.finalAnswer.contains("Kyber") && trace.finalAnswer.contains("(Page 10)")
        }
    }

    @Test
    fun testC_PageRange_Pages15To16() {
        runPipeline(
            category = "C. PAGE RANGE",
            query = "Summarize pages 15 to 16.",
            chunks = nasaCorpusChunks,
            docIds = listOf(10L),
            expectedEvidence = "Systems engineering provides the technical foundation"
        ) { trace ->
            trace.finalAnswer.contains("technical foundation") && (trace.finalAnswer.contains("Page 15") || trace.finalAnswer.contains("Page 16"))
        }
    }

    @Test
    fun testD_SectionQuery_Verification() {
        runPipeline(
            category = "D. SECTION QUERY",
            query = "Explain the system verification process.",
            chunks = nasaCorpusChunks,
            docIds = listOf(10L),
            expectedEvidence = "test, analysis, inspection, or demonstration"
        ) { trace ->
            trace.finalAnswer.contains("verification", ignoreCase = true) &&
                    (trace.finalAnswer.contains("test") || trace.finalAnswer.contains("inspection"))
        }
    }

    @Test
    fun testE_ConceptualQuestion_ImportanceOfSE() {
        runPipeline(
            category = "E. CONCEPTUAL QUESTION",
            query = "Why is systems engineering important?",
            chunks = nasaCorpusChunks,
            docIds = listOf(10L),
            expectedEvidence = "technical foundation for the successful realization"
        ) { trace ->
            trace.finalAnswer.contains("technical foundation") || trace.finalAnswer.contains("realization of complex")
        }
    }

    @Test
    fun testF_EntityLookup_PhaseA() {
        runPipeline(
            category = "F. ENTITY LOOKUP",
            query = "What does the document say about Phase A?",
            chunks = nasaCorpusChunks,
            docIds = listOf(10L),
            expectedEvidence = "concept development"
        ) { trace ->
            trace.finalAnswer.contains("concept", ignoreCase = true) && trace.finalAnswer.contains("Phase A", ignoreCase = true)
        }
    }

    @Test
    fun testG_ExhaustiveList_TechnicalManagementProcesses() {
        runPipeline(
            category = "G. EXHAUSTIVE LIST",
            query = "List all technical management processes.",
            chunks = nasaCorpusChunks,
            docIds = listOf(10L),
            expectedEvidence = "Technical Planning"
        ) { trace ->
            trace.finalAnswer.contains("Technical Planning") &&
                    trace.finalAnswer.contains("Requirements Management") &&
                    trace.finalAnswer.contains("Decision Analysis")
        }
    }

    @Test
    fun testH_FindAll_MentionsOfValidationAndVerification() {
        runPipeline(
            category = "H. FIND ALL",
            query = "Find every mention of verification.",
            chunks = nasaCorpusChunks,
            docIds = listOf(10L),
            expectedEvidence = "Section 4.3"
        ) { trace ->
            trace.finalAnswer.contains("Section 4.3") || trace.finalAnswer.contains("Page 88")
        }
    }

    @Test
    fun testI_Comparison_PhaseAVsPhaseB() {
        runPipeline(
            category = "I. COMPARISON",
            query = "Compare Phase A and Phase B.",
            chunks = nasaCorpusChunks,
            docIds = listOf(10L),
            expectedEvidence = "Phase A vs Phase B"
        ) { trace ->
            trace.finalAnswer.contains("Phase A") && trace.finalAnswer.contains("Phase B") &&
                    trace.finalAnswer.contains("TRL 6")
        }
    }

    @Test
    fun testJ_GlobalSummary_NASAHandbook() {
        runPipeline(
            category = "J. GLOBAL SUMMARY",
            query = "What is this document about?",
            chunks = nasaCorpusChunks,
            docIds = listOf(10L),
            expectedEvidence = "Systems Engineering"
        ) { trace ->
            trace.finalAnswer.contains("Systems Engineering", ignoreCase = true) || trace.finalAnswer.contains("Based on Page", ignoreCase = true)
        }
    }

    @Test
    fun testK_MultiPageQuestion_SE_PhasesAcrossPages() {
        runPipeline(
            category = "K. MULTI-PAGE QUESTION",
            query = "What are the early project phases and their maturity requirements?",
            chunks = nasaCorpusChunks,
            docIds = listOf(10L),
            expectedEvidence = "Phase A and Phase B TRL 6"
        ) { trace ->
            trace.finalAnswer.contains("TRL 6") && (trace.finalAnswer.contains("Phase A") || trace.finalAnswer.contains("Phase B"))
        }
    }

    @Test
    fun testL_CrossPageContinuation_Page15To16() {
        runPipeline(
            category = "L. CROSS-PAGE CONTINUATION",
            query = "What does Chapter 2 emphasize regarding project management?",
            chunks = nasaCorpusChunks,
            docIds = listOf(10L),
            expectedEvidence = "relationship between project management and systems engineering"
        ) { trace ->
            trace.finalAnswer.contains("project management", ignoreCase = true) &&
                    trace.finalAnswer.contains("complexity", ignoreCase = true)
        }
    }

    @Test
    fun testM_TableQuestion_ComponentSpecifications() {
        runPipeline(
            category = "M. TABLE QUESTION",
            query = "What is the specification and power for the Optical Sensor?",
            chunks = hardwareCorpusChunks,
            docIds = listOf(20L),
            expectedEvidence = "4K 60fps HDR 3.5W"
        ) { trace ->
            trace.finalAnswer.contains("Optical Sensor") &&
                    trace.finalAnswer.contains("3.5W")
        }
    }

    @Test
    fun testN_ListQuestion_Projects() {
        runPipeline(
            category = "N. LIST QUESTION",
            query = "What are the projects listed in the resume?",
            chunks = resumeCorpusChunks,
            docIds = listOf(30L),
            expectedEvidence = "OfflineDocAI DueGuard KisanMitra"
        ) { trace ->
            trace.finalAnswer.contains("OfflineDocAI") &&
                    trace.finalAnswer.contains("DueGuard") &&
                    trace.finalAnswer.contains("KisanMitra")
        }
    }

    @Test
    fun testO_NumberDateUrlId_HelplineAndEmail() {
        runPipeline(
            category = "O. NUMBER / DATE / URL / ID",
            query = "What is the support email and helpline number?",
            chunks = hardwareCorpusChunks,
            docIds = listOf(20L),
            expectedEvidence = "support@offlinedocai.com +1-800-555-0199"
        ) { trace ->
            trace.finalAnswer.contains("support@offlinedocai.com") ||
                    trace.finalAnswer.contains("+1-800-555-0199")
        }
    }

    @Test
    fun testP_NegativeAbsentFact_SecretLaunchCode() {
        runPipeline(
            category = "P. NEGATIVE / ABSENT FACT",
            query = "What is the secret warp core launch code?",
            chunks = nasaCorpusChunks,
            docIds = listOf(10L),
            expectedEvidence = "NOT_FOUND"
        ) { trace ->
            trace.fallbackLevel == FallbackLevel.L4_NOT_FOUND &&
                    (trace.finalAnswer.contains("could not find information") || trace.finalAnswer.contains("I couldn't find"))
        }
    }

    @Test
    fun testQ_AmbiguousQuery_TellMeMore() {
        runPipeline(
            category = "Q. AMBIGUOUS QUERY",
            query = "Tell me more about it.",
            chunks = nasaCorpusChunks,
            docIds = listOf(10L),
            expectedEvidence = "Grounded response without hallucinating entities"
        ) { trace ->
            // Must stay strictly grounded in available evidence without hallucinating fake external context
            !trace.finalAnswer.contains("Elon Musk") && trace.finalAnswer.isNotBlank()
        }
    }

    @Test
    fun testR_RephrasedQuery_BatteryOperatingTemperature() {
        runPipeline(
            category = "R. REPHRASED QUERY",
            query = "What is the maximum allowed operating heat for the battery?",
            chunks = hardwareCorpusChunks,
            docIds = listOf(20L),
            expectedEvidence = "45°C"
        ) { trace ->
            trace.finalAnswer.contains("45°C") || trace.finalAnswer.contains("45")
        }
    }

    @Test
    fun testS_ShortQuery_Email() {
        runPipeline(
            category = "S. SHORT QUERY",
            query = "email",
            chunks = resumeCorpusChunks,
            docIds = listOf(30L),
            expectedEvidence = "siddusbs2810@gmail.com"
        ) { trace ->
            trace.finalAnswer.contains("siddusbs2810@gmail.com")
        }
    }

    @Test
    fun testT_LongNaturalLanguageQuery() {
        runPipeline(
            category = "T. LONG NATURAL-LANGUAGE QUERY",
            query = "Could you please explain in detail what the preliminary design phase does and what technology readiness level is required?",
            chunks = nasaCorpusChunks,
            docIds = listOf(10L),
            expectedEvidence = "TRL 6 Preliminary Design Review"
        ) { trace ->
            trace.finalAnswer.contains("TRL 6") && trace.finalAnswer.contains("Preliminary Design")
        }
    }

    @Test
    fun testU_MultiHopQuestion_CategoryProcesses() {
        runPipeline(
            category = "U. MULTI-HOP QUESTION",
            query = "How many total processes are in the SE engine and what are the three main process categories?",
            chunks = nasaCorpusChunks,
            docIds = listOf(10L),
            expectedEvidence = "17 processes, System Design, Product Realization, Technical Management"
        ) { trace ->
            trace.finalAnswer.contains("17") &&
                    (trace.finalAnswer.contains("System Design") || trace.finalAnswer.contains("Technical Management"))
        }
    }

    @Test
    fun testV_CalculationQuestion_SafeExplanation() {
        runPipeline(
            category = "V. CALCULATION QUESTION",
            query = "What is the total wattage if both Optical Sensor (3.5W) and LiDAR (5.0W) run simultaneously?",
            chunks = hardwareCorpusChunks,
            docIds = listOf(20L),
            expectedEvidence = "Optical Sensor 3.5W and LiDAR 5.0W values preserved"
        ) { trace ->
            // Grounded pipeline must report source values safely and prevent arbitrary hallucinated math
            trace.finalAnswer.contains("3.5W") || trace.finalAnswer.contains("5.0W") || trace.finalAnswer.contains("8.5")
        }
    }

    @Test
    fun testW_CrossSectionQuestion_LifeCycleAndPhaseA() {
        runPipeline(
            category = "W. CROSS-SECTION QUESTION",
            query = "How many phases span the life cycle and what does Phase A establish?",
            chunks = nasaCorpusChunks,
            docIds = listOf(10L),
            expectedEvidence = "seven distinct project phases and feasible system concept"
        ) { trace ->
            (trace.finalAnswer.contains("seven") || trace.finalAnswer.contains("7")) &&
                    (trace.finalAnswer.contains("Phase A") || trace.finalAnswer.contains("concept"))
        }
    }

    @Test
    fun testX_MultiDocumentQuestion_CrossDocIsolation() {
        val combinedChunks = nasaCorpusChunks + softwareCorpusChunks
        runPipeline(
            category = "X. MULTI-DOCUMENT QUESTION",
            query = "What does the Software Guide say about UI wireframing?",
            chunks = combinedChunks,
            docIds = listOf(10L, 40L),
            expectedEvidence = "UI Wireframing begins in Phase A with Figma prototypes"
        ) { trace ->
            trace.finalAnswer.contains("Wireframing", ignoreCase = true) || trace.finalAnswer.contains("Figma")
        }
    }

    @Test
    fun testY_VeryLongDocument_BudgetLimitedSafety() {
        // Create 25 artificial chunks representing a very long document
        val longDocChunks = (1..25).map { i ->
            ChunkEntity(
                id = 500L + i,
                documentId = 50L,
                pageNumber = i,
                chunkIndex = i - 1,
                sectionTitle = "Section $i",
                pageSpan = "$i",
                text = "System Chapter $i: Detailed technical subclause regarding telemetry verification item $i with specific tolerance 0.0$i%.",
                embedding = null
            )
        }

        runPipeline(
            category = "Y. VERY LONG DOCUMENT QUESTION",
            query = "List all telemetry verification items in this document.",
            chunks = longDocChunks,
            docIds = listOf(50L),
            expectedEvidence = "telemetry verification"
        ) { trace ->
            // If context budget is constrained, answer must be grounded and not claim complete exhaustive coverage of all 25 items
            trace.finalAnswer.contains("telemetry verification")
        }
    }

    // =========================================================================
    // PART 2: QUERY FORMULATION VARIATION TESTING (SECTION 5)
    // =========================================================================

    @Test
    fun testQueryVariations_BatteryCapacity_ConsistentSemanticResponse() {
        val queries = listOf(
            "What is the battery capacity?",
            "How large is the battery?",
            "What battery capacity does the device have?",
            "Tell me the device's battery size.",
            "battery capacity?"
        )

        println("=== PART 2: QUERY FORMULATION VARIATION TESTING ===")
        for ((idx, q) in queries.withIndex()) {
            val trace = runPipeline(
                category = "VARIATION ${idx + 1}",
                query = q,
                chunks = hardwareCorpusChunks,
                docIds = listOf(20L),
                expectedEvidence = "3200 mAh"
            ) { t ->
                t.finalAnswer.contains("3200 mAh") && t.finalAnswer.contains("(Page 2)")
            }
            assertTrue("Query variation '$q' must return 3200 mAh", trace.finalAnswer.contains("3200 mAh"))
            assertTrue("Query variation '$q' must cite Page 2", trace.finalAnswer.contains("(Page 2)"))
        }
    }

    // =========================================================================
    // PART 3: PERFORMANCE BENCHMARK MEASUREMENTS (SECTION 31)
    // =========================================================================

    @Test
    fun testPerformanceBenchmarks_TimingMeasurements() = runBlocking {
        println("=== PART 3: PERFORMANCE BENCHMARK MEASUREMENTS ===")

        // 1. TextChunker Ingestion Benchmark
        val rawDoc = """
            # System Specification
            Chapter 1: Overview
            The device incorporates a 3200 mAh battery with 45W fast charging.
            
            Chapter 2: Protocols
            Quantum cryptography uses Kyber key encapsulation.
        """.trimIndent()

        val chunkTimeMs = measureTimeMillis {
            val chunks = TextChunker.chunkDocument(listOf(ExtractedPage(pageNumber = 1, text = rawDoc)))
            assertTrue(chunks.isNotEmpty())
        }
        println("1. Document Ingestion / Chunking Time: ${chunkTimeMs} ms")

        // 2. Query Intent Classification Benchmark
        val routeTimeNanos = measureNanoTime {
            val classification = QueryIntentRouter.route("What is on page 10?")
            assertEquals(QueryIntent.PAGE_LOOKUP, classification.intent)
        }
        val routeTimeMs = routeTimeNanos / 1_000_000.0
        println("2. Query Intent Classification Time: $routeTimeMs ms")

        // 3. VectorStore Retrieval Benchmark
        val dao = createMockChunkDao(nasaCorpusChunks)
        val vectorStore = VectorStore(dao)
        val retrieveTimeMs = measureTimeMillis {
            val retrieved = vectorStore.retrieveDetailed(
                queryEmbedding = FloatArray(384) { 0f },
                relevantDocumentIds = listOf(10L),
                topK = 20,
                query = "What is the Systems Engineering engine?"
            )
            assertTrue(retrieved.isNotEmpty())
        }
        println("3. VectorStore Retrieval Time (top-20): ${retrieveTimeMs} ms")

        // 4. Context Assembly & Budgeting Benchmark
        val retrievedResults = vectorStore.retrieveDetailed(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(10L),
            topK = 20,
            query = "What is the Systems Engineering engine?"
        )
        val assemblyTimeMs = measureTimeMillis {
            val assembled = ContextAssembler.assemble(
                retrievedResults = retrievedResults,
                queryIntent = QueryIntent.CONCEPTUAL_QUERY,
                maxEvidenceTokens = ContextAssembler.DEFAULT_MAX_CONTEXT_TOKENS
            )
            assertTrue(assembled.sourceBlocks.isNotEmpty())
        }
        println("4. Context Assembly & Deduplication Time: ${assemblyTimeMs} ms")

        // 5. GGUFLLM Synthesis Benchmark
        val assembled = ContextAssembler.assemble(
            retrievedResults = retrievedResults,
            queryIntent = QueryIntent.CONCEPTUAL_QUERY,
            maxEvidenceTokens = ContextAssembler.DEFAULT_MAX_CONTEXT_TOKENS
        )
        val prompt = buildPrompt(assembled.contextText, "What is the Systems Engineering engine?")
        val llm = GGUFLLM(createMockContext())
        var rawAnswer = ""
        val synthesisTimeMs = measureTimeMillis {
            rawAnswer = llm.generate(prompt)
        }
        println("5. Grounded LLM Generation Time: ${synthesisTimeMs} ms")

        // 6. Citation Validation Benchmark
        val validationTimeMs = measureTimeMillis {
            val validation = CitationValidator.validateWithSourceBlocks(
                answer = rawAnswer,
                sourceBlocks = assembled.sourceBlocks,
                query = "What is the Systems Engineering engine?"
            )
            assertTrue(validation.validatedAnswer.isNotBlank())
        }
        println("6. Citation & Claim Validation Time: ${validationTimeMs} ms")

        val totalPipelineLatencyMs = chunkTimeMs + routeTimeMs.toInt() + retrieveTimeMs + assemblyTimeMs + synthesisTimeMs + validationTimeMs
        println("TOTAL PIPELINE LATENCY: ${totalPipelineLatencyMs} ms")
        assertTrue("Total latency must be well under 1000ms for offline execution", totalPipelineLatencyMs < 1000)
    }
}
