package com.siddu.offlinedocai

import com.siddu.offlinedocai.ai.llm.GGUFLLM
import com.siddu.offlinedocai.ai.retrieval.*
import com.siddu.offlinedocai.data.entities.ChunkEntity
import org.junit.Assert.*
import org.junit.Test

class ChatRepositoryContextTest {

    private fun makeChunk(
        id: Long,
        page: Int,
        index: Int,
        text: String,
        section: String? = null,
        docId: Long = 1L,
        span: String = ""
    ): ChunkEntity {
        return ChunkEntity(
            id = id,
            documentId = docId,
            pageNumber = page,
            pageSpan = span,
            chunkIndex = index,
            sectionTitle = section,
            text = text,
            embedding = null
        )
    }

    private fun makeResult(
        chunk: ChunkEntity,
        score: Float = 1.0f,
        source: RetrievalSource = RetrievalSource.HYBRID_SEMANTIC,
        intent: QueryIntent = QueryIntent.CONCEPTUAL_QUERY
    ): RetrievalResult {
        return RetrievalResult(
            chunk = chunk,
            relevanceScore = score,
            source = source,
            intent = intent
        )
    }

    // =========================================================================
    // A. DOCUMENT ORDER
    // =========================================================================
    @Test
    fun testA_DocumentOrder_RetrievedPagesOrderedNaturally() {
        // Retrieved pages: 20 (score 9.2), 3 (score 8.8), 7 (score 8.5), 4 (score 7.9)
        val c20 = makeChunk(200, 20, 15, "Page 20 concluding remarks.")
        val c3 = makeChunk(30, 3, 2, "Page 3 introduction to architecture.")
        val c7 = makeChunk(70, 7, 5, "Page 7 data pipeline details.")
        val c4 = makeChunk(40, 4, 3, "Page 4 system requirements.")

        val retrieved = listOf(
            makeResult(c20, 9.2f),
            makeResult(c3, 8.8f),
            makeResult(c7, 8.5f),
            makeResult(c4, 7.9f)
        )

        val assembled = ContextAssembler.assemble(
            retrievedResults = retrieved,
            queryIntent = QueryIntent.CONCEPTUAL_QUERY
        )

        assertEquals(4, assembled.sourceBlocks.size)
        // Verify document order: Page 3, Page 4, Page 7, Page 20
        assertEquals(3, assembled.sourceBlocks[0].pageNumber)
        assertEquals(4, assembled.sourceBlocks[1].pageNumber)
        assertEquals(7, assembled.sourceBlocks[2].pageNumber)
        assertEquals(20, assembled.sourceBlocks[3].pageNumber)

        val idx3 = assembled.contextText.indexOf("[PAGE=3]")
        val idx4 = assembled.contextText.indexOf("[PAGE=4]")
        val idx7 = assembled.contextText.indexOf("[PAGE=7]")
        val idx20 = assembled.contextText.indexOf("[PAGE=20]")

        assertTrue("Page 3 must precede Page 4", idx3 < idx4)
        assertTrue("Page 4 must precede Page 7", idx4 < idx7)
        assertTrue("Page 7 must precede Page 20", idx7 < idx20)
    }

    // =========================================================================
    // B. DUPLICATE REMOVAL
    // =========================================================================
    @Test
    fun testB_DuplicateRemoval_DenseAndLexicalDuplicatesMerged() {
        val c1 = makeChunk(101, 5, 4, "Unique specification chunk text.")
        // Same chunk returned via dense semantic search (score 7.5) and lexical BM25 (score 9.1)
        val denseResult = makeResult(c1, 7.5f, RetrievalSource.HYBRID_SEMANTIC)
        val lexicalResult = makeResult(c1, 9.1f, RetrievalSource.EXACT_FACT_LEXICAL)

        val assembled = ContextAssembler.assemble(
            retrievedResults = listOf(denseResult, lexicalResult),
            queryIntent = QueryIntent.EXACT_FACT
        )

        assertEquals("Duplicate chunk must appear only once", 1, assembled.sourceBlocks.size)
        assertEquals(101L, assembled.sourceBlocks[0].sourceId)
        // Retained higher relevance score
        assertEquals(9.1f, assembled.sourceBlocks[0].relevanceScore, 0.01f)
    }

    // =========================================================================
    // C. ADJACENT MERGE
    // =========================================================================
    @Test
    fun testC_AdjacentMerge_ContiguousChunksMergedIntoSingleBlock() {
        // Chunks 10, 11, 12 from same document and section
        val c10 = makeChunk(10, 2, 10, "Paragraph one of methodology.", section = "Methodology")
        val c11 = makeChunk(11, 2, 11, "Paragraph two of methodology.", section = "Methodology")
        val c12 = makeChunk(12, 2, 12, "Paragraph three of methodology.", section = "Methodology")

        val retrieved = listOf(
            makeResult(c10, 8.0f),
            makeResult(c11, 8.5f),
            makeResult(c12, 8.2f)
        )

        val assembled = ContextAssembler.assemble(
            retrievedResults = retrieved,
            queryIntent = QueryIntent.CONCEPTUAL_QUERY
        )

        assertEquals("Contiguous adjacent chunks must merge into a single coherent block", 1, assembled.sourceBlocks.size)
        val merged = assembled.sourceBlocks[0]
        assertEquals(10L, merged.sourceId)
        assertEquals(10, merged.chunkIndex)
        assertEquals("Methodology", merged.sectionTitle)
        assertTrue(merged.text.contains("Paragraph one"))
        assertTrue(merged.text.contains("Paragraph two"))
        assertTrue(merged.text.contains("Paragraph three"))
        assertEquals(8.5f, merged.relevanceScore, 0.01f)
    }

    // =========================================================================
    // D. EXHAUSTIVE
    // =========================================================================
    @Test
    fun testD_Exhaustive_AllDistinctProjectsRetainedWithinBudget() {
        // Projects on pages 2, 10, 25, 40
        val p1 = makeChunk(1, 2, 0, "Project Apollo: Lunar landing mission.")
        val p2 = makeChunk(2, 10, 5, "Project Voyager: Deep space interstellar exploration.")
        val p3 = makeChunk(3, 25, 12, "Project Hubble: Orbital optical observatory.")
        val p4 = makeChunk(4, 40, 20, "Project Artemis: Modern crewed lunar exploration program.")

        val retrieved = listOf(
            makeResult(p1, 8.0f, RetrievalSource.EXHAUSTIVE_SCAN),
            makeResult(p2, 8.0f, RetrievalSource.EXHAUSTIVE_SCAN),
            makeResult(p3, 8.0f, RetrievalSource.EXHAUSTIVE_SCAN),
            makeResult(p4, 8.0f, RetrievalSource.EXHAUSTIVE_SCAN)
        )

        val assembled = ContextAssembler.assemble(
            retrievedResults = retrieved,
            queryIntent = QueryIntent.EXHAUSTIVE_LIST,
            maxEvidenceTokens = 1000
        )

        assertEquals(4, assembled.sourceBlocks.size)
        assertFalse("All items fit, so isBudgetLimited must be false", assembled.isBudgetLimited)
        val text = assembled.contextText
        assertTrue(text.contains("Project Apollo"))
        assertTrue(text.contains("Project Voyager"))
        assertTrue(text.contains("Project Hubble"))
        assertTrue(text.contains("Project Artemis"))
    }

    // =========================================================================
    // E. FIND_ALL
    // =========================================================================
    @Test
    fun testE_FindAll_OccurrencesAcrossPagesRemainDistinct() {
        // Occurrences of Authentication on pages 3, 12, 30, 55
        val o1 = makeChunk(1, 3, 1, "Section 1: Authentication uses SHA-256 password hashing.")
        val o2 = makeChunk(2, 12, 8, "Section 4: API Authentication requires bearer JWT tokens.")
        val o3 = makeChunk(3, 30, 18, "Section 8: Biometric Authentication via fingerprint scanner.")
        val o4 = makeChunk(4, 55, 35, "Section 14: OAuth2 Authentication flow with refresh token exchange.")

        val retrieved = listOf(
            makeResult(o1, 8.5f, RetrievalSource.FIND_ALL_SCAN),
            makeResult(o2, 8.3f, RetrievalSource.FIND_ALL_SCAN),
            makeResult(o3, 8.1f, RetrievalSource.FIND_ALL_SCAN),
            makeResult(o4, 8.0f, RetrievalSource.FIND_ALL_SCAN)
        )

        val assembled = ContextAssembler.assemble(
            retrievedResults = retrieved,
            queryIntent = QueryIntent.FIND_ALL
        )

        assertEquals(4, assembled.sourceBlocks.size)
        val pages = assembled.sourceBlocks.map { it.pageNumber }
        assertEquals(listOf(3, 12, 30, 55), pages)
    }

    // =========================================================================
    // F. COMPARISON
    // =========================================================================
    @Test
    fun testF_Comparison_BalancedRepresentationForBothEntities() {
        // 4 chunks for Apollo, 2 chunks for Voyager
        val a1 = makeChunk(1, 2, 0, "Apollo launched with Saturn V rocket.", section = "Apollo")
        val a2 = makeChunk(2, 3, 1, "Apollo 11 landed humans on the Moon.", section = "Apollo")
        val a3 = makeChunk(3, 4, 2, "Apollo used command module and lunar module.", section = "Apollo")
        val a4 = makeChunk(4, 5, 3, "Apollo telemetry used S-band transponders.", section = "Apollo")

        val v1 = makeChunk(5, 10, 6, "Voyager 1 was launched with Titan IIIE.", section = "Voyager")
        val v2 = makeChunk(6, 11, 7, "Voyager traversed interstellar space beyond heliosphere.", section = "Voyager")

        val classification = QueryClassification(
            intent = QueryIntent.COMPARISON,
            confidence = 0.95f,
            entityA = "Apollo",
            entityB = "Voyager",
            rawQuery = "Compare Apollo and Voyager"
        )

        val retrieved = listOf(
            makeResult(a1, 9.0f),
            makeResult(a2, 8.8f),
            makeResult(a3, 8.6f),
            makeResult(a4, 8.4f),
            makeResult(v1, 8.5f),
            makeResult(v2, 8.3f)
        )

        // Give a budget that accommodates ~4 chunks
        val assembled = ContextAssembler.assemble(
            retrievedResults = retrieved,
            queryIntent = QueryIntent.COMPARISON,
            queryClassification = classification,
            maxEvidenceTokens = 200
        )

        val fullText = assembled.contextText
        assertTrue("Entity A (Apollo) must be represented", fullText.contains("Apollo"))
        assertTrue("Entity B (Voyager) must be represented", fullText.contains("Voyager"))
    }

    // =========================================================================
    // G. EXACT FACT
    // =========================================================================
    @Test
    fun testG_ExactFact_PhoneEmailUrlPrioritized() {
        val boilerplate = makeChunk(1, 1, 0, "Welcome to Government College of Engineering official handbook.")
        val phoneChunk = makeChunk(2, 3, 4, "Contact Phone: +91 98765 43210. Direct office extension 402.")
        val genericAbout = makeChunk(3, 1, 1, "General engineering curriculum regulations and guidelines.")

        val retrieved = listOf(
            makeResult(boilerplate, 6.0f, RetrievalSource.HYBRID_SEMANTIC),
            makeResult(phoneChunk, 9.8f, RetrievalSource.EXACT_FACT_LEXICAL),
            makeResult(genericAbout, 6.5f, RetrievalSource.HYBRID_SEMANTIC)
        )

        val classification = QueryClassification(
            intent = QueryIntent.EXACT_FACT,
            confidence = 0.98f,
            factType = FactType.PHONE,
            rawQuery = "What is the phone number?"
        )

        val assembled = ContextAssembler.assemble(
            retrievedResults = retrieved,
            queryIntent = QueryIntent.EXACT_FACT,
            queryClassification = classification
        )

        val exactBlock = assembled.sourceBlocks.find { it.text.contains("+91 98765 43210") }
        assertNotNull("Exact fact chunk must be preserved in context", exactBlock)
        assertTrue(assembled.contextText.contains("+91 98765 43210"))
    }

    // =========================================================================
    // H. PAGE LOOKUP
    // =========================================================================
    @Test
    fun testH_PageLookup_RequestedPageEvidenceKeptIntact() {
        val p10_a = makeChunk(101, 10, 0, "Page 10 Paragraph 1: Database schemas.")
        val p10_b = makeChunk(102, 10, 1, "Page 10 Paragraph 2: Room persistence configurations.")
        val p12_noise = makeChunk(201, 12, 5, "Page 12: Network protocols.")

        val classification = QueryClassification(
            intent = QueryIntent.PAGE_LOOKUP,
            confidence = 0.99f,
            pageStart = 10,
            rawQuery = "What is on page 10?"
        )

        val retrieved = listOf(
            makeResult(p10_a, 9.0f, RetrievalSource.PAGE_DETERMINISTIC),
            makeResult(p10_b, 9.0f, RetrievalSource.PAGE_DETERMINISTIC),
            makeResult(p12_noise, 4.0f, RetrievalSource.HYBRID_SEMANTIC)
        )

        val assembled = ContextAssembler.assemble(
            retrievedResults = retrieved,
            queryIntent = QueryIntent.PAGE_LOOKUP,
            queryClassification = classification
        )

        assertEquals("Page lookup must keep only page 10 chunks", 1, assembled.sourceBlocks.size)
        // Notice chunks 0 and 1 on page 10 are adjacent, so they cleanly merge into 1 source block!
        val block = assembled.sourceBlocks[0]
        assertEquals(10, block.pageNumber)
        assertTrue(block.text.contains("Database schemas"))
        assertTrue(block.text.contains("Room persistence"))
        assertFalse(block.text.contains("Network protocols"))
    }

    // =========================================================================
    // I. CONTEXT LIMIT
    // =========================================================================
    @Test
    fun testI_ContextLimit_NeverExceedsSafeTokenBudget() {
        // Create 20 large chunks
        val largeChunks = (1..20).map { i ->
            val text = "Paragraph $i: Detailed technical architecture description of distributed fault tolerance, state replication, Byzantine agreement, quorum intersection, and storage durability guarantees across edge cluster nodes. ".repeat(4)
            makeResult(makeChunk(i.toLong(), i, i, text), 7.0f + (i * 0.1f))
        }

        val budgetLimit = 500
        val assembled = ContextAssembler.assemble(
            retrievedResults = largeChunks,
            queryIntent = QueryIntent.CONCEPTUAL_QUERY,
            maxEvidenceTokens = budgetLimit
        )

        assertTrue("Evidence set must be marked budget-limited", assembled.isBudgetLimited)
        assertTrue("Estimated context tokens must not exceed budget", assembled.totalEstimatedTokens <= budgetLimit + 50)
    }

    // =========================================================================
    // J. NO RAW CHUNK FALLBACK
    // =========================================================================
    @Test
    fun testJ_NoRawChunkFallback_FailedSynthesisExtractsConciseGroundedSentence() {
        val longChunkText = """
            Battery Safety Guidelines and Warning System:
            Do not expose the battery to extreme temperatures exceeding 60 degrees Celsius.
            Always use official manufacturer certified power adapters to prevent voltage surges.
            The internal rechargeable lithium-ion battery has a rated capacity of 5000 mAh and supports 45W fast charging.
            Overheating may trigger safety circuit disconnect mechanisms.
            Dispose of depleted battery units in accordance with municipal hazardous waste recycling standards.
        """.trimIndent()

        val block = SourceBlock(
            sourceId = 42,
            documentId = 1,
            pageNumber = 5,
            pageSpan = "5",
            chunkIndex = 8,
            sectionTitle = "Battery Specs",
            text = longChunkText,
            relevanceScore = 9.5f
        )

        val fallback = ContextAssembler.extractGroundedFallback(listOf(block), "What is the battery capacity?")

        println("FALLBACK OUTPUT: $fallback")
        assertTrue("Fallback must reference the source page", fallback.contains("Page 5"))
        assertTrue("Fallback must extract grounded information", fallback.contains("5000 mAh") || fallback.contains("battery"))
        assertTrue("Fallback must be concise and NOT dump the entire chunk", fallback.length < 250)
        assertFalse("Must not dump the entire raw chunk", fallback.contains("hazardous waste recycling standards"))
    }

    // =========================================================================
    // K. SOURCE METADATA
    // =========================================================================
    @Test
    fun testK_SourceMetadata_StructuredBlockContainsAllFields() {
        val block = SourceBlock(
            sourceId = 42,
            documentId = 1,
            pageNumber = 12,
            pageSpan = "12-13",
            chunkIndex = 87,
            sectionTitle = "Requirements",
            text = "System must support offline vector indexing.",
            relevanceScore = 9.2f
        )

        val formatted = block.formatBlock()
        println("FORMATTED SOURCE BLOCK:\n$formatted\n")

        assertTrue(formatted.contains("[SOURCE_ID=42]"))
        assertTrue(formatted.contains("[PAGE=12]"))
        assertTrue(formatted.contains("[PAGE_SPAN=12-13]"))
        assertTrue(formatted.contains("[SECTION=Requirements]"))
        assertTrue(formatted.contains("[CHUNK_INDEX=87]"))
        assertTrue(formatted.contains("System must support offline vector indexing."))

        // Verify JSON metadata output
        val assembled = ContextAssembler.assemble(
            retrievedResults = listOf(makeResult(makeChunk(42, 12, 87, "System must support offline vector indexing.", section = "Requirements", span = "12-13"))),
            queryIntent = QueryIntent.CONCEPTUAL_QUERY
        )
        val json = assembled.sourcesJson
        assertTrue(json.contains("\"sourceId\":42"))
        assertTrue(json.contains("\"page\":12"))
        assertTrue(json.contains("\"pageSpan\":\"12-13\""))
        assertTrue(json.contains("\"section\":\"Requirements\""))
    }

    // =========================================================================
    // L. REAL DOCUMENT TESTS (NASA STI & Hardware Engineering)
    // =========================================================================
    @Test
    fun testL_RealDocument_ThreeConcreteExamples() {
        // Realistic NASA STI Document Chunks
        val nasa1_Intro = makeChunk(
            id = 101, page = 1, index = 0,
            section = "Executive Summary",
            text = "NASA Technical Memorandum: Advanced Thermal Protection Systems for Atmospheric Reentry. This report outlines ceramic tile and ablative heat shield performance across Mach 10-25 hypersonic velocity profiles."
        )
        val nasa2_Thermal = makeChunk(
            id = 102, page = 4, index = 3,
            section = "Thermal Analysis",
            text = "Thermal insulation performance: Carbon-silicon carbide ceramic matrix composites withstand surface temperatures exceeding 1650°C during maximum aerodynamic deceleration."
        )
        val nasa3_Aero = makeChunk(
            id = 103, page = 5, index = 4,
            section = "Aerodynamics",
            text = "Hypersonic shock layer dynamics: Plasma sheath formation causes radio frequency blackout between 70 km and 40 km altitude."
        )

        // 1. Multi-Page Conceptual Query
        val query1 = "Explain thermal insulation and aerodynamic shock layer"
        val retrieved1 = listOf(
            makeResult(nasa2_Thermal, 9.4f),
            makeResult(nasa3_Aero, 8.8f),
            makeResult(nasa1_Intro, 7.1f)
        )
        val context1 = ContextAssembler.assemble(retrieved1, QueryIntent.CONCEPTUAL_QUERY)

        println("=== REAL-DOC EXAMPLE 1: MULTI-PAGE CONCEPTUAL QUERY ===")
        println("QUERY: $query1")
        println("RETRIEVED PAGES: ${retrieved1.map { it.chunk.pageNumber }}")
        println("FINAL ASSEMBLED CONTEXT:\n${context1.contextText}\n")
        assertEquals(listOf(1, 4, 5), context1.sourceBlocks.map { it.pageNumber })

        // 2. Comparison Query
        val query2 = "Compare ceramic matrix composites and ablative shields"
        val classification2 = QueryClassification(
            intent = QueryIntent.COMPARISON,
            confidence = 0.95f,
            entityA = "ceramic matrix composites",
            entityB = "ablative heat shield",
            rawQuery = query2
        )
        val context2 = ContextAssembler.assemble(retrieved1, QueryIntent.COMPARISON, classification2)
        println("=== REAL-DOC EXAMPLE 2: COMPARISON QUERY ===")
        println("QUERY: $query2")
        println("FINAL ASSEMBLED CONTEXT:\n${context2.contextText}\n")
        assertTrue(context2.contextText.contains("ceramic"))

        // 3. Exact Fact Query
        val hardwareChunk = makeChunk(
            id = 201, page = 8, index = 14,
            section = "Electrical Specifications",
            text = "Power Subsystem: Operating voltage is 28V DC nominal. Auxiliary battery capacity is 3200 mAh with maximum continuous discharge current of 15A."
        )
        val query3 = "What is the battery capacity?"
        val classification3 = QueryClassification(
            intent = QueryIntent.EXACT_FACT,
            confidence = 0.97f,
            factType = FactType.EXACT_VALUE,
            rawQuery = query3
        )
        val retrieved3 = listOf(
            makeResult(nasa1_Intro, 4.0f, RetrievalSource.HYBRID_SEMANTIC),
            makeResult(hardwareChunk, 9.9f, RetrievalSource.EXACT_FACT_LEXICAL)
        )
        val context3 = ContextAssembler.assemble(retrieved3, QueryIntent.EXACT_FACT, classification3)
        println("=== REAL-DOC EXAMPLE 3: EXACT FACT QUERY ===")
        println("QUERY: $query3")
        println("FINAL ASSEMBLED CONTEXT:\n${context3.contextText}\n")
        assertTrue(context3.contextText.contains("3200 mAh"))
    }

    // =========================================================================
    // M. EVIDENCE BUDGETING — SIMPLE QUERY CONSERVATIVE BUDGET
    // =========================================================================
    @Test
    fun testM_EvidenceBudgeting_SimpleQueryRestrictedToConservativeBudget() {
        // 10 candidate chunks (~60 tokens each = ~600 tokens total)
        val chunks = (1..10).map { i ->
            val text = "Item $i: Specific engineering parameter description with technical specifications and numerical constraints for component $i."
            makeResult(makeChunk(i.toLong(), i, i, text), 8.0f - (i * 0.1f))
        }

        val assembled = ContextAssembler.assemble(
            retrievedResults = chunks,
            queryIntent = QueryIntent.CONCEPTUAL_QUERY
        )

        // Must resolve to BUDGET_SIMPLE_QUERY_TOKENS (384 tokens)
        assertTrue("Assembled context must be marked budget-limited", assembled.isBudgetLimited)
        assertTrue(
            "Estimated context tokens (${assembled.totalEstimatedTokens}) must stay within simple budget (384 + safety)",
            assembled.totalEstimatedTokens <= ContextAssembler.BUDGET_SIMPLE_QUERY_TOKENS + 35
        )
        // Must have selected the strongest chunks (items 1, 2, 3...)
        assertTrue(assembled.contextText.contains("Item 1"))
        assertTrue(assembled.contextText.contains("Item 2"))
        // Complete blocks selected
        assertEquals(assembled.sourceBlocks.size, assembled.sourceBlocks.count { it.text.isNotBlank() })
    }

    // =========================================================================
    // N. EVIDENCE BUDGETING — EXHAUSTIVE QUERY RETAINS FULL CANDIDATES
    // =========================================================================
    @Test
    fun testN_EvidenceBudgeting_ExhaustiveQueryUsesFullBudget() {
        // 8 distinct project chunks across 8 different pages
        val chunks = (1..8).map { i ->
            val text = "Project $i: Strategic initiative on page $i with deliverables and timeline."
            makeResult(makeChunk(i.toLong(), i, 0, text), 7.5f, RetrievalSource.EXHAUSTIVE_SCAN)
        }

        val assembled = ContextAssembler.assemble(
            retrievedResults = chunks,
            queryIntent = QueryIntent.EXHAUSTIVE_LIST
        )

        // Exhaustive budget is 1152 tokens, so all 8 projects must easily fit without limitation
        assertFalse("All 8 projects should fit within 1152 token exhaustive budget", assembled.isBudgetLimited)
        assertEquals("All 8 project chunks must be retained", 8, assembled.sourceBlocks.size)
        val pages = assembled.sourceBlocks.map { it.pageNumber }
        assertEquals((1..8).toList(), pages)
    }

    // =========================================================================
    // O. EVIDENCE BUDGETING — NOISE FILTERING FOR SIMPLE QUERIES
    // =========================================================================
    @Test
    fun testO_EvidenceBudgeting_LowRelevanceNoiseFilteredForSimpleQuery() {
        val topChunk = makeChunk(1, 2, 0, "Top relevant fact: Callus healing temperature is 21C to 27C.")
        val closeChunk = makeChunk(2, 2, 1, "Secondary detail: Relative humidity should be maintained at 90%.")
        val noiseChunk = makeChunk(3, 14, 5, "Unrelated boilerplate text: Copyright notice and publication disclaimer.")

        val retrieved = listOf(
            makeResult(topChunk, 9.5f),
            makeResult(closeChunk, 7.8f),
            makeResult(noiseChunk, 1.2f) // < 30% of 9.5f (below 2.85)
        )

        val assembled = ContextAssembler.assemble(
            retrievedResults = retrieved,
            queryIntent = QueryIntent.CONCEPTUAL_QUERY
        )

        val text = assembled.contextText
        assertTrue("Top chunk must be present", text.contains("Callus healing temperature"))
        assertTrue("Close chunk must be present", text.contains("Relative humidity"))
        assertFalse("Low-relevance noise chunk must be filtered out", text.contains("Copyright notice"))
    }

    // =========================================================================
    // P. QWEN PROMPT FORMAT — ASSISTANT PREFIX ENDS WITH NEWLINE
    // =========================================================================
    @Test
    fun testP_QwenPromptFormat_AssistantPrefixEndsWithNewlineAndSpecialTokensPreserved() {
        val tempDir = java.io.File(System.getProperty("java.io.tmpdir"), "offlinedocai_prompt_test")
        tempDir.mkdirs()
        val mockContext = object : android.content.ContextWrapper(null) {
            override fun getFilesDir(): java.io.File = tempDir
        }
        val llm = GGUFLLM(mockContext)

        val contextSnippet = "[SOURCE_ID=1]\n[PAGE=1]\nCallus formation occurs at 21C to 27C."
        val question = "What is the callus healing temperature?"

        val prompt = llm.buildQwenPrompt(contextSnippet, question)

        // 1. Assistant generation prefix MUST end with <|im_start|>assistant\n
        assertTrue(
            "Prompt must terminate with <|im_start|>assistant\\n exactly",
            prompt.endsWith("<|im_start|>assistant\n")
        )

        // 2. System turn structure
        assertTrue("Prompt must start with <|im_start|>system\\n", prompt.startsWith("<|im_start|>system\n"))
        assertTrue("System message must end with <|im_end|>\\n", prompt.contains("<|im_end|>\n<|im_start|>user\n"))

        // 3. User turn structure
        assertTrue("User message must contain context", prompt.contains("Context:\n$contextSnippet"))
        assertTrue("User message must contain question", prompt.contains("Question: $question"))
        assertTrue("User message must end with <|im_end|>\\n", prompt.contains("<|im_end|>\n<|im_start|>assistant\n"))

        // 4. No BOS token
        assertFalse("Prompt should not inject leading BOS token", prompt.startsWith("<|endoftext|>"))
    }
}
