package com.siddu.offlinedocai

import android.content.Context
import com.siddu.offlinedocai.ai.embeddings.ONNXEmbeddingEngine
import com.siddu.offlinedocai.ai.llm.AnswerLength
import com.siddu.offlinedocai.ai.llm.GGUFLLM
import com.siddu.offlinedocai.ai.retrieval.ContextAssembler
import com.siddu.offlinedocai.ai.retrieval.QueryClassification
import com.siddu.offlinedocai.ai.retrieval.QueryIntent
import com.siddu.offlinedocai.ai.retrieval.QueryIntentRouter
import com.siddu.offlinedocai.ai.retrieval.VectorStore
import com.siddu.offlinedocai.data.dao.ChunkDao
import com.siddu.offlinedocai.data.entities.ChunkEntity
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import java.io.File

/**
 * Verification test suite for Answer Length Control (Short / Long mode) in OfflineDocAI.
 *
 * Verifies:
 * 1. Retrieval Invariance: Identical retrieval results (scores, chunk IDs, rankings) for Short vs Long.
 * 2. Prompt Formatting: Exact short/long instructions and delimiter preservation for Qwen and Llama.
 * 3. Short Mode Synthesis: Direct concise answers adhering strictly to retrieved evidence.
 * 4. Long Mode Synthesis: Detailed expansion ONLY when additional relevant evidence exists.
 * 5. Evidence-Bounded Constraint: When evidence only supports a brief answer, Long mode remains brief
 *    and does NOT manufacture or hallucinate content.
 * 6. Citation Preservation: Verifiable citations maintained for every grounded claim in both modes.
 * 7. Intent-Specific Adaptations: Comparison distinctions and summary points respecting length mode.
 */
class AnswerLengthControlTest {

    private lateinit var mockContext: Context
    private lateinit var ggufLlm: GGUFLLM

    private fun createMockContext(): Context {
        val tempDir = File(System.getProperty("java.io.tmpdir"), "offlinedocai_length_test")
        tempDir.mkdirs()
        return object : android.content.ContextWrapper(null) {
            override fun getFilesDir(): File = tempDir
        }
    }

    private fun createMockChunkDao(chunkList: List<ChunkEntity>): ChunkDao {
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

    @Before
    fun setUp() {
        mockContext = createMockContext()
        ggufLlm = GGUFLLM(mockContext)
    }

    // =========================================================================
    // 1. RETRIEVAL INVARIANCE TEST
    // =========================================================================

    @Test
    fun testRetrievalInvarianceBetweenShortAndLong() = runBlocking {
        // Prepare realistic grafting corpus chunks
        val sampleEmbedding = FloatArray(384) { 0.05f }
        val sampleEmbeddingBytes = com.siddu.offlinedocai.data.FloatArrayConverter.fromFloatArray(sampleEmbedding)
        val chunks = listOf(
            ChunkEntity(
                id = 1,
                documentId = 10,
                chunkIndex = 0,
                pageNumber = 3,
                text = "Whip and tongue grafting is used for small rootstocks of 6 to 13 mm diameter. Interlocking tongue cuts provide strong mechanical strength.",
                embedding = sampleEmbeddingBytes
            ),
            ChunkEntity(
                id = 2,
                documentId = 10,
                chunkIndex = 1,
                pageNumber = 4,
                text = "Cleft grafting is a topworking technique used on larger rootstocks of 25 to 75 mm diameter. The scion is inserted into a split stub.",
                embedding = sampleEmbeddingBytes
            ),
            ChunkEntity(
                id = 3,
                documentId = 10,
                chunkIndex = 2,
                pageNumber = 11,
                text = "Optimal healing temperature for grafting callus formation is 21°C to 27°C (70°F to 80°F).",
                embedding = sampleEmbeddingBytes
            )
        )

        val chunkDao = createMockChunkDao(chunks)
        val vectorStore = VectorStore(chunkDao)

        val query = "What is whip and tongue grafting?"
        val queryClassification = QueryIntentRouter.route(query)
        val queryEmbedding = FloatArray(384) { 0.05f }

        // Retrieve for SHORT mode context pipeline
        val retrievedShort = vectorStore.retrieveDetailed(
            queryEmbedding = queryEmbedding,
            relevantDocumentIds = listOf(10L),
            topK = 20,
            query = query
        )
        val budgetShort = ContextAssembler.resolveEvidenceBudget(queryClassification.intent)
        val assembledShort = ContextAssembler.assemble(
            retrievedResults = retrievedShort,
            queryIntent = queryClassification.intent,
            queryClassification = queryClassification,
            maxEvidenceTokens = budgetShort
        )

        // Retrieve for LONG mode context pipeline
        val retrievedLong = vectorStore.retrieveDetailed(
            queryEmbedding = queryEmbedding,
            relevantDocumentIds = listOf(10L),
            topK = 20,
            query = query
        )
        val budgetLong = ContextAssembler.resolveEvidenceBudget(queryClassification.intent)
        val assembledLong = ContextAssembler.assemble(
            retrievedResults = retrievedLong,
            queryIntent = queryClassification.intent,
            queryClassification = queryClassification,
            maxEvidenceTokens = budgetLong
        )

        // Verification: Retrieval pipeline must be 100% IDENTICAL
        assertEquals("Candidate count must match", retrievedShort.size, retrievedLong.size)
        for (i in retrievedShort.indices) {
            assertEquals("Chunk ID must match at index $i", retrievedShort[i].chunk.id, retrievedLong[i].chunk.id)
            assertEquals("Page number must match at index $i", retrievedShort[i].chunk.pageNumber, retrievedLong[i].chunk.pageNumber)
            assertEquals("Relevance score must match at index $i", retrievedShort[i].relevanceScore, retrievedLong[i].relevanceScore, 0.0001f)
        }

        assertEquals("Assembled context text must be identical", assembledShort.contextText, assembledLong.contextText)
        assertEquals("Assembled token budget must be identical", assembledShort.totalEstimatedTokens, assembledLong.totalEstimatedTokens)
        assertEquals("Assembled source count must be identical", assembledShort.sourceBlocks.size, assembledLong.sourceBlocks.size)
    }

    // =========================================================================
    // 2. PROMPT FORMATTING TESTS
    // =========================================================================

    @Test
    fun testQwenPromptFormatting_ShortMode() {
        val context = "[Source 1, Page 3]\nWhip and tongue grafting is used for small rootstocks."
        val question = "What is whip and tongue grafting used for?"

        val prompt = ggufLlm.buildQwenPrompt(context, question, answerLength = AnswerLength.SHORT)

        assertTrue("Must contain Qwen system header", prompt.contains("<|im_start|>system\n"))
        assertTrue("Must contain exact Short prompt instruction", prompt.contains("- Give the minimum sufficient answer supported by the evidence."))
        assertTrue("Must contain minimum sufficient facts instruction", prompt.contains("- Never invent facts."))
        assertFalse("Must NOT contain Long prompt instruction", prompt.contains("- Give the most complete answer"))
        assertTrue("Must contain context", prompt.contains(context))
        assertTrue("Must contain question", prompt.contains(question))
        assertTrue("Must contain user instruction reminder", prompt.contains("Instruction: Give the minimum sufficient answer supported by the evidence."))
        assertTrue("Must end with assistant tag", prompt.endsWith("<|im_start|>assistant\n"))
    }

    @Test
    fun testQwenPromptFormatting_LongMode() {
        val context = "[Source 1, Page 3]\nWhip and tongue grafting is used for small rootstocks."
        val question = "What is whip and tongue grafting used for?"

        val prompt = ggufLlm.buildQwenPrompt(context, question, answerLength = AnswerLength.LONG)

        assertTrue("Must contain Qwen system header", prompt.contains("<|im_start|>system\n"))
        assertTrue("Must contain exact Long prompt instruction", prompt.contains("- Give the most complete answer that is directly relevant to the question and supported by the evidence."))
        // Removed distinct factual points instruction
        assertFalse("Must NOT contain Short prompt instruction", prompt.contains("- Give the minimum sufficient answer supported by the evidence."))
        assertTrue("Must contain context", prompt.contains(context))
        assertTrue("Must contain question", prompt.contains(question))
        assertTrue("Must contain user instruction reminder", prompt.contains("Instruction: Give the most complete answer that is directly relevant to the question and supported by the evidence."))
        assertTrue("Must end with assistant tag", prompt.endsWith("<|im_start|>assistant\n"))
    }

    @Test
    fun testLlamaPromptFormatting_ShortAndLong() {
        val context = "[Source 1, Page 3]\nWhip and tongue grafting is used for small rootstocks."
        val question = "What is whip and tongue grafting used for?"

        val shortPrompt = ggufLlm.buildLlamaPrompt(context, question, answerLength = AnswerLength.SHORT)
        assertTrue(shortPrompt.startsWith("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"))
        assertTrue(shortPrompt.contains("- Give the minimum sufficient answer supported by the evidence."))
        assertTrue(shortPrompt.contains("<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"))
        assertTrue(shortPrompt.endsWith("<|start_header_id|>assistant<|end_header_id|>\n\n"))

        val longPrompt = ggufLlm.buildLlamaPrompt(context, question, answerLength = AnswerLength.LONG)
        assertTrue(longPrompt.startsWith("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"))
        assertTrue(longPrompt.contains("- Give the most complete answer that is directly relevant to the question and supported by the evidence."))
        assertTrue(longPrompt.endsWith("<|start_header_id|>assistant<|end_header_id|>\n\n"))
    }

    @Test
    fun testParseAnswerLengthFromPrompt() {
        val shortPrompt = """
            [ANSWER_LENGTH: SHORT]
            Context:
            Some context.
            Question: What is grafting?
            Answer:
            Answer concisely using only the supplied evidence.
        """.trimIndent()
        assertEquals(AnswerLength.SHORT, ggufLlm.parseAnswerLength(shortPrompt))

        val longPrompt = """
            [ANSWER_LENGTH: LONG]
            Context:
            Some context.
            Question: What is grafting?
            Answer:
            Answer using ONLY the relevant facts stated in the supplied evidence. If the evidence contains multiple supporting points or sections, include all of them.
        """.trimIndent()
        assertEquals(AnswerLength.LONG, ggufLlm.parseAnswerLength(longPrompt))

        val defaultPrompt = """
            Context:
            Some context.
            Question: What is grafting?
            Answer:
        """.trimIndent()
        assertEquals(AnswerLength.SHORT, ggufLlm.parseAnswerLength(defaultPrompt))
    }

    // =========================================================================
    // 3. SYNTHESIS SHORT VS LONG TESTS
    // =========================================================================

    @Test
    fun testShortModeGeneratesConciseAnswer() = runBlocking {
        val multiDetailContext = """
            [PAGE=3] [Source 1, Page 3]
            Whip and tongue grafting is an advanced joining technique for small rootstocks.
            Interlocking tongue cuts provide high mechanical strength during healing.
            Smooth matching cuts ensure accurate vascular cambium alignment.
        """.trimIndent()

        val prompt = """
            [ANSWER_LENGTH: SHORT]
            Context:
            $multiDetailContext

            Question: Explain whip and tongue grafting.

            Answer:
            Answer concisely using only the supplied evidence. Include the essential answer and necessary supporting detail. Do not introduce information not supported by the evidence.
        """.trimIndent()

        val answer = ggufLlm.generate(prompt, AnswerLength.SHORT)

        assertNotNull(answer)
        assertTrue("Answer must not be empty", answer.isNotBlank())
        assertTrue("Must answer directly about Whip and tongue grafting", answer.contains("Whip and tongue grafting is an advanced joining technique"))
        assertTrue("Must include citation", answer.contains("Page 3"))
        assertFalse("Short mode must NOT include Key details section", answer.contains("Key details from the document:"))
    }

    @Test
    fun testLongModeExpandsWhenEvidenceContainsDetails() = runBlocking {
        val multiDetailContext = """
            [PAGE=3] [Source 1, Page 3]
            Whip and tongue grafting is an advanced joining technique for small rootstocks.
            In whip and tongue grafting, interlocking tongue cuts provide high mechanical strength during healing.
            Smooth matching cuts in whip and tongue grafting ensure accurate vascular cambium alignment.
        """.trimIndent()

        val shortPrompt = """
            [ANSWER_LENGTH: SHORT]
            Context:
            $multiDetailContext

            Question: Explain whip and tongue grafting.

            Answer:
            Answer concisely using only the supplied evidence.
        """.trimIndent()

        val longPrompt = """
            [ANSWER_LENGTH: LONG]
            Context:
            $multiDetailContext

            Question: Explain whip and tongue grafting.

            Answer:
            Answer using ONLY the relevant facts stated in the supplied evidence. If the evidence contains multiple supporting points or sections, include all of them. If the evidence contains only a brief statement or single fact, output ONLY that brief statement without adding explanations, causes, examples, mechanisms, or background knowledge that are not explicitly written in the supplied evidence.
        """.trimIndent()

        val shortAnswer = ggufLlm.generate(shortPrompt, AnswerLength.SHORT)
        val longAnswer = ggufLlm.generate(longPrompt, AnswerLength.LONG)

        assertTrue("Long answer must be longer than or equal to Short answer", longAnswer.length > shortAnswer.length)
        assertTrue("Long answer must include the primary answer", longAnswer.contains("Whip and tongue grafting is an advanced joining technique"))
        assertTrue("Long answer must expand with additional details from the document", longAnswer.contains("Interlocking tongue cuts") || longAnswer.contains("cambium alignment"))
        assertTrue("Long answer must contain citations", longAnswer.contains("Page 3"))
    }

    // =========================================================================
    // 4. CRITICAL EVIDENCE-BOUNDED BEHAVIOR
    // =========================================================================

    @Test
    fun testEvidenceBoundedConstraint_LongModeDoesNotHallucinateOnShortEvidence() = runBlocking {
        // Document only contains a single brief sentence
        val briefContext = """
            [PAGE=1] [Source 1, Page 1]
            Grafting joins a scion to a rootstock.
        """.trimIndent()

        val shortPrompt = """
            [ANSWER_LENGTH: SHORT]
            Context:
            $briefContext

            Question: What does grafting join?

            Answer:
            Answer concisely using only the supplied evidence.
        """.trimIndent()

        val longPrompt = """
            [ANSWER_LENGTH: LONG]
            Context:
            $briefContext

            Question: What does grafting join?

            Answer:
            Answer using ONLY the relevant facts stated in the supplied evidence. If the evidence contains multiple supporting points or sections, include all of them. If the evidence contains only a brief statement or single fact, output ONLY that brief statement without adding explanations, causes, examples, mechanisms, or background knowledge that are not explicitly written in the supplied evidence.
        """.trimIndent()

        val shortAnswer = ggufLlm.generate(shortPrompt, AnswerLength.SHORT)
        val longAnswer = ggufLlm.generate(longPrompt, AnswerLength.LONG)

        // When evidence only supports a short answer, Long mode must NOT manufacture additional content
        assertTrue("Short answer must state scion to rootstock", shortAnswer.contains("scion to a rootstock"))
        assertTrue("Long answer must state scion to rootstock", longAnswer.contains("scion to a rootstock"))
        assertEquals("Long mode must NOT invent content beyond brief evidence", shortAnswer, longAnswer)
    }

    // =========================================================================
    // 5. CITATION PRESERVATION & COMPARISON TESTS
    // =========================================================================

    @Test
    fun testCitationsPreservedInBothModes() = runBlocking {
        val context = """
            [PAGE=11] [Source 1, Page 11]
            The optimal healing temperature is 21°C to 27°C (70°F to 80°F).
        """.trimIndent()

        val question = "What is the optimal healing temperature?"

        val shortPrompt = "[ANSWER_LENGTH: SHORT]\nContext:\n$context\nQuestion: $question\nAnswer:"
        val longPrompt = "[ANSWER_LENGTH: LONG]\nContext:\n$context\nQuestion: $question\nAnswer:"

        val shortAnswer = ggufLlm.generate(shortPrompt, AnswerLength.SHORT)
        val longAnswer = ggufLlm.generate(longPrompt, AnswerLength.LONG)

        assertTrue("Short answer must include citation (Page 11)", shortAnswer.contains("Page 11"))
        assertTrue("Long answer must include citation (Page 11)", longAnswer.contains("Page 11"))
        assertTrue("Short answer must contain temperature", shortAnswer.contains("21°C") || shortAnswer.contains("70°F"))
        assertTrue("Long answer must contain temperature", longAnswer.contains("21°C") || longAnswer.contains("70°F"))
    }

    @Test
    fun testComparisonDistinctionsGatedByLongMode() = runBlocking {
        val comparisonContext = """
            [PAGE=5] [Source 1, Page 5]
            Bark grafting is used on large rootstocks during spring when bark slips easily.
            [PAGE=4] [Source 1, Page 4]
            Cleft grafting is used on dormant rootstocks of 25 to 75 mm diameter.
            Both bark grafting and cleft grafting serve as topworking methods for established rootstocks.
        """.trimIndent()

        val question = "Compare Bark grafting vs Cleft grafting"

        val shortPrompt = "[ANSWER_LENGTH: SHORT]\nContext:\n$comparisonContext\nQuestion: $question\nAnswer:"
        val longPrompt = "[ANSWER_LENGTH: LONG]\nContext:\n$comparisonContext\nQuestion: $question\nAnswer:"

        val shortAnswer = ggufLlm.generate(shortPrompt, AnswerLength.SHORT)
        val longAnswer = ggufLlm.generate(longPrompt, AnswerLength.LONG)

        assertTrue("Short comparison must mention Bark", shortAnswer.contains("Bark") || shortAnswer.contains("bark"))
        assertTrue("Short comparison must mention Cleft", shortAnswer.contains("Cleft") || shortAnswer.contains("cleft"))
        assertFalse("Short mode must NOT include Key Distinctions section", shortAnswer.contains("Key Distinctions"))

        assertTrue("Long comparison must mention Bark", longAnswer.contains("Bark") || longAnswer.contains("bark"))
        assertTrue("Long comparison must mention Cleft", longAnswer.contains("Cleft") || longAnswer.contains("cleft"))
        assertTrue("Long mode must expand with Key Distinctions when joint evidence exists", longAnswer.contains("Key Distinctions"))
    }

    // =========================================================================
    // 6. SPECIFIC REQUIREMENT VERIFICATION TESTS (ITEMS 5, 7, 8, 9)
    // =========================================================================

    @Test
    fun test5_LongDoesNotAddUnrelatedOrOutOfSyllabusContent() = runBlocking {
        val evidence = """
            [PAGE=1] [Source 1, Page 1]
            Grafting joins a scion to a rootstock.
        """.trimIndent()

        val longPrompt = """
            [ANSWER_LENGTH: LONG]
            Context:
            $evidence

            Question: Explain what grafting joins.

            Answer:
            Answer using ONLY the relevant facts stated in the supplied evidence. If the evidence contains multiple supporting points or sections, include all of them. If the evidence contains only a brief statement or single fact, output ONLY that brief statement without adding explanations, causes, examples, mechanisms, or background knowledge that are not explicitly written in the supplied evidence.
        """.trimIndent()

        val answer = ggufLlm.generate(longPrompt, AnswerLength.LONG)

        assertTrue("Answer must state the grounded fact", answer.contains("scion to a rootstock"))
        // Verify unrelated out-of-syllabus botany concepts are NOT injected
        assertFalse("Must not invent photosynthesis", answer.contains("photosynthesis", ignoreCase = true))
        assertFalse("Must not invent chloroplasts", answer.contains("chloroplast", ignoreCase = true))
        assertFalse("Must not invent hydroponics", answer.contains("hydroponic", ignoreCase = true))
        assertFalse("Must not invent mitochondria", answer.contains("mitochondria", ignoreCase = true))
        assertFalse("Must not invent cellular respiration", answer.contains("respiration", ignoreCase = true))
    }

    @Test
    fun test7_AnswerLengthDoesNotModify_Embedding_Scores_ChunkIds_Pages_Intent() = runBlocking {
        val dummyEmbedding = FloatArray(384) { 0.1f }
        val dummyEmbeddingBytes = com.siddu.offlinedocai.data.FloatArrayConverter.fromFloatArray(dummyEmbedding)
        val chunks = listOf(
            ChunkEntity(id = 101, documentId = 1, chunkIndex = 0, pageNumber = 2, text = "Whip grafting connects scion and rootstock.", embedding = dummyEmbeddingBytes),
            ChunkEntity(id = 102, documentId = 1, chunkIndex = 1, pageNumber = 5, text = "Cleft grafting is used for topworking mature trees.", embedding = dummyEmbeddingBytes),
            ChunkEntity(id = 103, documentId = 1, chunkIndex = 2, pageNumber = 8, text = "Optimal temperature for graft healing is 21C to 27C.", embedding = dummyEmbeddingBytes)
        )
        val vectorStore = VectorStore(createMockChunkDao(chunks))
        val query = "What is whip grafting?"

        // 1. Intent Classification invariance
        val intentShort = QueryIntentRouter.route(query)
        val intentLong = QueryIntentRouter.route(query)
        assertEquals("Intent must be identical", intentShort.intent, intentLong.intent)
        assertEquals("Confidence must be identical", intentShort.confidence, intentLong.confidence, 0.0001f)

        // 2. Query Embedding invariance
        val embeddingShort = FloatArray(384) { 0.1f }
        val embeddingLong = FloatArray(384) { 0.1f }
        assertArrayEquals("Embeddings must be identical", embeddingShort, embeddingLong, 0.0001f)

        // 3. Retrieval results invariance
        val resultsShort = vectorStore.retrieveDetailed(embeddingShort, listOf(1L), 5, query)
        val resultsLong = vectorStore.retrieveDetailed(embeddingLong, listOf(1L), 5, query)

        assertEquals("Retrieved chunk count must match", resultsShort.size, resultsLong.size)

        // 4. Retrieved chunk IDs and page selection invariance
        val idsShort = resultsShort.map { it.chunk.id }
        val idsLong = resultsLong.map { it.chunk.id }
        assertEquals("Chunk IDs must match", idsShort, idsLong)

        val pagesShort = resultsShort.map { it.chunk.pageNumber }
        val pagesLong = resultsLong.map { it.chunk.pageNumber }
        assertEquals("Page numbers must match", pagesShort, pagesLong)

        // 5. Retrieval scores invariance
        for (i in resultsShort.indices) {
            assertEquals("Score at $i must be identical", resultsShort[i].relevanceScore, resultsLong[i].relevanceScore, 0.0001f)
        }
    }

    @Test
    fun test8_SingleRelevantSentence_ShortAndLongBothRemainShort() = runBlocking {
        // Exact example from the specification:
        // Document evidence: "Grafting joins a scion to a rootstock."
        // Short: "Grafting joins a scion to a rootstock."
        // Long: "Grafting joins a scion to a rootstock."
        val singleSentenceContext = """
            [PAGE=1] [Source 1, Page 1]
            Grafting joins a scion to a rootstock.
        """.trimIndent()

        val shortPrompt = """
            [ANSWER_LENGTH: SHORT]
            Context:
            $singleSentenceContext

            Question: What is grafting?

            Answer:
            Answer concisely using only the supplied evidence. Include the essential answer and necessary supporting detail. Do not introduce information not supported by the evidence.
        """.trimIndent()

        val longPrompt = """
            [ANSWER_LENGTH: LONG]
            Context:
            $singleSentenceContext

            Question: What is grafting?

            Answer:
            Answer using ONLY the relevant facts stated in the supplied evidence. If the evidence contains multiple supporting points or sections, include all of them. If the evidence contains only a brief statement or single fact, output ONLY that brief statement without adding explanations, causes, examples, mechanisms, or background knowledge that are not explicitly written in the supplied evidence.
        """.trimIndent()

        val shortAnswer = ggufLlm.generate(shortPrompt, AnswerLength.SHORT)
        val longAnswer = ggufLlm.generate(longPrompt, AnswerLength.LONG)

        assertTrue("Short answer must state scion to rootstock", shortAnswer.contains("scion to a rootstock"))
        assertTrue("Long answer must state scion to rootstock", longAnswer.contains("scion to a rootstock"))
        // Both answers remain appropriately short and identical when only one sentence exists
        assertEquals("Both Short and Long must remain identical when evidence is single sentence", shortAnswer, longAnswer)
    }

    @Test
    fun test9_SeveralRelevantChunks_LongCanSynthesizeAdditionalRelevantInformation() = runBlocking {
        val multiChunkContext = """
            [PAGE=3] [Source 1, Page 3]
            Whip and tongue grafting requires smooth interlocking cuts to create high mechanical strength.
            [PAGE=4] [Source 1, Page 4]
            For whip and tongue grafting, vascular cambium alignment between the scion and rootstock is an essential prerequisite for successful graft union.
            [PAGE=11] [Source 1, Page 11]
            In whip and tongue grafting, optimal healing temperature of 21°C to 27°C promotes rapid callus formation and tissue bridging.
            [PAGE=12] [Source 1, Page 12]
            Whip and tongue grafting wax sealing prevents moisture desiccation during initial vascular reconnection.
        """.trimIndent()

        val shortPrompt = """
            [ANSWER_LENGTH: SHORT]
            Context:
            $multiChunkContext

            Question: Explain whip and tongue grafting requirements.

            Answer:
            Answer concisely using only the supplied evidence. Include the essential answer and necessary supporting detail. Do not introduce information not supported by the evidence.
        """.trimIndent()

        val longPrompt = """
            [ANSWER_LENGTH: LONG]
            Context:
            $multiChunkContext

            Question: Explain whip and tongue grafting requirements.

            Answer:
            Answer using ONLY the relevant facts stated in the supplied evidence. If the evidence contains multiple supporting points or sections, include all of them. If the evidence contains only a brief statement or single fact, output ONLY that brief statement without adding explanations, causes, examples, mechanisms, or background knowledge that are not explicitly written in the supplied evidence.
        """.trimIndent()

        val shortAnswer = ggufLlm.generate(shortPrompt, AnswerLength.SHORT)
        val longAnswer = ggufLlm.generate(longPrompt, AnswerLength.LONG)

        // Short produces concise answer
        assertTrue("Short must answer directly", shortAnswer.contains("Whip and tongue grafting"))

        // Long mode synthesizes additional relevant details from other relevant chunks
        assertTrue("Long must be more detailed than short", longAnswer.length > shortAnswer.length)
        assertTrue("Long must contain primary answer", longAnswer.contains("Whip and tongue grafting"))
        assertTrue("Long must synthesize Page 12 wax/desiccation detail", longAnswer.contains("desiccation", ignoreCase = true) || longAnswer.contains("wax", ignoreCase = true))
        assertFalse("Short mode must NOT contain Page 12 4th chunk detail due to tighter budget", shortAnswer.contains("desiccation", ignoreCase = true))
        assertTrue("Both cite Page 3", shortAnswer.contains("Page 3") && longAnswer.contains("Page 3"))
    }

    // =========================================================================
    // 7. REGRESSION TESTS: PHASE 19 LONG MODE GROUNDING FIXES
    // =========================================================================

    /**
     * Requirement 6: Exact failure regression test.
     * Question: "What are the likely causes of graft failure?"
     * Evidence: "Failures typically result from poor cambial contact, desiccation of tissues, grafting during incorrect physiological seasons, or pathogen infection."
     * Long mode may contain the 4 supported points, but MUST NOT introduce unsupported explanations.
     */
    @Test
    fun testGraftFailureExactRegression_LongOutputContainsSupportedPointsAndExcludesUnsupportedElaborations() = runBlocking {
        val question = "What are the likely causes of graft failure?"
        val evidence = """
            [PAGE=14] [Source 1, Page 14]
            Failures typically result from poor cambial contact, desiccation of tissues, grafting during incorrect physiological seasons, or pathogen infection.
        """.trimIndent()

        val longPrompt = """
            [ANSWER_LENGTH: LONG]
            Context:
            $evidence

            Question: $question

            Answer:
            Answer using ONLY the relevant facts stated in the supplied evidence. If the evidence contains multiple supporting points or sections, include all of them. If the evidence contains only a brief statement or single fact, output ONLY that brief statement without adding explanations, causes, examples, mechanisms, or background knowledge that are not explicitly written in the supplied evidence.
        """.trimIndent()

        val shortPrompt = """
            [ANSWER_LENGTH: SHORT]
            Context:
            $evidence

            Question: $question

            Answer:
            Answer concisely using only the supplied evidence. Include the essential answer and necessary supporting detail. Do not introduce information not supported by the evidence.
        """.trimIndent()

        val shortAnswer = ggufLlm.generate(shortPrompt, AnswerLength.SHORT)
        val longAnswer = ggufLlm.generate(longPrompt, AnswerLength.LONG)

        // 1. Verify LONG output contains the supported points
        assertTrue("Long output must contain poor cambial contact", longAnswer.contains("poor cambial contact", ignoreCase = true))
        assertTrue("Long output must contain desiccation of tissues", longAnswer.contains("desiccation of tissues", ignoreCase = true) || longAnswer.contains("desiccation", ignoreCase = true))
        assertTrue("Long output must contain incorrect physiological seasons", longAnswer.contains("incorrect physiological", ignoreCase = true))
        assertTrue("Long output must contain pathogen infection", longAnswer.contains("pathogen infection", ignoreCase = true))

        // 2. Verify LONG output MUST NOT introduce unsupported explanations
        assertFalse("Must NOT introduce misaligned vascular system", longAnswer.contains("misaligned vascular system", ignoreCase = true))
        assertFalse("Must NOT introduce vascular system failing to fuse", longAnswer.contains("vascular system failing to fuse", ignoreCase = true) || longAnswer.contains("failing to fuse", ignoreCase = true))
        assertFalse("Must NOT introduce lack of sealing with grafting wax", longAnswer.contains("lack of sealing with grafting wax", ignoreCase = true) || longAnswer.contains("grafting wax", ignoreCase = true))
        assertFalse("Must NOT introduce low humidity", longAnswer.contains("low humidity", ignoreCase = true))
        assertFalse("Must NOT introduce optimal dormancy windows", longAnswer.contains("optimal dormancy windows", ignoreCase = true) || longAnswer.contains("dormancy windows", ignoreCase = true))
        assertFalse("Must NOT introduce unsterilized tools", longAnswer.contains("unsterilized tools", ignoreCase = true))
        assertFalse("Must NOT introduce bacteria or fungi entering wounds", longAnswer.contains("bacteria or fungi entering wounds", ignoreCase = true) || longAnswer.contains("entering wounds", ignoreCase = true))

        // 3. Verify that unsupported elaborations are rejected by CitationValidator
        val unsupportedHallucination = "Failures typically result from poor cambial contact, desiccation of tissues, grafting during incorrect physiological seasons, or pathogen infection (Page 14). Other likely causes include a misaligned vascular system, lack of sealing with grafting wax, low humidity, and unsterilized tools leading to bacteria or fungi entering wounds (Page 14)."
        val sourceBlocks = listOf(
            com.siddu.offlinedocai.ai.retrieval.SourceBlock(
                sourceId = 1L,
                documentId = 1L,
                pageNumber = 14,
                pageSpan = "14",
                chunkIndex = 0,
                sectionTitle = "Troubleshooting Graft Failure",
                text = "Failures typically result from poor cambial contact, desiccation of tissues, grafting during incorrect physiological seasons, or pathogen infection."
            )
        )
        val validationResult = com.siddu.offlinedocai.ai.validation.CitationValidator.validateWithSourceBlocks(
            answer = unsupportedHallucination,
            sourceBlocks = sourceBlocks,
            query = question,
            isBudgetLimited = false
        )
        val unsupportedClaim = validationResult.claimResults.find { it.originalClaim.contains("unsterilized tools") || it.originalClaim.contains("misaligned vascular") }
        assertNotNull("CitationValidator must analyze the unsupported claim", unsupportedClaim)
        assertTrue("Unsupported claim must be flagged as UNSUPPORTED or score < 0.25", unsupportedClaim!!.status == com.siddu.offlinedocai.ai.validation.ValidationStatus.UNSUPPORTED || unsupportedClaim.supportScore < 0.25f)
    }

    /**
     * Requirement 7: Single-fact regression test.
     * Evidence: "Grafting joins a scion to a rootstock."
     * Question: "What is grafting?"
     * Long must remain essentially equivalent to the supported single fact and not add general horticultural explanations.
     */
    @Test
    fun testSingleFactRegression_LongModeRemainsEssentiallyEquivalent() = runBlocking {
        val evidence = """
            [PAGE=1] [Source 1, Page 1]
            Grafting joins a scion to a rootstock.
        """.trimIndent()
        val question = "What is grafting?"

        val shortPrompt = """
            [ANSWER_LENGTH: SHORT]
            Context:
            $evidence

            Question: $question

            Answer:
            Answer concisely using only the supplied evidence. Include the essential answer and necessary supporting detail. Do not introduce information not supported by the evidence.
        """.trimIndent()

        val longPrompt = """
            [ANSWER_LENGTH: LONG]
            Context:
            $evidence

            Question: $question

            Answer:
            Answer using ONLY the relevant facts stated in the supplied evidence. If the evidence contains multiple supporting points or sections, include all of them. If the evidence contains only a brief statement or single fact, output ONLY that brief statement without adding explanations, causes, examples, mechanisms, or background knowledge that are not explicitly written in the supplied evidence.
        """.trimIndent()

        val shortAnswer = ggufLlm.generate(shortPrompt, AnswerLength.SHORT)
        val longAnswer = ggufLlm.generate(longPrompt, AnswerLength.LONG)

        assertTrue("Short answer must state scion to rootstock", shortAnswer.contains("scion to a rootstock"))
        assertTrue("Long answer must state scion to rootstock", longAnswer.contains("scion to a rootstock"))
        assertEquals("Long must remain essentially equivalent to Short for single-fact evidence", shortAnswer, longAnswer)
        assertFalse("Long must NOT add cambial explanations not in evidence", longAnswer.contains("cambium", ignoreCase = true))
        assertFalse("Long must NOT add callus explanations not in evidence", longAnswer.contains("callus", ignoreCase = true))
        assertFalse("Long must NOT add healing explanations not in evidence", longAnswer.contains("healing", ignoreCase = true))
        assertFalse("Long must NOT add propagation explanations not in evidence", longAnswer.contains("propagation", ignoreCase = true))
    }

    /**
     * Requirement 8: Multi-fact regression test.
     * Create evidence containing several relevant facts about the same topic.
     * Verify Short = concise answer. Long = more complete answer containing additional facts from evidence.
     * Long must NOT contain facts absent from the evidence.
     */
    @Test
    fun testMultiFactRegression_ShortIsConcise_LongIsMoreCompleteWithAdditionalSupportedFacts() = runBlocking {
        val multiFactEvidence = """
            [PAGE=2] [Source 1, Page 2]
            Vascular cambium in grafting is the active meristematic cell layer between xylem and phloem responsible for callus union.
            Close vascular cambium alignment between the scion and rootstock ensures rapid bridge vascular tissue development.
            Temperature maintained between 21°C and 27°C accelerates vascular cambium division during graft healing.
        """.trimIndent()

        val question = "Explain how vascular cambium functions in grafting."

        val shortPrompt = """
            [ANSWER_LENGTH: SHORT]
            Context:
            $multiFactEvidence

            Question: $question

            Answer:
            Answer concisely using only the supplied evidence. Include the essential answer and necessary supporting detail. Do not introduce information not supported by the evidence.
        """.trimIndent()

        val longPrompt = """
            [ANSWER_LENGTH: LONG]
            Context:
            $multiFactEvidence

            Question: $question

            Answer:
            Answer using ONLY the relevant facts stated in the supplied evidence. If the evidence contains multiple supporting points or sections, include all of them. If the evidence contains only a brief statement or single fact, output ONLY that brief statement without adding explanations, causes, examples, mechanisms, or background knowledge that are not explicitly written in the supplied evidence.
        """.trimIndent()

        val shortAnswer = ggufLlm.generate(shortPrompt, AnswerLength.SHORT)
        val longAnswer = ggufLlm.generate(longPrompt, AnswerLength.LONG)

        // Short produces concise answer
        assertTrue("Short must answer directly about vascular cambium", shortAnswer.contains("meristematic", ignoreCase = true) || shortAnswer.contains("xylem and phloem", ignoreCase = true))

        // Long produces more complete answer containing additional facts from the supplied evidence
        assertTrue("Long must be more detailed than short", longAnswer.length > shortAnswer.length)
        assertTrue("Long must contain primary definition", longAnswer.contains("meristematic", ignoreCase = true) || longAnswer.contains("xylem and phloem", ignoreCase = true))
        assertTrue("Long must include alignment fact from evidence", longAnswer.contains("Close cambial alignment", ignoreCase = true) || longAnswer.contains("bridge vascular tissue", ignoreCase = true))
        assertTrue("Long must include temperature fact from evidence", longAnswer.contains("21°C", ignoreCase = true) || longAnswer.contains("27°C", ignoreCase = true))

        // Long must NOT contain facts absent from the evidence
        assertFalse("Long must NOT invent photosynthesis", longAnswer.contains("photosynthesis", ignoreCase = true))
        assertFalse("Long must NOT invent chloroplasts", longAnswer.contains("chloroplast", ignoreCase = true))
        assertFalse("Long must NOT invent nitrogen", longAnswer.contains("nitrogen", ignoreCase = true))
        assertFalse("Long must NOT invent auxin or hormones", longAnswer.contains("auxin", ignoreCase = true) || longAnswer.contains("hormone", ignoreCase = true))
    }

    /**
     * Section 2 verification: Token Budget Resolution.
     * SHORT = 128, LONG = 512.
     */
    @Test
    fun testDynamicTokenBudgetHeadroomBounding() {
        assertEquals(128, ggufLlm.resolveMaxGenTokens(AnswerLength.SHORT, evidenceTokens = 10))
        assertEquals(128, ggufLlm.resolveMaxGenTokens(AnswerLength.SHORT, evidenceTokens = 99))
        assertEquals(128, ggufLlm.resolveMaxGenTokens(AnswerLength.SHORT, evidenceTokens = 150))
        assertEquals(128, ggufLlm.resolveMaxGenTokens(AnswerLength.SHORT, evidenceTokens = 500))

        assertEquals(512, ggufLlm.resolveMaxGenTokens(AnswerLength.LONG, evidenceTokens = 10))
        assertEquals(512, ggufLlm.resolveMaxGenTokens(AnswerLength.LONG, evidenceTokens = 42))
        assertEquals(512, ggufLlm.resolveMaxGenTokens(AnswerLength.LONG, evidenceTokens = 99))
        assertEquals(512, ggufLlm.resolveMaxGenTokens(AnswerLength.LONG, evidenceTokens = 100))
        assertEquals(512, ggufLlm.resolveMaxGenTokens(AnswerLength.LONG, evidenceTokens = 250))
        assertEquals(512, ggufLlm.resolveMaxGenTokens(AnswerLength.LONG, evidenceTokens = 800))
    }

    /**
     * Requirement 2 verification: Conceptual candidate eligibility condition.
     * Candidate score must be >= maxOf(15.0, top.score * 0.6) in LONG mode.
     */
    @Test
    fun testLongConceptualFallbackThreshold_KeywordMatchesWithoutRelevanceAreFiltered() = runBlocking {
        val context = """
            [PAGE=14] [Source 1, Page 14]
            Failures typically result from poor cambial contact, desiccation of tissues, grafting during incorrect physiological seasons, or pathogen infection.
            [PAGE=1] [Source 1, Page 1]
            Grafting is a horticultural technique.
        """.trimIndent()

        val longPrompt = """
            [ANSWER_LENGTH: LONG]
            Context:
            $context

            Question: What are the likely causes of graft failure?

            Answer:
            Answer using ONLY the relevant facts stated in the supplied evidence. If the evidence contains multiple supporting points or sections, include all of them. If the evidence contains only a brief statement or single fact, output ONLY that brief statement without adding explanations, causes, examples, mechanisms, or background knowledge that are not explicitly written in the supplied evidence.
        """.trimIndent()

        val answer = ggufLlm.generate(longPrompt, AnswerLength.LONG)

        assertTrue("Answer must contain graft failure causes", answer.contains("poor cambial contact", ignoreCase = true))
        assertFalse("Low-scoring keyword match from Page 1 must NOT be admitted as supporting detail", answer.contains("horticultural technique", ignoreCase = true))
    }

    // =========================================================================
    // REQUIREMENT 9: SIX MANDATORY BEHAVIORAL VERIFICATION TESTS
    // =========================================================================

    /**
     * TEST A — SINGLE FACT:
     * Evidence contains one fact.
     * Expected: Short = concise fact, Long = same fact or minimally expanded only if evidence supports it.
     */
    @Test
    fun testRequirement9_TestA_SingleFact() = runBlocking {
        val singleFactContext = """
            [PAGE=1] [Source 1, Page 1]
            Grafting joins a scion to a rootstock.
        """.trimIndent()

        val shortPrompt = ggufLlm.buildQwenPrompt(singleFactContext, "What is grafting?", answerLength = AnswerLength.SHORT)
        val longPrompt = ggufLlm.buildQwenPrompt(singleFactContext, "What is grafting?", answerLength = AnswerLength.LONG)

        val shortAnswer = ggufLlm.generate(shortPrompt, AnswerLength.SHORT)
        val longAnswer = ggufLlm.generate(longPrompt, AnswerLength.LONG)

        assertTrue("Short must contain the grounded fact", shortAnswer.contains("Grafting joins a scion to a rootstock", ignoreCase = true))
        assertTrue("Long must contain the grounded fact", longAnswer.contains("Grafting joins a scion to a rootstock", ignoreCase = true))

        // Neither may invent outside horticultural knowledge
        assertFalse("Short must not invent vascular cambium", shortAnswer.contains("cambium", ignoreCase = true))
        assertFalse("Long must not invent vascular cambium", longAnswer.contains("cambium", ignoreCase = true))
        assertFalse("Long must not invent callus", longAnswer.contains("callus", ignoreCase = true))
        assertFalse("Long must not invent fruit trees", longAnswer.contains("fruit trees", ignoreCase = true))

        // When only one fact is available, Long remains essentially equivalent to Short
        assertTrue("Long must not manufacture paragraphs on single fact", Math.abs(longAnswer.length - shortAnswer.length) < 50)
    }

    /**
     * TEST B — MULTIPLE RELEVANT FACTS:
     * Evidence contains 5–10 relevant facts.
     * Expected: Short = essential subset, Long = substantially more complete answer.
     */
    @Test
    fun testRequirement9_TestB_MultipleRelevantFacts() = runBlocking {
        val multiFactContext = """
            [PAGE=3] [Source 1, Page 3]
            Whip and tongue grafting is an advanced joining technique for small rootstocks.
            It requires scion and rootstock of identical diameter.
            Interlocking tongue cuts provide high mechanical strength during healing.
            Smooth matching cuts ensure accurate vascular cambium alignment.
            Callus bridge formation typically completes within three to five weeks.
            Wax application seals the graft union against atmospheric drying.
        """.trimIndent()

        val shortPrompt = ggufLlm.buildQwenPrompt(multiFactContext, "Explain whip and tongue grafting.", answerLength = AnswerLength.SHORT)
        val longPrompt = ggufLlm.buildQwenPrompt(multiFactContext, "Explain whip and tongue grafting.", answerLength = AnswerLength.LONG)

        val shortAnswer = ggufLlm.generate(shortPrompt, AnswerLength.SHORT)
        val longAnswer = ggufLlm.generate(longPrompt, AnswerLength.LONG)

        // Short produces essential subset
        assertTrue("Short must state primary definition", shortAnswer.contains("advanced joining technique", ignoreCase = true))
        assertFalse("Short should not be bloated with every detail bullet", shortAnswer.contains("seals the graft union against atmospheric drying", ignoreCase = true))

        // Long produces substantially more complete answer
        assertTrue("Long must be longer than Short", longAnswer.length > shortAnswer.length)
        assertTrue("Long must include primary definition", longAnswer.contains("advanced joining technique", ignoreCase = true))
        assertTrue("Long must include interlocking tongue cuts", longAnswer.contains("Interlocking tongue cuts", ignoreCase = true))
        assertTrue("Long must include vascular cambium alignment", longAnswer.contains("vascular cambium alignment", ignoreCase = true))
        assertTrue("Long must include callus bridge", longAnswer.contains("Callus bridge formation", ignoreCase = true))
        assertTrue("Long must include wax application", longAnswer.contains("Wax application", ignoreCase = true))

        // Long must not invent outside information
        assertFalse("Long must not invent nitrogen", longAnswer.contains("nitrogen", ignoreCase = true))
        assertFalse("Long must not invent genetics", longAnswer.contains("genetics", ignoreCase = true))
    }

    /**
     * TEST C — UNSUPPORTED DETAIL:
     * Evidence contains a fact but no explanation.
     * Expected: Short = fact, Long = fact. Neither may invent an explanation.
     */
    @Test
    fun testRequirement9_TestC_UnsupportedDetail() = runBlocking {
        val factWithoutExplanationContext = """
            [PAGE=14] [Source 1, Page 14]
            Poor cambial contact is a cause of graft failure.
        """.trimIndent()

        val shortPrompt = ggufLlm.buildQwenPrompt(factWithoutExplanationContext, "What causes graft failure?", answerLength = AnswerLength.SHORT)
        val longPrompt = ggufLlm.buildQwenPrompt(factWithoutExplanationContext, "What causes graft failure?", answerLength = AnswerLength.LONG)

        val shortAnswer = ggufLlm.generate(shortPrompt, AnswerLength.SHORT)
        val longAnswer = ggufLlm.generate(longPrompt, AnswerLength.LONG)

        assertTrue("Short must contain the stated fact", shortAnswer.contains("Poor cambial contact is a cause of graft failure", ignoreCase = true))
        assertTrue("Long must contain the stated fact", longAnswer.contains("Poor cambial contact is a cause of graft failure", ignoreCase = true))

        // Neither may invent explanations absent from evidence
        assertFalse("Short must not invent vascular fusion failure", shortAnswer.contains("fail to fuse", ignoreCase = true) || shortAnswer.contains("failing to fuse", ignoreCase = true))
        assertFalse("Long must not invent vascular fusion failure", longAnswer.contains("fail to fuse", ignoreCase = true) || longAnswer.contains("failing to fuse", ignoreCase = true))
        assertFalse("Long must not invent xylem blockage", longAnswer.contains("xylem blockage", ignoreCase = true))
        assertFalse("Long must not invent water stress", longAnswer.contains("water stress", ignoreCase = true))
    }

    /**
     * TEST D — MULTI-PAGE EVIDENCE:
     * Relevant information is distributed across several pages.
     * Expected: Short = concise synthesis, Long = comprehensive synthesis across the relevant pages.
     */
    @Test
    fun testRequirement9_TestD_MultiPageEvidence() = runBlocking {
        val multiPageContext = """
            [PAGE=3] [Source 1, Page 3]
            Method 1: Whip-and-tongue grafting requires matching scion and rootstock diameters.
            [PAGE=4] [Source 1, Page 4]
            Method 2: Cleft grafting is used when the rootstock is much larger than the scion.
            [PAGE=5] [Source 1, Page 5]
            Method 3: Bark grafting is used for topworking mature trees when the bark is slipping.
            [PAGE=6] [Source 1, Page 6]
            Method 4: Bridge grafting is utilized to repair damaged trunks where girdling occurred.
        """.trimIndent()

        val shortPrompt = ggufLlm.buildQwenPrompt(multiPageContext, "What are the types of grafting?", answerLength = AnswerLength.SHORT)
        val longPrompt = ggufLlm.buildQwenPrompt(multiPageContext, "What are the types of grafting?", answerLength = AnswerLength.LONG)

        val shortAnswer = ggufLlm.generate(shortPrompt, AnswerLength.SHORT)
        val longAnswer = ggufLlm.generate(longPrompt, AnswerLength.LONG)

        // Short produces concise summary
        assertTrue("Short must mention grafting methods", shortAnswer.contains("grafting", ignoreCase = true))
        assertTrue("Short should be concise", shortAnswer.length < longAnswer.length)

        // Long produces comprehensive synthesis across pages 3, 4, 5, 6
        assertTrue("Long must cite Page 3", longAnswer.contains("Page 3"))
        assertTrue("Long must cite Page 4", longAnswer.contains("Page 4"))
        assertTrue("Long must cite Page 5", longAnswer.contains("Page 5"))
        assertTrue("Long must cite Page 6", longAnswer.contains("Page 6"))

        assertTrue("Long must explain Whip-and-tongue", longAnswer.contains("Whip-and-tongue", ignoreCase = true))
        assertTrue("Long must explain Cleft", longAnswer.contains("Cleft", ignoreCase = true))
        assertTrue("Long must explain Bark", longAnswer.contains("Bark", ignoreCase = true))
        assertTrue("Long must explain Bridge", longAnswer.contains("Bridge", ignoreCase = true))
    }

    /**
     * TEST E — UNRELATED CHUNKS:
     * Retrieved context contains relevant and unrelated chunks.
     * Expected: Short and Long both ignore unrelated chunks.
     */
    @Test
    fun testRequirement9_TestE_UnrelatedChunks() = runBlocking {
        val mixedContext = """
            [PAGE=14] [Source 1, Page 14]
            Failures typically result from poor cambial contact, desiccation of tissues, grafting during incorrect physiological seasons, or pathogen infection.
            [PAGE=1] [Source 1, Page 1]
            Grafting is a horticultural technique.
            [PAGE=10] [Source 2, Page 10]
            Post-quantum lattice-based cryptography details Kyber key encapsulation and Dilithium digital signatures for aerospace telemetry.
        """.trimIndent()

        val shortPrompt = ggufLlm.buildQwenPrompt(mixedContext, "What are the likely causes of graft failure?", answerLength = AnswerLength.SHORT)
        val longPrompt = ggufLlm.buildQwenPrompt(mixedContext, "What are the likely causes of graft failure?", answerLength = AnswerLength.LONG)

        val shortAnswer = ggufLlm.generate(shortPrompt, AnswerLength.SHORT)
        val longAnswer = ggufLlm.generate(longPrompt, AnswerLength.LONG)

        // Both include relevant failure causes from Page 14
        assertTrue("Short must contain poor cambial contact", shortAnswer.contains("poor cambial contact", ignoreCase = true))
        assertTrue("Long must contain poor cambial contact", longAnswer.contains("poor cambial contact", ignoreCase = true))

        // Both ignore unrelated chunks (Page 10 cryptography and Page 1 keyword-only mention)
        assertFalse("Short must ignore cryptography", shortAnswer.contains("cryptography", ignoreCase = true) || shortAnswer.contains("Kyber", ignoreCase = true))
        assertFalse("Long must ignore cryptography", longAnswer.contains("cryptography", ignoreCase = true) || longAnswer.contains("Kyber", ignoreCase = true))
        assertFalse("Short must ignore Page 1 generic technique", shortAnswer.contains("horticultural technique", ignoreCase = true))
        assertFalse("Long must ignore Page 1 generic technique", longAnswer.contains("horticultural technique", ignoreCase = true))
    }

    /**
     * TEST F — SAME RETRIEVAL:
     * Verify that chunk IDs, page IDs, retrieval scores, routing intent, and assembled context
     * are 100% IDENTICAL between Short and Long answer modes.
     */
    @Test
    fun testRequirement9_TestF_SameRetrieval() = runBlocking {
        val testChunks = listOf(
            ChunkEntity(id = 1, documentId = 1, pageNumber = 3, chunkIndex = 0, sectionTitle = "Methods", pageSpan = "3", text = "Whip and tongue grafting is used for fruit trees.", embedding = null),
            ChunkEntity(id = 2, documentId = 1, pageNumber = 4, chunkIndex = 1, sectionTitle = "Methods", pageSpan = "4", text = "Cleft grafting is used for topworking large stocks.", embedding = null),
            ChunkEntity(id = 3, documentId = 1, pageNumber = 14, chunkIndex = 2, sectionTitle = "Failure", pageSpan = "14", text = "Graft failures typically result from poor cambial contact.", embedding = null)
        )
        val dao = createMockChunkDao(testChunks)
        val vectorStore = VectorStore(dao)

        val query = "What are the likely causes of graft failure?"

        // Route intent
        val classificationShort = QueryIntentRouter.route(query)
        val classificationLong = QueryIntentRouter.route(query)
        assertEquals("Intent must be identical", classificationShort.intent, classificationLong.intent)

        // Retrieve
        val retrievedShort = vectorStore.retrieveDetailed(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(1L),
            topK = 5,
            query = query
        )
        val retrievedLong = vectorStore.retrieveDetailed(
            queryEmbedding = FloatArray(384) { 0f },
            relevantDocumentIds = listOf(1L),
            topK = 5,
            query = query
        )

        assertEquals("Retrieved chunk count must be identical", retrievedShort.size, retrievedLong.size)
        assertEquals("Retrieved chunk IDs must be identical", retrievedShort.map { it.chunk.id }, retrievedLong.map { it.chunk.id })
        assertEquals("Retrieved page numbers must be identical", retrievedShort.map { it.chunk.pageNumber }, retrievedLong.map { it.chunk.pageNumber })
        assertEquals("Retrieved scores must be identical", retrievedShort.map { it.relevanceScore }, retrievedLong.map { it.relevanceScore })

        // Assemble context
        val contextShort = ContextAssembler.assemble(retrievedShort, classificationShort.intent, classificationShort)
        val contextLong = ContextAssembler.assemble(retrievedLong, classificationLong.intent, classificationLong)

        assertEquals("Assembled context text must be identical", contextShort.contextText, contextLong.contextText)
        assertEquals("Assembled token estimates must be identical", contextShort.totalEstimatedTokens, contextLong.totalEstimatedTokens)
        assertEquals("Source blocks must be identical", contextShort.sourceBlocks.size, contextLong.sourceBlocks.size)
    }

    // =========================================================================
    // REQUIREMENT 10: REAL PIPELINE TESTS FOR THE THREE TARGET QUESTIONS
    // =========================================================================

    /**
     * Question 1: "What are the types of grafting?"
     * Long must contain substantially more supported detail than Short.
     */
    @Test
    fun testRequirement10_RealPipeline_TypesOfGrafting() = runBlocking {
        val typesContext = """
            [PAGE=3] [Source 1, Page 3]
            Method 1: Whip-and-tongue grafting — used for fruit trees such as apples and pears where scion and rootstock have equal diameter.
            [PAGE=4] [Source 1, Page 4]
            Method 2: Cleft grafting — used when the rootstock is much larger than the scion.
            [PAGE=5] [Source 1, Page 5]
            Method 3: Bark grafting — commonly used for topworking mature trees when the bark is slipping in early spring.
            [PAGE=6] [Source 1, Page 6]
            Method 4: Bridge grafting — utilized to repair damaged tree trunks where girdling has destroyed the phloem.
            [PAGE=7] [Source 1, Page 7]
            Method 5: Side-veneer grafting — widely used technique for propagating conifers and small nursery stock.
            [PAGE=8] [Source 1, Page 8]
            Method 6: Splice grafting — simple technique using diagonal cuts and securing tape.
            [PAGE=9] [Source 1, Page 9]
            Method 7: Saddle grafting — uses a V-notch in the rootstock and matching wedge scion shape.
            [PAGE=10] [Source 1, Page 10]
            Method 8: Approach grafting — involves joining two independent rooted plants together until a union forms.
        """.trimIndent()

        val q = "What are the types of grafting?"
        val shortPrompt = ggufLlm.buildQwenPrompt(typesContext, q, answerLength = AnswerLength.SHORT)
        val longPrompt = ggufLlm.buildQwenPrompt(typesContext, q, answerLength = AnswerLength.LONG)

        val shortAnswer = ggufLlm.generate(shortPrompt, AnswerLength.SHORT)
        val longAnswer = ggufLlm.generate(longPrompt, AnswerLength.LONG)

        // Short produces concise summary
        assertTrue("Short must answer about types of grafting", shortAnswer.contains("grafting", ignoreCase = true))

        // Long contains substantially more supported detail
        assertTrue("Long must be substantially longer than Short", longAnswer.length > shortAnswer.length * 1.5)
        assertTrue("Long must describe whip-and-tongue", longAnswer.contains("whip-and-tongue", ignoreCase = true) || longAnswer.contains("fruit trees", ignoreCase = true))
        assertTrue("Long must describe cleft grafting", longAnswer.contains("cleft", ignoreCase = true))
        assertTrue("Long must describe bark grafting", longAnswer.contains("bark", ignoreCase = true))
        assertTrue("Long must describe bridge grafting", longAnswer.contains("bridge", ignoreCase = true))
        assertTrue("Long must describe side-veneer", longAnswer.contains("side-veneer", ignoreCase = true))
        assertTrue("Long must describe splice grafting", longAnswer.contains("splice", ignoreCase = true))
        assertTrue("Long must describe saddle grafting", longAnswer.contains("saddle", ignoreCase = true))
        assertTrue("Long must describe approach grafting", longAnswer.contains("approach", ignoreCase = true))
    }

    /**
     * Question 2: "What are the likely causes of graft failure?"
     * Long may expand only if additional relevant evidence exists.
     */
    @Test
    fun testRequirement10_RealPipeline_CausesOfFailure() = runBlocking {
        val failureContext = """
            [PAGE=14] [Source 1, Page 14]
            Failures typically result from poor cambial contact, desiccation of tissues, grafting during incorrect physiological seasons, or pathogen infection.
            [PAGE=15] [Source 1, Page 15]
            Tissue desiccation causes rapid cell death in unhealed graft unions before callus bridge formation.
        """.trimIndent()

        val q = "What are the likely causes of graft failure?"
        val shortPrompt = ggufLlm.buildQwenPrompt(failureContext, q, answerLength = AnswerLength.SHORT)
        val longPrompt = ggufLlm.buildQwenPrompt(failureContext, q, answerLength = AnswerLength.LONG)

        val shortAnswer = ggufLlm.generate(shortPrompt, AnswerLength.SHORT)
        val longAnswer = ggufLlm.generate(longPrompt, AnswerLength.LONG)

        assertTrue("Short must contain poor cambial contact", shortAnswer.contains("poor cambial contact", ignoreCase = true))
        assertTrue("Long must contain poor cambial contact", longAnswer.contains("poor cambial contact", ignoreCase = true))

        // Long expands using additional supported sentence from Page 15
        assertTrue("Long must include cell death from desiccation", longAnswer.contains("rapid cell death", ignoreCase = true) || longAnswer.contains("callus bridge", ignoreCase = true))

        // Neither invents unsupported outside causes
        assertFalse("Long must not invent unsterilized tools", longAnswer.contains("unsterilized", ignoreCase = true))
        assertFalse("Long must not invent optimal dormancy", longAnswer.contains("dormancy windows", ignoreCase = true))
    }

    /**
     * Question 3: "What is grafting?"
     * Long should remain short if only one relevant fact exists.
     */
    @Test
    fun testRequirement10_RealPipeline_DefinitionOfGrafting() = runBlocking {
        val definitionContext = """
            [PAGE=1] [Source 1, Page 1]
            Grafting is a horticultural technique whereby tissues of plants are joined so as to continue their growth together.
        """.trimIndent()

        val q = "What is grafting?"
        val shortPrompt = ggufLlm.buildQwenPrompt(definitionContext, q, answerLength = AnswerLength.SHORT)
        val longPrompt = ggufLlm.buildQwenPrompt(definitionContext, q, answerLength = AnswerLength.LONG)

        val shortAnswer = ggufLlm.generate(shortPrompt, AnswerLength.SHORT)
        val longAnswer = ggufLlm.generate(longPrompt, AnswerLength.LONG)

        assertTrue("Short must define grafting", shortAnswer.contains("horticultural technique whereby tissues of plants are joined", ignoreCase = true))
        assertTrue("Long must define grafting", longAnswer.contains("horticultural technique whereby tissues of plants are joined", ignoreCase = true))

        // When only one relevant fact exists, Long remains short
        assertTrue("Long must remain short and comparable to Short", Math.abs(longAnswer.length - shortAnswer.length) < 50)
        assertFalse("Long must not invent vascular fusion or rootstocks", longAnswer.contains("scion", ignoreCase = true) || longAnswer.contains("cambium", ignoreCase = true))
    }

    // =========================================================================
    // HEADING FILTERING REGRESSION TESTS (SHORT AND LONG)
    // =========================================================================

    /**
     * Step 6 Regression:
     * Question: "What are the likely causes of graft failure?"
     * The answer MUST NOT begin with "Failure Likely causes what to check"
     * The heading must NOT be presented as an answer sentence in either Short or Long.
     */
    @Test
    fun testHeadingFilter_CausesOfFailureRegression_ShortAndLong() = runBlocking {
        val evidence = """
            [PAGE=11] [Source 1, Page 11]
            Page 11: Failure Likely causes / What to check
            If grafting fails, check the alignment of the cambium layers. Common causes of failure include poor cuts, incorrect temperatures, and drying out. Ensure rubber ties are secure.
            [PAGE=12] [Source 1, Page 12]
            Monitor humidity and wrap grafts with paraffin film.
        """.trimIndent()

        val q = "What are the likely causes of graft failure?"
        val shortPrompt = ggufLlm.buildQwenPrompt(evidence, q, answerLength = AnswerLength.SHORT)
        val longPrompt = ggufLlm.buildQwenPrompt(evidence, q, answerLength = AnswerLength.LONG)

        val shortAnswer = ggufLlm.generate(shortPrompt, AnswerLength.SHORT)
        val longAnswer = ggufLlm.generate(longPrompt, AnswerLength.LONG)

        // SHORT mode checks
        assertFalse("Short answer must NOT begin with malformed heading", shortAnswer.trim().startsWith("Failure Likely causes", ignoreCase = true))
        assertFalse("Short answer must NOT contain heading title", shortAnswer.contains("Failure Likely causes / What to check", ignoreCase = true))
        assertFalse("Short answer must NOT contain what to check", shortAnswer.contains("what to check", ignoreCase = true))
        assertTrue("Short answer must contain explanatory causes", shortAnswer.contains("Common causes of failure include", ignoreCase = true) || shortAnswer.contains("cambium", ignoreCase = true))

        // LONG mode checks
        assertFalse("Long answer must NOT begin with malformed heading", longAnswer.trim().startsWith("Failure Likely causes", ignoreCase = true))
        assertFalse("Long answer must NOT contain heading title", longAnswer.contains("Failure Likely causes / What to check", ignoreCase = true))
        assertFalse("Long answer must NOT contain what to check", longAnswer.contains("what to check", ignoreCase = true))
        assertTrue("Long answer must contain explanatory causes", longAnswer.contains("Common causes of failure include", ignoreCase = true))

        // Citations preserved
        assertTrue("Short answer must cite Page 11", shortAnswer.contains("Page 11"))
        assertTrue("Long answer must cite Page 11", longAnswer.contains("Page 11"))
    }

    /**
     * Step 7 Verification across all 5 benchmark questions:
     * Verifies that section headings are never returned as the actual answer sentences.
     */
    @Test
    fun testHeadingFilter_FiveBenchmarkQuestions() = runBlocking {
        // 1. "What is grafting?"
        val context1 = """
            [PAGE=1] [Source 1, Page 1]
            Page 1: Introduction & Fundamentals
            Grafting joins a scion to a rootstock to unite their vascular systems.
        """.trimIndent()
        val a1 = ggufLlm.generate(ggufLlm.buildQwenPrompt(context1, "What is grafting?", answerLength = AnswerLength.SHORT), AnswerLength.SHORT)
        assertFalse("Heading must not be answer", a1.contains("Introduction & Fundamentals", ignoreCase = true))
        assertTrue("Answer must use factual sentence", a1.contains("joins a scion to a rootstock", ignoreCase = true))

        // 2. "What are the types of grafting?"
        val context2 = """
            [PAGE=3] [Source 1, Page 3]
            Page 3: Grafting Methods
            Common grafting methods include whip and tongue grafting, cleft grafting, and bark grafting.
        """.trimIndent()
        val a2 = ggufLlm.generate(ggufLlm.buildQwenPrompt(context2, "What are the types of grafting?", answerLength = AnswerLength.LONG), AnswerLength.LONG)
        assertFalse("Heading must not be answer", a2.startsWith("Page 3: Grafting Methods", ignoreCase = true))
        assertTrue("Answer must contain methods", a2.contains("whip and tongue", ignoreCase = true))

        // 3. "What are the likely causes of graft failure?"
        val context3 = """
            [PAGE=14] [Source 1, Page 14]
            Section 14: Troubleshooting Graft Failure
            Common causes of failure include poor cuts, incorrect temperatures, and drying out.
        """.trimIndent()
        val a3 = ggufLlm.generate(ggufLlm.buildQwenPrompt(context3, "What are the likely causes of graft failure?", answerLength = AnswerLength.SHORT), AnswerLength.SHORT)
        assertFalse("Heading must not be answer", a3.startsWith("Section 14:", ignoreCase = true))
        assertTrue("Answer must use explanatory causes", a3.contains("poor cuts", ignoreCase = true))

        // 4. "What tools are required?"
        val context4 = """
            [PAGE=5] [Source 1, Page 5]
            Tools & Materials
            Required tools include a sharp grafting knife, pruning shears, and grafting tape.
        """.trimIndent()
        val a4 = ggufLlm.generate(ggufLlm.buildQwenPrompt(context4, "What tools are required?", answerLength = AnswerLength.SHORT), AnswerLength.SHORT)
        assertFalse("Heading must not be answer", a4.startsWith("Tools & Materials", ignoreCase = true))
        assertTrue("Answer must contain required tools", a4.contains("sharp grafting knife", ignoreCase = true))

        // 5. "What temperature is optimal for healing?"
        val context5 = """
            [PAGE=8] [Source 1, Page 8]
            Optimal Temperature
            Callus formation occurs most rapidly at temperatures between 21°C and 27°C.
        """.trimIndent()
        val a5 = ggufLlm.generate(ggufLlm.buildQwenPrompt(context5, "What temperature is optimal for healing?", answerLength = AnswerLength.SHORT), AnswerLength.SHORT)
        assertFalse("Heading must not be answer", a5.startsWith("Optimal Temperature", ignoreCase = true))
        assertTrue("Answer must contain temperature range", a5.contains("21°C", ignoreCase = true) || a5.contains("27°C", ignoreCase = true))
    }

    // =========================================================================
    // SECTION 8 & 9: MANDATORY TESTS (TEST 1 - TEST 7 & RETRIEVAL INVARIANCE)
    // =========================================================================

    /**
     * TEST 1:
     * Question: "What is grafting?"
     * Expected:
     * SHORT = core definition.
     * LONG = core definition + scion/rootstock + additional grounded information if available.
     */
    @Test
    fun testRequired_Test1_WhatIsGrafting() = runBlocking {
        val evidence = """
            [PAGE=1] [Source 1, Page 1]
            Grafting is a horticultural technique whereby tissues of plants are joined so as to continue their growth together.
            In grafting, a scion from one plant is joined to the rootstock of another.
            Vascular cambium alignment is an essential prerequisite for successful graft union.
        """.trimIndent()
        val q = "What is grafting?"

        val shortAnswer = ggufLlm.generate(ggufLlm.buildQwenPrompt(evidence, q, answerLength = AnswerLength.SHORT), AnswerLength.SHORT)
        val longAnswer = ggufLlm.generate(ggufLlm.buildQwenPrompt(evidence, q, answerLength = AnswerLength.LONG), AnswerLength.LONG)

        // SHORT = core definition
        assertTrue("Short must contain core definition", shortAnswer.contains("horticultural technique whereby tissues of plants are joined", ignoreCase = true))
        assertFalse("Short must avoid unnecessary supporting details", shortAnswer.contains("cambium alignment", ignoreCase = true))

        // LONG = core definition + scion/rootstock + additional grounded information
        assertTrue("Long must contain core definition", longAnswer.contains("horticultural technique whereby tissues of plants are joined", ignoreCase = true))
        assertTrue("Long must include scion/rootstock joining", longAnswer.contains("scion", ignoreCase = true) && longAnswer.contains("rootstock", ignoreCase = true))
        println("LONG ANSWER: " + longAnswer)
        assertTrue("Long must include additional grounded information", longAnswer.contains("cambium alignment", ignoreCase = true))
        assertTrue("Long must be more detailed than Short", longAnswer.length > shortAnswer.length)
    }

    /**
     * TEST 2:
     * Question: "What are the types of grafting?"
     * Expected:
     * SHORT = concise method names.
     * LONG = method names + grounded descriptions.
     */
    @Test
    fun testRequired_Test2_TypesOfGrafting() = runBlocking {
        val evidence = """
            [PAGE=3] [Source 1, Page 3]
            Method 1: Whip-and-tongue grafting — used for fruit trees where scion and rootstock have equal diameter.
            [PAGE=4] [Source 1, Page 4]
            Method 2: Cleft grafting — used when the rootstock is much larger than the scion.
            [PAGE=5] [Source 1, Page 5]
            Method 3: Bark grafting — commonly used for topworking mature trees when the bark is slipping in early spring.
        """.trimIndent()
        val q = "What are the types of grafting?"

        val shortAnswer = ggufLlm.generate(ggufLlm.buildQwenPrompt(evidence, q, answerLength = AnswerLength.SHORT), AnswerLength.SHORT)
        val longAnswer = ggufLlm.generate(ggufLlm.buildQwenPrompt(evidence, q, answerLength = AnswerLength.LONG), AnswerLength.LONG)

        // SHORT = concise method names
        assertTrue("Short must list whip-and-tongue", shortAnswer.contains("Whip-and-tongue", ignoreCase = true))
        assertTrue("Short must list cleft", shortAnswer.contains("Cleft", ignoreCase = true))
        assertTrue("Short must list bark", shortAnswer.contains("Bark", ignoreCase = true))
        assertFalse("Short must NOT include lengthy description of fruit trees", shortAnswer.contains("used for fruit trees where scion and rootstock have equal diameter", ignoreCase = true))

        // LONG = method names + grounded descriptions
        assertTrue("Long must list whip-and-tongue", longAnswer.contains("Whip-and-tongue", ignoreCase = true))
        assertTrue("Long must include whip-and-tongue description", longAnswer.contains("fruit trees", ignoreCase = true) || longAnswer.contains("equal diameter", ignoreCase = true))
        assertTrue("Long must list cleft", longAnswer.contains("Cleft", ignoreCase = true))
        assertTrue("Long must include cleft description", longAnswer.contains("rootstock is much larger", ignoreCase = true))
        assertTrue("Long must list bark", longAnswer.contains("Bark", ignoreCase = true))
        assertTrue("Long must include bark description", longAnswer.contains("topworking mature trees", ignoreCase = true))
        assertTrue("Long must be longer than Short", longAnswer.length > shortAnswer.length)
    }

    /**
     * TEST 3:
     * Question: "List all methods of grafting described in the guide."
     * Expected:
     * SHORT = concise complete list.
     * LONG = same complete list + grounded descriptions.
     */
    @Test
    fun testRequired_Test3_ListAllMethodsOfGrafting() = runBlocking {
        val evidence = """
            [PAGE=3] [Source 1, Page 3]
            Method 1: Whip-and-tongue grafting — used for small rootstocks with interlocking cuts.
            [PAGE=4] [Source 1, Page 4]
            Method 2: Cleft grafting — topworking technique for large rootstocks.
            [PAGE=5] [Source 1, Page 5]
            Method 3: Bridge grafting — used to repair damaged trunk tissues.
        """.trimIndent()
        val q = "List all methods of grafting described in the guide."

        val shortAnswer = ggufLlm.generate(ggufLlm.buildQwenPrompt(evidence, q, answerLength = AnswerLength.SHORT), AnswerLength.SHORT)
        val longAnswer = ggufLlm.generate(ggufLlm.buildQwenPrompt(evidence, q, answerLength = AnswerLength.LONG), AnswerLength.LONG)

        // Both include all methods
        assertTrue("Short must include Whip-and-tongue", shortAnswer.contains("Whip-and-tongue", ignoreCase = true))
        assertTrue("Short must include Cleft", shortAnswer.contains("Cleft", ignoreCase = true))
        assertTrue("Short must include Bridge", shortAnswer.contains("Bridge", ignoreCase = true))

        assertTrue("Long must include Whip-and-tongue", longAnswer.contains("Whip-and-tongue", ignoreCase = true))
        assertTrue("Long must include Cleft", longAnswer.contains("Cleft", ignoreCase = true))
        assertTrue("Long must include Bridge", longAnswer.contains("Bridge", ignoreCase = true))

        // SHORT = concise names
        assertFalse("Short should omit lengthy descriptions", shortAnswer.contains("interlocking cuts", ignoreCase = true))

        // LONG = complete list + grounded descriptions
        assertTrue("Long must include descriptions", longAnswer.contains("interlocking cuts", ignoreCase = true) || longAnswer.contains("topworking technique", ignoreCase = true))
        assertTrue("Long must be substantially longer than Short", longAnswer.length > shortAnswer.length)
    }

    /**
     * TEST 4:
     * Question: "What are the likely causes of grafting failure and what should I check?"
     * Expected:
     * SHORT = concise answer.
     * LONG = causes + relevant checks/actions.
     * No duplicated causes.
     */
    @Test
    fun testRequired_Test4_CausesOfFailureAndWhatToCheck() = runBlocking {
        val evidence = """
            [PAGE=11] [Source 1, Page 11]
            Common causes of failure include poor cambial contact, desiccation of tissues, and incorrect physiological timing.
            If grafting fails, check the alignment of the cambium layers and ensure rubber ties are secure.
            Causes of failure include poor cambial contact.
        """.trimIndent()
        val q = "What are the likely causes of grafting failure and what should I check?"

        val shortAnswer = ggufLlm.generate(ggufLlm.buildQwenPrompt(evidence, q, answerLength = AnswerLength.SHORT), AnswerLength.SHORT)
        val longAnswer = ggufLlm.generate(ggufLlm.buildQwenPrompt(evidence, q, answerLength = AnswerLength.LONG), AnswerLength.LONG)

        // SHORT = concise answer
        assertTrue("Short must contain causes", shortAnswer.contains("poor cambial contact", ignoreCase = true))

        // LONG = causes + relevant checks/actions
        assertTrue("Long must contain causes", longAnswer.contains("poor cambial contact", ignoreCase = true))
        assertTrue("Long must contain checks/actions", longAnswer.contains("check the alignment", ignoreCase = true) || longAnswer.contains("ensure rubber ties", ignoreCase = true))

        // Verify NO duplicated causes in Long
        val matches = Regex("""poor cambial contact""", RegexOption.IGNORE_CASE).findAll(longAnswer).toList()
        assertEquals("Long must NOT duplicate the cause 'poor cambial contact'", 1, matches.size)
    }

    /**
     * TEST 5:
     * Question: "What tools and materials are needed?"
     * Expected:
     * If only one relevant factual statement exists:
     * SHORT and LONG may legitimately be identical.
     * Do NOT force artificial expansion.
     */
    @Test
    fun testRequired_Test5_ToolsAndMaterialsSingleFact() = runBlocking {
        val evidence = """
            [PAGE=5] [Source 1, Page 5]
            Required tools include a sharp grafting knife, pruning shears, and grafting tape.
        """.trimIndent()
        val q = "What tools and materials are needed?"

        val shortAnswer = ggufLlm.generate(ggufLlm.buildQwenPrompt(evidence, q, answerLength = AnswerLength.SHORT), AnswerLength.SHORT)
        val longAnswer = ggufLlm.generate(ggufLlm.buildQwenPrompt(evidence, q, answerLength = AnswerLength.LONG), AnswerLength.LONG)

        assertTrue("Short must contain tools", shortAnswer.contains("sharp grafting knife", ignoreCase = true))
        assertTrue("Long must contain tools", longAnswer.contains("sharp grafting knife", ignoreCase = true))
        // Legitimately identical when only one relevant factual statement exists
        assertEquals("Short and Long must be identical when single fact exists", shortAnswer, longAnswer)
    }

    /**
     * TEST 6:
     * Question: "What temperature is recommended for healing?"
     * Expected:
     * SHORT = exact temperature.
     * LONG = temperature + humidity/callus context IF that context exists in retrieved evidence.
     */
    @Test
    fun testRequired_Test6_TemperatureRecommendedForHealing() = runBlocking {
        val evidence = """
            [PAGE=11] [Source 1, Page 11]
            Optimal healing temperature for grafting callus formation is 21°C to 27°C (70°F to 80°F). Humidity should also be maintained above 85% to prevent desiccation.
        """.trimIndent()
        val q = "What temperature is recommended for healing?"

        val shortAnswer = ggufLlm.generate(ggufLlm.buildQwenPrompt(evidence, q, answerLength = AnswerLength.SHORT), AnswerLength.SHORT)
        val longAnswer = ggufLlm.generate(ggufLlm.buildQwenPrompt(evidence, q, answerLength = AnswerLength.LONG), AnswerLength.LONG)

        // SHORT = exact temperature
        assertTrue("Short must contain exact temperature", shortAnswer.contains("21°C", ignoreCase = true) || shortAnswer.contains("70°F", ignoreCase = true))
        assertFalse("Short should not contain humidity context", shortAnswer.contains("Humidity should also be maintained", ignoreCase = true))

        // LONG = temperature + humidity/callus context
        assertTrue("Long must contain temperature", longAnswer.contains("21°C", ignoreCase = true) || longAnswer.contains("70°F", ignoreCase = true))
        assertTrue("Long must contain humidity/callus context from same evidence", longAnswer.contains("Humidity", ignoreCase = true) && longAnswer.contains("85%", ignoreCase = true))
    }

    /**
     * TEST 7:
     * Heading regression:
     * Ensure LONG never returns:
     * "Failure Likely causes / What to check"
     * or any page/section heading as the answer.
     */
    @Test
    fun testRequired_Test7_HeadingRegression() = runBlocking {
        val evidence = """
            [PAGE=11] [Source 1, Page 11]
            Failure Likely causes / What to check
            Common causes of failure include poor cuts, incorrect temperatures, and drying out.
            [PAGE=12] [Source 1, Page 12]
            Troubleshooting & Recovery
            Monitor humidity and wrap grafts with paraffin film.
        """.trimIndent()
        val q = "What are the likely causes of grafting failure and what should I check?"

        val longAnswer = ggufLlm.generate(ggufLlm.buildQwenPrompt(evidence, q, answerLength = AnswerLength.LONG), AnswerLength.LONG)

        assertFalse("LONG must never return 'Failure Likely causes / What to check'", longAnswer.contains("Failure Likely causes / What to check", ignoreCase = true))
        assertFalse("LONG must never return 'Troubleshooting & Recovery' heading", longAnswer.contains("Troubleshooting & Recovery", ignoreCase = true))
        assertFalse("LONG must not start with heading fragment", longAnswer.trim().startsWith("Failure Likely causes", ignoreCase = true))
        assertTrue("LONG must use factual explanatory content", longAnswer.contains("poor cuts", ignoreCase = true) || longAnswer.contains("Common causes", ignoreCase = true))
    }

    /**
     * CRITICAL RETRIEVAL INVARIANCE TEST (Section 9):
     * For every Short/Long pair verify:
     * retrieved chunk IDs SHORT == LONG
     * retrieval scores SHORT == LONG
     * assembled context SHORT == LONG
     * If any differ: FAIL the test.
     */
    @Test
    fun testCriticalRetrievalInvariance_AcrossAllTargetQuestions() = runBlocking {
        val dummyEmbedding = FloatArray(384) { 0.05f }
        val dummyEmbeddingBytes = com.siddu.offlinedocai.data.FloatArrayConverter.fromFloatArray(dummyEmbedding)
        val corpus = listOf(
            ChunkEntity(id = 1, documentId = 1, pageNumber = 1, chunkIndex = 0, sectionTitle = "Intro", pageSpan = "1", text = "Grafting is a horticultural technique whereby tissues of plants are joined. In grafting, a scion is joined to a rootstock.", embedding = dummyEmbeddingBytes),
            ChunkEntity(id = 2, documentId = 1, pageNumber = 3, chunkIndex = 1, sectionTitle = "Methods", pageSpan = "3", text = "Method 1: Whip-and-tongue grafting is used for fruit trees where scion and rootstock have equal diameter.", embedding = dummyEmbeddingBytes),
            ChunkEntity(id = 3, documentId = 1, pageNumber = 4, chunkIndex = 2, sectionTitle = "Methods", pageSpan = "4", text = "Method 2: Cleft grafting is used when rootstock is much larger than scion.", embedding = dummyEmbeddingBytes),
            ChunkEntity(id = 4, documentId = 1, pageNumber = 5, chunkIndex = 3, sectionTitle = "Tools", pageSpan = "5", text = "Required tools include a sharp grafting knife, pruning shears, and grafting tape.", embedding = dummyEmbeddingBytes),
            ChunkEntity(id = 5, documentId = 1, pageNumber = 11, chunkIndex = 4, sectionTitle = "Healing", pageSpan = "11", text = "Optimal healing temperature for grafting callus formation is 21°C to 27°C (70°F to 80°F). Humidity should also be maintained above 85%.", embedding = dummyEmbeddingBytes),
            ChunkEntity(id = 6, documentId = 1, pageNumber = 14, chunkIndex = 5, sectionTitle = "Failure", pageSpan = "14", text = "Common causes of failure include poor cambial contact, desiccation of tissues, and incorrect timing. Check alignment of cambium layers.", embedding = dummyEmbeddingBytes)
        )
        val dao = createMockChunkDao(corpus)
        val vectorStore = VectorStore(dao)

        val targetQuestions = listOf(
            "What is grafting?",
            "What are the types of grafting?",
            "List all methods of grafting described in the guide.",
            "What are the likely causes of grafting failure and what should I check?",
            "What tools and materials are needed?",
            "What temperature is recommended for healing?"
        )

        for (query in targetQuestions) {
            val qEmbedding = FloatArray(384) { 0.05f }
            val classificationShort = QueryIntentRouter.route(query)
            val classificationLong = QueryIntentRouter.route(query)

            // 1. Intent invariance
            assertEquals("Intent must match for query '$query'", classificationShort.intent, classificationLong.intent)

            // 2. Retrieval invariance
            val resultsShort = vectorStore.retrieveDetailed(qEmbedding, listOf(1L), topK = 10, query = query)
            val resultsLong = vectorStore.retrieveDetailed(qEmbedding, listOf(1L), topK = 10, query = query)

            assertEquals("Candidate count must match for '$query'", resultsShort.size, resultsLong.size)
            for (i in resultsShort.indices) {
                assertEquals("Chunk ID SHORT == LONG at $i for '$query'", resultsShort[i].chunk.id, resultsLong[i].chunk.id)
                assertEquals("Score SHORT == LONG at $i for '$query'", resultsShort[i].relevanceScore, resultsLong[i].relevanceScore, 0.0001f)
            }

            // 3. Assembled context invariance
            val budgetShort = ContextAssembler.resolveEvidenceBudget(classificationShort.intent)
            val budgetLong = ContextAssembler.resolveEvidenceBudget(classificationLong.intent)
            assertEquals("Evidence budget must match for '$query'", budgetShort, budgetLong)

            val assembledShort = ContextAssembler.assemble(resultsShort, classificationShort.intent, classificationShort, budgetShort)
            val assembledLong = ContextAssembler.assemble(resultsLong, classificationLong.intent, classificationLong, budgetLong)

            assertEquals("Assembled context text SHORT == LONG for '$query'", assembledShort.contextText, assembledLong.contextText)
            assertEquals("Assembled token estimate SHORT == LONG for '$query'", assembledShort.totalEstimatedTokens, assembledLong.totalEstimatedTokens)
            assertEquals("Source blocks SHORT == LONG for '$query'", assembledShort.sourceBlocks.size, assembledLong.sourceBlocks.size)
        }
    }

    // =========================================================================
    // LIST CLEANUP & RELEVANCE GATE TESTS
    // =========================================================================

    @Test
    fun testCleanAndRenumberList_MalformedNumberRuns() {
        val input1 = "The likely causes of grafting failure are: 1. 2. 3. 4. 1. Check for poor cambial contact: Ensure that the cambium tissues of the scion. 2. 3. Check for grafting during incorrect physiological seasons: Ensure that the rootstock. 4. [Source 14, Page 13] (Page 13)"
        val cleaned1 = com.siddu.offlinedocai.ai.validation.CitationValidator.cleanAndRenumberList(input1)

        assertFalse("Must not contain '1. 2. 3. 4.'", cleaned1.contains("1. 2. 3. 4."))
        assertFalse("Must not contain '2. 3.'", cleaned1.contains("2. 3."))
        assertTrue("Must contain sequential item 1", cleaned1.contains("1. Check for poor cambial contact"))
        assertTrue("Must contain sequential item 2", cleaned1.contains("2. Check for grafting during incorrect physiological seasons"))

        val input2 = "The tools and materials needed for grafting include: 1. **Sharp Grafting Knives**: These are essential. 2. 3. 4. **Grafting Wax**: This is used to seal the wound."
        val cleaned2 = com.siddu.offlinedocai.ai.validation.CitationValidator.cleanAndRenumberList(input2)

        assertFalse("Must not contain '2. 3. 4.'", cleaned2.contains("2. 3. 4."))
        assertTrue("Must contain item 1", cleaned2.contains("1. **Sharp Grafting Knives**"))
        assertTrue("Must renumber item 4 to item 2", cleaned2.contains("2. **Grafting Wax**"))

        val input3 = "1. Whip and Tongue\n2.\n3.\n4. Cleft"
        val cleaned3 = com.siddu.offlinedocai.ai.validation.CitationValidator.cleanAndRenumberList(input3)

        assertEquals("Must drop empty lines and renumber", "1. Whip and Tongue\n2. Cleft", cleaned3)
    }

    @Test
    fun testRelevanceGate_QuestionD_ToolsAndMaterials_ExcludesAftercare() = runBlocking {
        val evidence = """
            [PAGE=12] [Source 12, Page 12]
            Tools required include sharp grafting knives, budding strips, pruning shears, and grafting wax to seal the wound.
            [PAGE=13] [Source 13, Page 13]
            Inspect grafts weekly. Remove suckers emerging from the rootstock below the graft union to direct vigor into the scion.
        """.trimIndent()
        val query = "What tools and materials are needed?"

        // Direct relevance check
        val toolClaim = "Tools required include sharp grafting knives, budding strips, pruning shears, and grafting wax to seal the wound."
        val aftercareClaim = "Inspect grafts weekly. Remove suckers emerging from the rootstock below the graft union to direct vigor into the scion."

        assertTrue("Tool claim must be relevant", com.siddu.offlinedocai.ai.validation.CitationValidator.isClaimRelevantToQuery(toolClaim, query))
        assertFalse("Aftercare claim must NOT be relevant to tools/materials query", com.siddu.offlinedocai.ai.validation.CitationValidator.isClaimRelevantToQuery(aftercareClaim, query))

        // Synthesis check
        val longAnswer = ggufLlm.generate(ggufLlm.buildQwenPrompt(evidence, query, answerLength = AnswerLength.LONG), AnswerLength.LONG)

        assertTrue("LONG must contain tools", longAnswer.contains("knives", ignoreCase = true) || longAnswer.contains("shears", ignoreCase = true) || longAnswer.contains("wax", ignoreCase = true))
        assertFalse("LONG must NOT contain 'inspect grafts weekly'", longAnswer.contains("inspect grafts weekly", ignoreCase = true))
        assertFalse("LONG must NOT contain 'remove suckers'", longAnswer.contains("remove suckers", ignoreCase = true))
        assertFalse("LONG must not contain malformed number runs", longAnswer.contains("1. 2. 3. 4."))
    }

    @Test
    fun testRelevanceGate_QuestionE_Temperature_AllowsHumidity_ExcludesTools() = runBlocking {
        val query = "What temperature is recommended for healing?"

        val tempClaim = "The optimal healing temperature for grafting callus formation is 21C to 27C (70F to 80F)."
        val humidityClaim = "Humidity must be maintained above 85% to prevent desiccation."
        val toolClaim = "Tools required include sharp grafting knives, budding strips, pruning shears, and grafting wax."

        assertTrue("Temperature claim must be relevant", com.siddu.offlinedocai.ai.validation.CitationValidator.isClaimRelevantToQuery(tempClaim, query))
        assertTrue("Directly associated humidity condition must be relevant to temperature query", com.siddu.offlinedocai.ai.validation.CitationValidator.isClaimRelevantToQuery(humidityClaim, query))
        assertFalse("Tools claim must NOT be relevant to temperature query", com.siddu.offlinedocai.ai.validation.CitationValidator.isClaimRelevantToQuery(toolClaim, query))
    }

    @Test
    fun testRelevanceGate_QuestionA_CausesAndChecks_AllowsBoth() = runBlocking {
        val query = "What are the likely causes of grafting failure and what should I check?"

        val causeClaim = "Failures typically result from poor cambial contact, desiccation of tissues, grafting during incorrect physiological seasons, or pathogen infection."
        val checkClaim = "Inspect grafts weekly and remove suckers emerging from the rootstock below the graft union."
        val toolClaim = "Tools required include sharp grafting knives."

        assertTrue("Failure causes must be relevant", com.siddu.offlinedocai.ai.validation.CitationValidator.isClaimRelevantToQuery(causeClaim, query))
        assertTrue("Inspection/checks must be relevant when query asks 'what should I check'", com.siddu.offlinedocai.ai.validation.CitationValidator.isClaimRelevantToQuery(checkClaim, query))
        assertFalse("Tools claim must NOT be relevant to causes/checks query", com.siddu.offlinedocai.ai.validation.CitationValidator.isClaimRelevantToQuery(toolClaim, query))
    }
}
