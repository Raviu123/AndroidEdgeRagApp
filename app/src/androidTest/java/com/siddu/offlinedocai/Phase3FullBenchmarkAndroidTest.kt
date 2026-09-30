package com.siddu.offlinedocai

import android.content.Context
import android.net.Uri
import androidx.room.Room
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.siddu.offlinedocai.ai.chunking.TextChunker
import com.siddu.offlinedocai.ai.embeddings.ONNXEmbeddingEngine
import com.siddu.offlinedocai.ai.llm.GGUFLLM
import com.siddu.offlinedocai.ai.retrieval.ContextAssembler
import com.siddu.offlinedocai.ai.retrieval.QueryIntentRouter
import com.siddu.offlinedocai.ai.retrieval.VectorStore
import com.siddu.offlinedocai.ai.validation.CitationValidator
import com.siddu.offlinedocai.data.AppDatabase
import com.siddu.offlinedocai.data.FloatArrayConverter
import com.siddu.offlinedocai.data.entities.ChunkEntity
import com.siddu.offlinedocai.data.entities.DocumentEntity
import com.siddu.offlinedocai.domain.rag.DocumentProcessor
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class Phase3FullBenchmarkAndroidTest {

    private lateinit var context: Context
    private lateinit var database: AppDatabase
    private lateinit var onnxEngine: ONNXEmbeddingEngine
    private lateinit var vectorStore: VectorStore
    private lateinit var llm: GGUFLLM
    private lateinit var testPdfFile: File
    private var docId: Long = 0L

    data class QueryExpectation(
        val query: String,
        val expectedPages: List<Int>,
        val primaryExpectedPage: Int
    )

    @Before
    fun setup() = runBlocking {
        context = InstrumentationRegistry.getInstrumentation().targetContext
        PDFBoxResourceLoader.init(context)

        database = Room.inMemoryDatabaseBuilder(context, AppDatabase::class.java)
            .allowMainThreadQueries()
            .build()

        vectorStore = VectorStore(database.chunkDao())
        onnxEngine = ONNXEmbeddingEngine(context)
        llm = GGUFLLM(context)

        // Read physical PDF pushed to /data/local/tmp or cache
        val pushedFile = File("/data/local/tmp/Grafting_15_Page_Guide.pdf")
        testPdfFile = File(context.cacheDir, "Grafting_15_Page_Guide.pdf")
        if (pushedFile.exists() && pushedFile.canRead()) {
            pushedFile.copyTo(testPdfFile, overwrite = true)
        }
        assertTrue("Test PDF must exist on device disk", testPdfFile.exists() && testPdfFile.length() > 1000)
    }

    @After
    fun tearDown() {
        llm.llamaRunner.freeModel()
        database.close()
        if (testPdfFile.exists()) {
            testPdfFile.delete()
        }
        onnxEngine.close()
    }

    @Test
    fun testPhase3_FullIngestionAnd17QueriesBenchmark() = runBlocking {
        println("=================================================================")
        println("OFFLINEDOCAI — PHASE 3 FULL ANDROID INGESTION & RAG BENCHMARK")
        println("=================================================================")

        // =====================================================================
        // TASK 3 — REAL ANDROID INGESTION
        // =====================================================================
        println("\n>>> TASK 3: RUNNING REAL ANDROID INGESTION <<<")

        // 1. PDF extraction via PDFBox
        val extractedDoc = DocumentProcessor.extractDocument(context, Uri.fromFile(testPdfFile))
        println("PDF Extraction: ${extractedDoc.pageCount} pages extracted")
        assertEquals("PDF extraction must yield exactly 15 pages", 15, extractedDoc.pageCount)
        assertEquals("Page list size must match page count", 15, extractedDoc.pages.size)

        for ((idx, page) in extractedDoc.pages.withIndex()) {
            assertEquals("Page index must be 1-based sequential", idx + 1, page.pageNumber)
            assertTrue("Page ${page.pageNumber} text must not be blank", page.text.isNotBlank())
        }

        docId = database.documentDao().insert(
            DocumentEntity(
                uri = Uri.fromFile(testPdfFile).toString(),
                name = "Grafting_15_Page_Guide.pdf",
                type = "application/pdf",
                pageCount = 15,
                status = "READY"
            )
        )
        println("Document Entity Inserted in Room: ID=$docId")

        // 2. TextChunker
        val documentChunks = TextChunker.chunkDocument(extractedDoc.pages, targetWords = 35, maxWords = 50)
        println("Chunk Creation: ${documentChunks.size} chunks created across 15 pages")
        for (c in documentChunks) {
            println("CHUNK_INFO: Index=${c.index}, Page=${c.pageNumber}, Span=${c.pageSpan}, Section='${c.sectionTitle}', Words=${c.text.split(Regex("\\s+")).size}")
            println("CHUNK_TEXT: ${c.text.replace("\n", " -- ")}")
        }

        // 3. ONNX Embedding Engine
        val allChunks = documentChunks.map { c ->
            val embedding = onnxEngine.getEmbedding(c.text)
            assertEquals("Embedding must be exactly 384D", 384, embedding.size)
            val nonZero = embedding.count { it != 0f }
            assertTrue("Embedding must have non-zero components", nonZero > 350)
            ChunkEntity(
                documentId = docId,
                pageNumber = c.pageNumber,
                chunkIndex = c.index,
                text = c.text,
                embedding = FloatArrayConverter.fromFloatArray(embedding),
                pageSpan = c.pageSpan,
                sectionTitle = c.sectionTitle,
                paragraphIndex = c.paragraphIndex
            )
        }

        // 4. Room Insertion
        database.chunkDao().insertAll(allChunks)
        val insertedChunks = database.chunkDao().getAllChunksForDocumentIds(listOf(docId))
        println("Room Persistence: ${insertedChunks.size} chunks verified in SQLite Room database")
        assertEquals("All 15 chunks must be in Room database", 15, insertedChunks.size)

        println(">>> TASK 3 REAL ANDROID INGESTION: VERIFIED PASS <<<")

        // =====================================================================
        // TASK 4 — RUN RETRIEVAL BENCHMARK (17 QUERIES)
        // =====================================================================
        println("\n>>> TASK 4: RUNNING 17-QUERY RETRIEVAL BENCHMARK <<<")

        val benchmarkQueries = listOf(
            QueryExpectation("What is grafting?", listOf(1), 1),
            QueryExpectation("What are the scion and rootstock?", listOf(1, 15), 1),
            QueryExpectation("What is vascular cambium?", listOf(2), 2),
            QueryExpectation("What are the types of grafting?", (3..10).toList(), 3),
            QueryExpectation("What is whip and tongue grafting?", listOf(3), 3),
            QueryExpectation("What is cleft grafting?", listOf(4), 4),
            QueryExpectation("What is bark grafting?", listOf(5), 5),
            QueryExpectation("What is bridge grafting?", listOf(6), 6),
            QueryExpectation("What is side-veneer grafting?", listOf(7), 7),
            QueryExpectation("What is splice grafting?", listOf(8), 8),
            QueryExpectation("What is saddle grafting?", listOf(9), 9),
            QueryExpectation("What is approach grafting?", listOf(10), 10),
            QueryExpectation("What temperature is optimal for graft healing?", listOf(11), 11),
            QueryExpectation("What tools and materials are required?", listOf(12), 12),
            QueryExpectation("What should be done during aftercare?", listOf(13), 13),
            QueryExpectation("What are common causes of graft failure?", listOf(14), 14),
            QueryExpectation("What is a graft union?", listOf(15), 15)
        )

        for ((qIdx, qItem) in benchmarkQueries.withIndex()) {
            val queryNumber = qIdx + 1
            val query = qItem.query
            val qClassification = QueryIntentRouter.route(query)
            val qEmbedding = onnxEngine.getEmbedding(query)

            val retrieved = vectorStore.retrieveDetailed(
                queryEmbedding = qEmbedding,
                relevantDocumentIds = listOf(docId),
                topK = 10,
                query = query
            )

            val retrievedPages = retrieved.map { it.chunk.pageNumber }
            val topRetrievalPage = retrieved.firstOrNull()?.chunk?.pageNumber ?: -1
            val topScore = retrieved.firstOrNull()?.relevanceScore ?: 0f

            // Find rank of primary expected page
            val rankOfExpected = retrieved.indexOfFirst { qItem.expectedPages.contains(it.chunk.pageNumber) }.let { if (it >= 0) it + 1 else -1 }
            val correctEvidenceRetrieved = rankOfExpected > 0

            // Assemble and Generate
            val assembled = ContextAssembler.assemble(
                retrievedResults = retrieved,
                queryIntent = qClassification.intent,
                queryClassification = qClassification,
                maxEvidenceTokens = 300
            )

            val prompt = llm.buildQwenPrompt(assembled.contextText, query)
            val answer = try {
                llm.generate(prompt)
            } catch (e: Exception) {
                "ERROR: ${e.message}"
            }
            val answerGenerated = answer.isNotBlank() && !answer.startsWith("ERROR")

            // Citation Validation
            val validation = CitationValidator.validateWithSourceBlocks(
                answer = answer,
                sourceBlocks = assembled.sourceBlocks,
                query = query,
                isBudgetLimited = assembled.isBudgetLimited
            )
            val citationCorrect = validation.validCitations.any { qItem.expectedPages.contains(it.page) } ||
                    (answer.contains("Page ${qItem.primaryExpectedPage}") || answer.contains("page ${qItem.primaryExpectedPage}"))

            println("-----------------------------------------------------------------")
            println("QUERY $queryNumber: $query")
            println("EXPECTED PAGE(S): ${qItem.expectedPages.joinToString(", ")}")
            println("RETRIEVED PAGE(S): ${retrievedPages.joinToString(", ")}")
            println("TOP RETRIEVAL: Page $topRetrievalPage")
            println("RANK: ${if (rankOfExpected > 0) rankOfExpected.toString() else "NOT_FOUND"}")
            println("RETRIEVAL SCORE: $topScore")
            println("CORRECT EVIDENCE RETRIEVED: ${if (correctEvidenceRetrieved) "YES" else "NO"}")
            println("ANSWER GENERATED: ${if (answerGenerated) "YES" else "NO"}")
            println("CITATION CORRECT: ${if (citationCorrect) "YES" else "NO"}")
            println("GENERATED ANSWER EXCERPT: ${answer.take(150).replace("\n", " ")}")
        }

        // =====================================================================
        // TASK 5 — TEST EXHAUSTIVE RETRIEVAL ("What are the types of grafting?")
        // =====================================================================
        println("\n>>> TASK 5: EVALUATING EXHAUSTIVE RETRIEVAL <<<")
        val exhaustiveQuery = "What are the types of grafting?"
        val exClassification = QueryIntentRouter.route(exhaustiveQuery)
        val exEmbedding = onnxEngine.getEmbedding(exhaustiveQuery)
        val exRetrieved = vectorStore.retrieveDetailed(
            queryEmbedding = exEmbedding,
            relevantDocumentIds = listOf(docId),
            topK = 15,
            query = exhaustiveQuery
        )
        val exPages = exRetrieved.map { it.chunk.pageNumber }
        val methodPages = (3..10).toList()
        val foundMethods = methodPages.filter { exPages.contains(it) }
        val missingMethods = methodPages.filter { !exPages.contains(it) }
        println("Exhaustive Query: '$exhaustiveQuery'")
        println("All Retrieved Pages (top 15): $exPages")
        println("Target Method Pages (3..10): $methodPages")
        println("Retrieved Method Pages (${foundMethods.size}/8): $foundMethods")
        println("Missing Method Pages (${missingMethods.size}/8): $missingMethods")
        val exhaustiveVerified = missingMethods.isEmpty()
        println("EXHAUSTIVE RETRIEVAL FULLY VERIFIED (ALL 8 PAGES RETRIEVED): ${if (exhaustiveVerified) "YES" else "NO"}")

        // =====================================================================
        // TASK 6 — TEST TEMPERATURE PAGE (Page 11)
        // =====================================================================
        println("\n>>> TASK 6: TEST TEMPERATURE PAGE <<<")
        val tempQuery = "What temperature is optimal for graft healing?"
        val tempEmbedding = onnxEngine.getEmbedding(tempQuery)
        val tempRetrieved = vectorStore.retrieveDetailed(
            queryEmbedding = tempEmbedding,
            relevantDocumentIds = listOf(docId),
            topK = 5,
            query = tempQuery
        )
        val tempTopPage = tempRetrieved.firstOrNull()?.chunk?.pageNumber ?: -1
        val tempPage11Chunk = tempRetrieved.firstOrNull { it.chunk.pageNumber == 11 }
        val tempText = tempPage11Chunk?.chunk?.text ?: ""
        val tempPrompt = llm.buildQwenPrompt(tempText, tempQuery)
        val tempAnswer = llm.generate(tempPrompt)
        println("Query: '$tempQuery'")
        println("Top Retrieved Page: Page $tempTopPage")
        println("Top Score: ${tempRetrieved.firstOrNull()?.relevanceScore}")
        println("Page 11 Evidence Text: $tempText")
        println("Answer: $tempAnswer")
        val tempGrounded = tempAnswer.contains("21") || tempAnswer.contains("27") || tempAnswer.contains("70") || tempAnswer.contains("80")
        println("TEMPERATURE RETRIEVED PAGE 11: ${if (tempTopPage == 11) "YES" else "NO"}")
        println("TEMPERATURE GROUNDED IN PAGE 11 (21C-27C / 70F-80F): ${if (tempGrounded) "YES" else "NO"}")

        // =====================================================================
        // TASK 7 — TEST TOOLS (Page 12)
        // =====================================================================
        println("\n>>> TASK 7: TEST TOOLS PAGE <<<")
        val toolsQuery = "What tools and materials are required?"
        val toolsEmbedding = onnxEngine.getEmbedding(toolsQuery)
        val toolsRetrieved = vectorStore.retrieveDetailed(
            queryEmbedding = toolsEmbedding,
            relevantDocumentIds = listOf(docId),
            topK = 5,
            query = toolsQuery
        )
        val toolsTopPage = toolsRetrieved.firstOrNull()?.chunk?.pageNumber ?: -1
        val toolsPage12Chunk = toolsRetrieved.firstOrNull { it.chunk.pageNumber == 12 }
        val toolsText = toolsPage12Chunk?.chunk?.text ?: ""
        val toolsPrompt = llm.buildQwenPrompt(toolsText, toolsQuery)
        val toolsAnswer = llm.generate(toolsPrompt)
        println("Query: '$toolsQuery'")
        println("Top Retrieved Page: Page $toolsTopPage")
        println("Top Score: ${toolsRetrieved.firstOrNull()?.relevanceScore}")
        println("Page 12 Evidence Text: $toolsText")
        println("Answer: $toolsAnswer")
        val toolsGrounded = toolsAnswer.contains("knife", ignoreCase = true) ||
                toolsAnswer.contains("knives", ignoreCase = true) ||
                toolsAnswer.contains("wax", ignoreCase = true) ||
                toolsAnswer.contains("strip", ignoreCase = true) ||
                toolsAnswer.contains("shear", ignoreCase = true)
        println("TOOLS RETRIEVED PAGE 12: ${if (toolsTopPage == 12) "YES" else "NO"}")
        println("TOOLS GROUNDED IN PAGE 12: ${if (toolsGrounded) "YES" else "NO"}")

        println("\n=================================================================")
        println("PHASE 3 FULL ANDROID TEST COMPLETE")
        println("=================================================================")
    }
}
