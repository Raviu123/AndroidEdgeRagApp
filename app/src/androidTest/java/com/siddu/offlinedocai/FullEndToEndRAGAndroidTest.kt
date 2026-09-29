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
import com.siddu.offlinedocai.data.AppDatabase
import com.siddu.offlinedocai.data.FloatArrayConverter
import com.siddu.offlinedocai.data.entities.ChunkEntity
import com.siddu.offlinedocai.data.entities.DocumentEntity
import com.siddu.offlinedocai.domain.rag.DocumentProcessor
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.pdmodel.PDPage
import com.tom_roush.pdfbox.pdmodel.PDPageContentStream
import com.tom_roush.pdfbox.pdmodel.font.PDType1Font
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.math.sqrt

/**
 * End-to-End Real On-Device Verification Suite for OfflineDocAI.
 *
 * Verifies the complete offline RAG pipeline on actual Android runtime:
 * 1. Real ONNX Embedding Generation (all-MiniLM-L6-v2.onnx)
 *    - 384 dimensions, non-zero values, L2 normalized (~1.0f)
 *    - Valid cosine similarity separation (semantic > orthogonal)
 * 2. Real Document Ingestion & Structure-Preserving Chunking
 *    - Programmatic generation of 15-page Grafting Guide PDF
 *    - Text extraction via PDFBox-Android
 *    - Chunking with page and section preservation
 * 3. Real Vector Storage & Retrieval
 *    - Insertion into Room Database with Room-backed VectorStore
 *    - Intent-based hybrid semantic retrieval
 * 4. Real Grounded Synthesis via On-Device GGUF (Qwen 2.5 0.5B via llama.cpp)
 *    - Closed-world generation, exact metrics, multi-source listing, absent-fact fallback
 */
@RunWith(AndroidJUnit4::class)
class FullEndToEndRAGAndroidTest {

    private lateinit var context: Context
    private lateinit var database: AppDatabase
    private lateinit var onnxEngine: ONNXEmbeddingEngine
    private lateinit var vectorStore: VectorStore
    private lateinit var llm: GGUFLLM
    private lateinit var testPdfFile: File

    @Before
    fun setup() {
        context = InstrumentationRegistry.getInstrumentation().targetContext
        PDFBoxResourceLoader.init(context)

        // In-memory Room database for clean test isolation
        database = Room.inMemoryDatabaseBuilder(context, AppDatabase::class.java)
            .allowMainThreadQueries()
            .build()

        vectorStore = VectorStore(database.chunkDao())
        onnxEngine = ONNXEmbeddingEngine(context)
        llm = GGUFLLM(context)

        testPdfFile = File(context.cacheDir, "Grafting_15_Page_Guide.pdf")
        createGraftingGuidePdf(testPdfFile)
    }

    @After
    fun tearDown() {
        database.close()
        if (testPdfFile.exists()) {
            testPdfFile.delete()
        }
        onnxEngine.close()
    }

    private fun createGraftingGuidePdf(file: File) {
        val doc = PDDocument()
        val pagesContent = listOf(
            "Page 1: Title and Overview. Section 1: Introduction to Grafting. Grafting is a horticultural technique whereby tissues of plants are joined so as to continue their growth together. The upper part of the combined plant is called the scion while the lower part is called the rootstock.",
            "Page 2: Biological Mechanisms. Section 2: Vascular Cambium. For successful grafting to occur, the vascular cambium tissues of the rootstock and scion must be placed in contact with each other. Both tissues must be kept alive until the graft has taken, usually a period of several weeks.",
            "Page 3: Whip and Tongue Grafting. Section 3: Whip-and-Tongue Grafting. Method 1: Whip-and-tongue grafting is one of the oldest methods of grafting. It is commonly used for grafting fruit trees such as apples and pears where scion and rootstock have equal diameter.",
            "Page 4: Cleft Grafting. Section 4: Cleft Grafting. Method 2: Cleft grafting is used when the rootstock is much larger than the scion. The stock is split down the center and two scions are inserted into the cleft.",
            "Page 5: Bark Grafting. Section 5: Bark Grafting. Method 3: Bark grafting, also known as crown grafting, is used for topworking mature trees when the bark is slipping in early spring.",
            "Page 6: Bridge Grafting. Section 6: Bridge Grafting. Method 4: Bridge grafting is utilized to repair damaged trunks where girdling has destroyed the phloem and cambium layer.",
            "Page 7: Side-Veneer Grafting. Section 7: Side-Veneer Grafting. Method 5: Side-veneer grafting is a widely used technique for propagating conifers and small deciduous nursery stock.",
            "Page 8: Splice Grafting. Section 8: Splice Grafting. Method 6: Splice grafting is a simple technique where simple diagonal cuts are made on both scion and stock and taped together.",
            "Page 9: Saddle Grafting. Section 9: Saddle Grafting. Method 7: Saddle grafting is performed by cutting a V-notch in the rootstock and shaping the scion into a matching wedge or saddle.",
            "Page 10: Approach Grafting. Section 10: Approach Grafting. Method 8: Approach grafting involves joining two independent rooted plants together until a union forms, after which the top of the rootstock and base of the scion are removed.",
            "Page 11: Environmental Conditions and Optimal Temperature. Section 11: Environmental Factors and Callus Formation. The optimal healing temperature for grafting callus formation is 21C to 27C (70F to 80F). Humidity must be maintained above 85% to prevent desiccation.",
            "Page 12: Tools and Materials. Section 12: Grafting Equipment. Tools required include sharp grafting knives, budding strips, pruning shears, and grafting wax to seal the wound.",
            "Page 13: Aftercare and Maintenance. Section 13: Post-Graft Care. Inspect grafts weekly. Remove suckers emerging from the rootstock below the graft union to direct vigor into the scion.",
            "Page 14: Common Causes of Failure. Section 14: Troubleshooting Graft Failure. Failures typically result from poor cambial contact, desiccation of tissues, grafting during incorrect physiological seasons, or pathogen infection.",
            "Page 15: Summary and Glossary. Section 15: Reference Glossary. Scion: the detached shoot or bud. Rootstock: the rooted plant part providing root system. Graft union: the junction where cambium tissues fuse."
        )

        for (text in pagesContent) {
            val page = PDPage()
            doc.addPage(page)
            val stream = PDPageContentStream(doc, page)
            stream.beginText()
            stream.setFont(PDType1Font.HELVETICA_BOLD, 12f)
            stream.newLineAtOffset(50f, 700f)
            stream.showText(text.take(80))
            if (text.length > 80) {
                stream.setFont(PDType1Font.HELVETICA, 10f)
                stream.newLineAtOffset(0f, -20f)
                stream.showText(text.substring(80).take(80))
            }
            if (text.length > 160) {
                stream.newLineAtOffset(0f, -20f)
                stream.showText(text.substring(160).take(80))
            }
            if (text.length > 240) {
                stream.newLineAtOffset(0f, -20f)
                stream.showText(text.substring(240).take(80))
            }
            stream.endText()
            stream.close()
        }
        doc.save(file)
        doc.close()
    }

    private fun cosineSimilarity(v1: FloatArray, v2: FloatArray): Float {
        var dot = 0f
        var normA = 0f
        var normB = 0f
        for (i in v1.indices) {
            dot += v1[i] * v2[i]
            normA += v1[i] * v1[i]
            normB += v2[i] * v2[i]
        }
        return if (normA > 0f && normB > 0f) dot / (sqrt(normA) * sqrt(normB)) else 0f
    }

    private suspend fun ingestTestDocument(): Long {
        val extractedDoc = DocumentProcessor.extractDocument(context, Uri.fromFile(testPdfFile))
        val docId = database.documentDao().insert(
            DocumentEntity(
                uri = Uri.fromFile(testPdfFile).toString(),
                name = "Grafting_15_Page_Guide.pdf",
                type = "application/pdf",
                pageCount = 15,
                status = "READY"
            )
        )

        val documentChunks = TextChunker.chunkDocument(extractedDoc.pages, targetWords = 35, maxWords = 50)
        val allChunks = documentChunks.map { c ->
            val embedding = onnxEngine.getEmbedding(c.text)
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

        database.chunkDao().insertAll(allChunks)
        return docId
    }

    @Test
    fun test1_RealONNXEmbeddingEngine_OnDeviceVerification() = runBlocking {
        println("=== 1. VERIFY REAL ON-DEVICE ONNX EMBEDDINGS ===")
        val text1 = "Grafting is a horticultural technique for joining plant tissues."
        val text2 = "Plant propagation via scion and rootstock fusion."
        val text3 = "Quantum electrodynamics describes relativistic electromagnetic interactions."

        val startMs = System.currentTimeMillis()
        val emb1 = onnxEngine.getEmbedding(text1)
        val elapsedMs = System.currentTimeMillis() - startMs

        println("ONNX Embedding Dim: ${emb1.size}, latency: $elapsedMs ms")

        // 1. Must be exactly 384 dimensions
        assertEquals("Embedding dimension must be 384", 384, emb1.size)

        // 2. Must NOT be a fake zero vector
        val nonZeroCount = emb1.count { it != 0f }
        assertTrue("Embedding must have non-zero components (got $nonZeroCount non-zero)", nonZeroCount > 350)

        // 3. Must be L2 normalized (~1.0f)
        var normSq = 0f
        for (v in emb1) normSq += v * v
        val norm = sqrt(normSq)
        assertTrue("L2 norm must be approximately 1.0 (got $norm)", norm in 0.98f..1.02f)

        // 4. Semantic separation
        val emb2 = onnxEngine.getEmbedding(text2)
        val emb3 = onnxEngine.getEmbedding(text3)

        val simRelated = cosineSimilarity(emb1, emb2)
        val simUnrelated = cosineSimilarity(emb1, emb3)
        println("Similarity (Grafting vs Propagation): $simRelated")
        println("Similarity (Grafting vs Quantum): $simUnrelated")

        assertTrue("Related texts must have higher cosine similarity than unrelated texts ($simRelated vs $simUnrelated)", simRelated > simUnrelated)
        assertTrue("Related similarity must exceed 0.3", simRelated > 0.3f)
    }

    @Test
    fun test2_RealDocumentIngestionAndRetrieval() = runBlocking {
        println("=== 2. INGEST REAL 15-PAGE GRAFTING GUIDE PDF ===")
        assertTrue("Generated PDF must exist on device disk", testPdfFile.exists() && testPdfFile.length() > 1000)

        val docId = ingestTestDocument()
        val totalChunks = database.chunkDao().getAllChunksForDocumentIds(listOf(docId))
        println("Total Indexed Chunks in Room: ${totalChunks.size}")
        assertTrue("Must index at least 15 chunks (got ${totalChunks.size})", totalChunks.size >= 15)

        // Verify all chunks have valid non-null non-empty embeddings
        for (chunk in totalChunks) {
            assertNotNull("Chunk embedding must not be null", chunk.embedding)
            assertTrue("Chunk embedding must be 1536 bytes (384 floats * 4)", chunk.embedding!!.size == 384 * 4)
        }

        // Test Query A: "What is grafting?" -> Should retrieve Page 1
        val embA = onnxEngine.getEmbedding("What is grafting?")
        val retrievedA = vectorStore.retrieveDetailed(embA, listOf(docId), topK = 10, query = "What is grafting?")
        println("Retrieved pages for 'What is grafting?': ${retrievedA.map { "${it.chunk.pageNumber}(score=${it.relevanceScore})" }}")
        assertTrue("Retrieval A must return results", retrievedA.isNotEmpty())
        assertTrue("Top retrieval for 'What is grafting?' should include Page 1", retrievedA.any { it.chunk.pageNumber == 1 })

        // Test Query B: "What is the optimal healing temperature for grafting?" -> Should retrieve Page 11
        val embB = onnxEngine.getEmbedding("What is the optimal healing temperature for grafting?")
        val retrievedB = vectorStore.retrieveDetailed(embB, listOf(docId), topK = 5, query = "What is the optimal healing temperature for grafting?")
        assertTrue("Retrieval B must return results", retrievedB.isNotEmpty())
        assertTrue("Top retrieval for temperature must include Page 11", retrievedB.any { it.chunk.pageNumber == 11 })
        val tempChunk = retrievedB.first { it.chunk.pageNumber == 11 }
        assertTrue("Chunk from Page 11 must contain 21C or 27C", tempChunk.chunk.text.contains("21C") || tempChunk.chunk.text.contains("27C"))

        // Test Query C: Absent Fact Query -> Sentinel behavior
        val qClassificationC = QueryIntentRouter.route("What is the warranty period for grafting shears?")
        val embC = onnxEngine.getEmbedding("What is the warranty period for grafting shears?")
        val retrievedC = vectorStore.retrieveDetailed(embC, listOf(docId), topK = 5, query = "What is the warranty period for grafting shears?")
        val assembledC = ContextAssembler.assemble(
            retrievedResults = retrievedC,
            queryIntent = qClassificationC.intent,
            queryClassification = qClassificationC,
            maxEvidenceTokens = 800
        )
        assertNotNull("Assembled context must not be null", assembledC)
    }

    @Test
    fun test3_RealOnDeviceNeuralRAGInference() = runBlocking {
        println("=== 3. REAL ON-DEVICE NEURAL RAG INFERENCE ===")
        val docId = ingestTestDocument()

        val query = "What is the optimal healing temperature for grafting?"
        val qClassification = QueryIntentRouter.route(query)
        val qEmbedding = onnxEngine.getEmbedding(query)
        val retrieved = vectorStore.retrieveDetailed(
            queryEmbedding = qEmbedding,
            relevantDocumentIds = listOf(docId),
            topK = 2,
            query = query
        )

        val assembled = ContextAssembler.assemble(
            retrievedResults = retrieved,
            queryIntent = qClassification.intent,
            queryClassification = qClassification,
            maxEvidenceTokens = 300
        )

        val prompt = """
            Context:
            ${assembled.contextText}

            Question: $query
        """.trimIndent()

        val answer = llm.generate(prompt)
        println("\nQuestion: $query")
        println("Neural Grounded Answer: $answer")
        println("AI Engine State: ${llm.currentEngineState}")

        assertEquals("Engine state must be GGUF_NATIVE on Android", GGUFLLM.EngineState.GGUF_NATIVE, llm.currentEngineState)
        assertTrue("Answer must state the optimal temperature",
            answer.contains("21") || answer.contains("27") || answer.contains("70") || answer.contains("80"))
        assertTrue("Answer must cite Page 11", answer.contains("11") || answer.contains("Page"))
    }

    @Test
    fun test4_ClosedWorldAbsentFact_NoHallucination() = runBlocking {
        println("=== 4. CLOSED-WORLD ABSENT FACT EVALUATION ===")
        val query = "What is the warranty period for grafting shears?"
        val emptyPrompt = """
            Context:
            [NO_EVIDENCE_FOUND]

            Question: $query
        """.trimIndent()

        val answer = llm.generate(emptyPrompt)
        println("\nAbsent Query: $query")
        println("Answer: $answer")

        assertTrue("Answer must state evidence was not found",
            answer.contains("could not find", ignoreCase = true) ||
                    answer.contains("not found", ignoreCase = true) ||
                    answer.contains("not mentioned", ignoreCase = true))
        assertFalse("Answer must not hallucinate a warranty period",
            answer.contains("lifetime warranty", ignoreCase = true) ||
                    answer.contains("1 year warranty", ignoreCase = true))
    }

    @Test
    fun test5_PromptFormatAndContextBudgetVerification() {
        println("=== 5. PROMPT FORMAT AND CONTEXT BUDGET VERIFICATION ===")

        // 1. Verify Qwen ChatML prompt formatting
        val contextSnippet = "[SOURCE_ID=12]\n[PAGE=11]\nThe optimal healing temperature for grafting callus formation is 21C to 27C."
        val question = "What is the optimal healing temperature for grafting callus formation?"
        val prompt = llm.buildQwenPrompt(contextSnippet, question)

        println("Serialized Prompt Head:\n${prompt.take(150)}")
        println("Serialized Prompt Tail:\n${prompt.takeLast(100)}")

        assertTrue("Prompt must start with <|im_start|>system\\n", prompt.startsWith("<|im_start|>system\n"))
        assertTrue("System turn must end with <|im_end|>\\n", prompt.contains("<|im_end|>\n<|im_start|>user\n"))
        assertTrue("User turn must end with <|im_end|>\\n", prompt.contains("<|im_end|>\n<|im_start|>assistant\n"))
        assertTrue("Prompt must terminate with exact <|im_start|>assistant\\n", prompt.endsWith("<|im_start|>assistant\n"))
        assertFalse("Prompt must not inject leading BOS token", prompt.startsWith("<|endoftext|>"))

        // 2. Ingest 15 chunks from database to verify budgeting
        val chunks = runBlocking { database.chunkDao().getAllChunksForDocumentIds(listOf(1L)) }
        val retrieved = chunks.mapIndexed { idx, chunk ->
            com.siddu.offlinedocai.ai.retrieval.RetrievalResult(
                chunk = chunk,
                relevanceScore = 1.0f - (idx * 0.05f),
                source = com.siddu.offlinedocai.ai.retrieval.RetrievalSource.HYBRID_SEMANTIC,
                intent = com.siddu.offlinedocai.ai.retrieval.QueryIntent.CONCEPTUAL_QUERY
            )
        }

        // 3. Simple query budgeting (target ~384 tokens)
        val simpleContext = ContextAssembler.assemble(
            retrievedResults = retrieved,
            queryIntent = com.siddu.offlinedocai.ai.retrieval.QueryIntent.CONCEPTUAL_QUERY
        )
        println("Simple Query Assembled Tokens: ${simpleContext.totalEstimatedTokens} (Blocks: ${simpleContext.sourceBlocks.size})")
        assertTrue("Simple query context must be constrained within budget (384 + safety)",
            simpleContext.totalEstimatedTokens <= ContextAssembler.BUDGET_SIMPLE_QUERY_TOKENS + 35)

        // 4. Exhaustive query budgeting (uses full 1152 tokens)
        val exhaustiveContext = ContextAssembler.assemble(
            retrievedResults = retrieved,
            queryIntent = com.siddu.offlinedocai.ai.retrieval.QueryIntent.EXHAUSTIVE_LIST
        )
        println("Exhaustive Query Assembled Tokens: ${exhaustiveContext.totalEstimatedTokens} (Blocks: ${exhaustiveContext.sourceBlocks.size})")
        assertTrue("Exhaustive query must retain significantly more blocks than simple query",
            exhaustiveContext.sourceBlocks.size > simpleContext.sourceBlocks.size)
    }
}
