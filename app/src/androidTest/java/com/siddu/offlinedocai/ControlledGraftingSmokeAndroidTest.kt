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

/**
 * Controlled End-to-End Smoke Test using Grafting_15_Page_Guide.pdf
 * and the production Qwen2.5-1.5B-Instruct-GGUF pipeline.
 *
 * Verifies:
 * 1. Real PDF extraction
 * 2. Real all-MiniLM-L6-v2 ONNX 384D embeddings
 * 3. Room database persistence
 * 4. Real hybrid retrieval (dense cosine + lexical BM25 + exact matching)
 * 5. Grounded context assembly
 * 6. Real Qwen2.5-1.5B GGUF inference via llama.cpp JNI
 * 7. Verification of the 3 required queries:
 *    - "What is grafting?"
 *    - "What temperature is recommended for healing?"
 *    - "Does the document mention Falcon Heavy?"
 */
@RunWith(AndroidJUnit4::class)
class ControlledGraftingSmokeAndroidTest {

    private lateinit var context: Context
    private lateinit var database: AppDatabase
    private lateinit var onnxEngine: ONNXEmbeddingEngine
    private lateinit var vectorStore: VectorStore
    private lateinit var llm: GGUFLLM
    private lateinit var testPdfFile: File
    private var docId: Long = 0L

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

        testPdfFile = File(context.cacheDir, "Grafting_15_Page_Guide.pdf")
        createGraftingGuidePdf(testPdfFile)

        docId = ingestTestDocument()
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

    private fun createGraftingGuidePdf(file: File) {
        val doc = PDDocument()
        val pagesContent = listOf(
            "Page 1: Title and Overview. Section 1: Introduction to Grafting.\nGrafting is a horticultural technique whereby tissues of plants are joined so as to continue their growth together.\nThe upper part of the combined plant is called the scion while the lower part is called the rootstock.",
            "Page 2: Biological Mechanisms. Section 2: Vascular Cambium.\nFor successful grafting to occur, the vascular cambium tissues of the rootstock and scion must be placed in contact with each other.\nBoth tissues must be kept alive until the graft has taken, usually a period of several weeks.",
            "Page 3: Whip and Tongue Grafting. Section 3: Whip-and-Tongue Grafting.\nMethod 1: Whip-and-tongue grafting is one of the oldest methods of grafting.\nIt is commonly used for grafting fruit trees such as apples and pears where scion and rootstock have equal diameter.",
            "Page 4: Cleft Grafting. Section 4: Cleft Grafting.\nMethod 2: Cleft grafting is used when the rootstock is much larger than the scion.\nThe stock is split down the center and two scions are inserted into the cleft.",
            "Page 5: Bark Grafting. Section 5: Bark Grafting.\nMethod 3: Bark grafting, also known as crown grafting, is used for topworking mature trees when the bark is slipping in early spring.",
            "Page 6: Bridge Grafting. Section 6: Bridge Grafting.\nMethod 4: Bridge grafting is utilized to repair damaged trunks where girdling has destroyed the phloem and cambium layer.",
            "Page 7: Side-Veneer Grafting. Section 7: Side-Veneer Grafting.\nMethod 5: Side-veneer grafting is a widely used technique for propagating conifers and small deciduous nursery stock.",
            "Page 8: Splice Grafting. Section 8: Splice Grafting.\nMethod 6: Splice grafting is a simple technique where simple diagonal cuts are made on both scion and stock and taped together.",
            "Page 9: Saddle Grafting. Section 9: Saddle Grafting.\nMethod 7: Saddle grafting is performed by cutting a V-notch in the rootstock and shaping the scion into a matching wedge or saddle.",
            "Page 10: Approach Grafting. Section 10: Approach Grafting.\nMethod 8: Approach grafting involves joining two independent rooted plants together until a union forms, after which the top of the rootstock and base of the scion are removed.",
            "Page 11: Environmental Conditions and Optimal Temperature. Section 11: Environmental Factors and Callus Formation.\nThe optimal healing temperature for grafting callus formation is 21C to 27C (70F to 80F).\nHumidity must be maintained above 85% to prevent desiccation.",
            "Page 12: Tools and Materials. Section 12: Grafting Equipment.\nTools required include sharp grafting knives, budding strips, pruning shears, and grafting wax to seal the wound.",
            "Page 13: Aftercare and Maintenance. Section 13: Post-Graft Care.\nInspect grafts weekly.\nRemove suckers emerging from the rootstock below the graft union to direct vigor into the scion.",
            "Page 14: Common Causes of Failure. Section 14: Troubleshooting Graft Failure.\nFailures typically result from poor cambial contact, desiccation of tissues, grafting during incorrect physiological seasons, or pathogen infection.",
            "Page 15: Summary and Glossary. Section 15: Reference Glossary.\nScion: the detached shoot or bud.\nRootstock: the rooted plant part providing root system.\nGraft union: the junction where cambium tissues fuse."
        )

        for (pageText in pagesContent) {
            val page = PDPage()
            doc.addPage(page)
            val stream = PDPageContentStream(doc, page)
            stream.beginText()
            stream.setFont(PDType1Font.HELVETICA_BOLD, 12f)
            stream.newLineAtOffset(50f, 700f)
            val lines = pageText.lines()
            for ((lineIdx, line) in lines.withIndex()) {
                if (lineIdx > 0) {
                    stream.setFont(PDType1Font.HELVETICA, 10f)
                    stream.newLineAtOffset(0f, -20f)
                }
                stream.showText(line)
            }
            stream.endText()
            stream.close()
        }
        doc.save(file)
        doc.close()
    }

    private suspend fun ingestTestDocument(): Long {
        val extractedDoc = DocumentProcessor.extractDocument(context, Uri.fromFile(testPdfFile))
        val id = database.documentDao().insert(
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
                documentId = id,
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
        return id
    }

    @Test
    fun testControlledGraftingSmokeQueries() = runBlocking {
        println("==================================================")
        println("OFFLINEDOCAI — CONTROLLED END-TO-END SMOKE TEST")
        println("==================================================")

        // Verify Model Selection
        assertEquals("Production model must be QWEN_1_5B", GGUFLLM.SupportedModel.QWEN_1_5B, llm.getActiveModel())
        val modelFile = llm.resolveModelFile()
        assertNotNull("Model file must be found", modelFile)
        println("Production Model File: ${modelFile!!.name} (${modelFile.length()} bytes)")
        assertTrue("Model must be Qwen 1.5B GGUF", modelFile.name.contains("1.5b", ignoreCase = true))

        val queries = listOf(
            "What is grafting?",
            "What temperature is recommended for healing?",
            "Does the document mention Falcon Heavy?"
        )

        for ((idx, query) in queries.withIndex()) {
            println("\n--- Query ${idx + 1}: \"$query\" ---")
            val startTime = System.currentTimeMillis()

            val qClassification = QueryIntentRouter.route(query)
            println("Query Intent: ${qClassification.intent}")

            val qEmbedding = onnxEngine.getEmbedding(query)
            val retrieved = vectorStore.retrieveDetailed(
                queryEmbedding = qEmbedding,
                relevantDocumentIds = listOf(docId),
                topK = 2,
                query = query
            )
            println("Retrieved ${retrieved.size} candidate chunks: ${retrieved.map { "P${it.chunk.pageNumber}(${it.relevanceScore})" }}")

            val budget = ContextAssembler.resolveEvidenceBudget(qClassification.intent)
            val assembled = ContextAssembler.assemble(
                retrievedResults = retrieved,
                queryIntent = qClassification.intent,
                queryClassification = qClassification,
                maxEvidenceTokens = minOf(budget, 120)
            )
            println("Assembled context: ${assembled.sourceBlocks.size} blocks (${assembled.totalEstimatedTokens} est tokens)")

            val prompt = """
                Context:
                ${assembled.contextText}

                Question: $query
            """.trimIndent()

            val answer = llm.generate(prompt)
            val elapsed = System.currentTimeMillis() - startTime

            println("Engine State: ${llm.currentEngineState}")
            println("Response Time: ${elapsed}ms")
            println("Generated Answer:\n$answer")

            // Real native inference check
            assertEquals(
                "Execution must use real native JNI inference on device",
                GGUFLLM.EngineState.GGUF_NATIVE,
                llm.currentEngineState
            )
            assertTrue("Answer must not be blank", answer.isNotBlank())

            when (idx) {
                0 -> {
                    // "What is grafting?"
                    assertTrue("Must have retained source blocks", assembled.sourceBlocks.isNotEmpty())
                    assertEquals("Top retained chunk must be Page 1 (Introduction/Definition)", 1, assembled.sourceBlocks.first().pageNumber)
                    assertTrue(
                        "Answer must define grafting or plant tissues",
                        answer.contains("grafting", ignoreCase = true) ||
                                answer.contains("horticultural", ignoreCase = true) ||
                                answer.contains("tissue", ignoreCase = true) ||
                                answer.contains("plant", ignoreCase = true)
                    )
                }
                1 -> {
                    // "What temperature is recommended for healing?"
                    assertTrue("Must have retained source blocks", assembled.sourceBlocks.isNotEmpty())
                    assertEquals("Top retained chunk must be Page 11 (Optimal Temperature)", 11, assembled.sourceBlocks.first().pageNumber)
                    assertTrue(
                        "Answer must mention 21 or 27 or 70 or 80 degrees",
                        answer.contains("21") || answer.contains("27") || answer.contains("70") || answer.contains("80")
                    )
                }
                2 -> {
                    // "Does the document mention Falcon Heavy?"
                    assertTrue("Context must be empty or no evidence for non-existent topic", assembled.sourceBlocks.isEmpty() || assembled.contextText.contains("NO_EVIDENCE_FOUND"))
                    assertTrue(
                        "Answer must state not found or no mention",
                        answer.contains("could not find", ignoreCase = true) ||
                                answer.contains("not found", ignoreCase = true) ||
                                answer.contains("not mention", ignoreCase = true) ||
                                answer.contains("no", ignoreCase = true) ||
                                answer.contains("does not", ignoreCase = true)
                    )
                    assertFalse(
                        "Answer must not fabricate Falcon Heavy facts",
                        answer.contains("rocket", ignoreCase = true) || answer.contains("SpaceX", ignoreCase = true)
                    )
                }
            }
        }

        println("\n==================================================")
        println("ALL THREE CONTROLLED SMOKE QUERIES PASSED VIA REAL NATIVE QWEN 1.5B!")
        println("==================================================")
    }
}
