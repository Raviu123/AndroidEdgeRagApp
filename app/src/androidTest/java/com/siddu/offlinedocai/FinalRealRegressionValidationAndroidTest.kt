package com.siddu.offlinedocai

import android.content.Context
import android.net.Uri
import androidx.room.Room
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.siddu.offlinedocai.ai.chunking.TextChunker
import com.siddu.offlinedocai.ai.embeddings.ONNXEmbeddingEngine
import com.siddu.offlinedocai.ai.llm.AnswerLength
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

@RunWith(AndroidJUnit4::class)
class FinalRealRegressionValidationAndroidTest {

    private lateinit var context: Context
    private lateinit var database: AppDatabase
    private lateinit var onnxEngine: ONNXEmbeddingEngine
    private lateinit var vectorStore: VectorStore
    private lateinit var llm: GGUFLLM
    private lateinit var testPdfFile: File
    private var docId: Long = 0L

    data class ValidationQuery(
        val code: String,
        val text: String,
        val expectedPages: List<Int>,
        val primaryHeadingToReject: List<String> = emptyList()
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

        val pushedFile = File("/data/local/tmp/Grafting_15_Page_Guide.pdf")
        testPdfFile = File(context.cacheDir, "Grafting_15_Page_Guide.pdf")
        if (pushedFile.exists() && pushedFile.canRead() && pushedFile.length() > 5000) {
            pushedFile.copyTo(testPdfFile, overwrite = true)
        } else {
            createDeterministicGraftingPdf(testPdfFile)
        }
        assertTrue("Test PDF must exist on device disk", testPdfFile.exists() && testPdfFile.length() > 5000)
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

    private fun createDeterministicGraftingPdf(file: File) {
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

    @Test
    fun testRealAndroidRegressionValidation() = runBlocking {
        println("================================================================================")
        println("[REGRESSION_VAL] STARTING FINAL REAL ANDROID REGRESSION VALIDATION")
        println("================================================================================")

        // 1. Ingestion Pipeline Verification
        println("\n[REGRESSION_VAL] 1. INGESTION PIPELINE VERIFICATION")
        println("[REGRESSION_VAL] PDF File Path: ${testPdfFile.absolutePath}")
        println("[REGRESSION_VAL] PDF File Size: ${testPdfFile.length()} bytes")

        val extractedDoc = DocumentProcessor.extractDocument(context, Uri.fromFile(testPdfFile))
        val totalWords = extractedDoc.pages.sumOf { it.text.split(Regex("\\s+")).filter { w -> w.isNotBlank() }.size }
        println("[REGRESSION_VAL] PDFBox Extracted Pages: ${extractedDoc.pageCount}")
        println("[REGRESSION_VAL] PDFBox Extracted Total Words: $totalWords")
        assertEquals(15, extractedDoc.pageCount)

        docId = database.documentDao().insert(
            DocumentEntity(
                uri = Uri.fromFile(testPdfFile).toString(),
                name = "Grafting_15_Page_Guide.pdf",
                type = "application/pdf",
                pageCount = 15,
                status = "READY"
            )
        )
        println("[REGRESSION_VAL] Document Inserted to Room: docId=$docId")

        val documentChunks = TextChunker.chunkDocument(extractedDoc.pages, targetWords = 35, maxWords = 50)
        println("[REGRESSION_VAL] TextChunker Chunks Created: ${documentChunks.size}")
        assertTrue("Must create at least 15 chunks", documentChunks.size >= 15)

        val allChunks = documentChunks.map { c ->
            val embedding = onnxEngine.getEmbedding(c.text)
            assertEquals(384, embedding.size)
            val nonZero = embedding.count { it != 0f }
            assertTrue(nonZero > 350)
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
        val insertedChunks = database.chunkDao().getAllChunksForDocumentIds(listOf(docId))
        println("[REGRESSION_VAL] Room Persistence Verified: ${insertedChunks.size} chunks persisted in SQLite")
        assertEquals(documentChunks.size, insertedChunks.size)

        // Verify Model
        val modelFile = llm.resolveModelFile()
        println("\n[REGRESSION_VAL] MODEL VERIFICATION:")
        println("[REGRESSION_VAL] Active Model Name: ${llm.getActiveModel()}")
        println("[REGRESSION_VAL] Model File Path: ${modelFile?.absolutePath}")
        println("[REGRESSION_VAL] Model File Size: ${modelFile?.length()} bytes")
        assertNotNull(modelFile)

        val questions = listOf(
            ValidationQuery(
                code = "A",
                text = "What are the likely causes of grafting failure and what should I check?",
                expectedPages = listOf(14),
                primaryHeadingToReject = listOf(
                    "Failure Likely causes",
                    "What to check",
                    "Section 14: Troubleshooting Graft Failure",
                    "Page 14: Common Causes of Failure"
                )
            ),
            ValidationQuery(
                code = "B",
                text = "What is grafting?",
                expectedPages = listOf(1),
                primaryHeadingToReject = listOf(
                    "Page 1: Title and Overview",
                    "Section 1: Introduction to Grafting"
                )
            ),
            ValidationQuery(
                code = "C",
                text = "What are the types of grafting?",
                expectedPages = (3..10).toList(),
                primaryHeadingToReject = listOf(
                    "Page 3: Whip and Tongue Grafting",
                    "Section 3: Whip-and-Tongue Grafting"
                )
            ),
            ValidationQuery(
                code = "D",
                text = "What tools and materials are needed?",
                expectedPages = listOf(12),
                primaryHeadingToReject = listOf(
                    "Page 12: Tools and Materials",
                    "Section 12: Grafting Equipment"
                )
            ),
            ValidationQuery(
                code = "E",
                text = "What temperature is recommended for healing?",
                expectedPages = listOf(11),
                primaryHeadingToReject = listOf(
                    "Page 11: Environmental Conditions",
                    "Section 11: Environmental Factors"
                )
            ),
            ValidationQuery(
                code = "F",
                text = "List all methods of grafting described in the guide.",
                expectedPages = (3..10).toList(),
                primaryHeadingToReject = listOf(
                    "Method 1: Whip-and-tongue",
                    "Method 2: Cleft"
                )
            )
        )

        println("\n================================================================================")
        println("[REGRESSION_VAL] RUNNING REAL ANDROID QUERY VALIDATION (SHORT & LONG MODES)")
        println("================================================================================")

        for (q in questions) {
            println("\n--------------------------------------------------------------------------------")
            println("[REGRESSION_VAL] QUESTION ${q.code}: \"${q.text}\"")
            println("--------------------------------------------------------------------------------")

            // Retrieval & Intent (Short and Long must use identical retrieval)
            val classification = QueryIntentRouter.route(q.text)
            val qEmbedding = onnxEngine.getEmbedding(q.text)
            val retrieved = vectorStore.retrieveDetailed(
                queryEmbedding = qEmbedding,
                relevantDocumentIds = listOf(docId),
                topK = 20,
                query = q.text
            )

            println("[REGRESSION_VAL] Intent: ${classification.intent}")
            println("[REGRESSION_VAL] Retrieved Chunks Count: ${retrieved.size}")
            println("[REGRESSION_VAL] Retrieved Pages: ${retrieved.map { it.chunk.pageNumber }}")
            println("[REGRESSION_VAL] Top Retrieved Page: ${retrieved.firstOrNull()?.chunk?.pageNumber} (Score: ${retrieved.firstOrNull()?.relevanceScore})")

            val evidenceBudget = ContextAssembler.resolveEvidenceBudget(classification.intent)
            val assembled = ContextAssembler.assemble(
                retrievedResults = retrieved,
                queryIntent = classification.intent,
                queryClassification = classification,
                maxEvidenceTokens = evidenceBudget
            )
            println("[REGRESSION_VAL] Assembled Context Blocks: ${assembled.sourceBlocks.size} (${assembled.totalEstimatedTokens} tokens)")

            // Test SHORT mode
            val shortPrompt = llm.buildPromptForModel(llm.getActiveModel(), assembled.contextText, q.text, null, AnswerLength.SHORT)
            val shortStart = System.currentTimeMillis()
            val shortAnswer = llm.generate(shortPrompt, AnswerLength.SHORT)
            val shortTime = System.currentTimeMillis() - shortStart
            val shortEngineState = llm.currentEngineState

            println("\n[REGRESSION_VAL] [MODE: SHORT]")
            println("[REGRESSION_VAL] Engine State: $shortEngineState")
            println("[REGRESSION_VAL] Generation Latency: ${shortTime}ms")
            println("[REGRESSION_VAL] Short Answer:\n$shortAnswer")

            val shortValidation = CitationValidator.validateWithSourceBlocks(
                answer = shortAnswer,
                sourceBlocks = assembled.sourceBlocks,
                query = q.text,
                isBudgetLimited = assembled.isBudgetLimited
            )
            println("[REGRESSION_VAL] Short Citations: ${shortValidation.validCitations.map { "P${it.page}" }}")
            println("[REGRESSION_VAL] Short Supported Ratio: ${shortValidation.overallSupportedRatio}")
            println("[REGRESSION_VAL] Short Unsupported Claims Count: ${shortValidation.claimResults.count { it.status != com.siddu.offlinedocai.ai.validation.ValidationStatus.SUPPORTED }}")

            // Test LONG mode
            val longPrompt = llm.buildPromptForModel(llm.getActiveModel(), assembled.contextText, q.text, null, AnswerLength.LONG)
            val longStart = System.currentTimeMillis()
            val longAnswer = llm.generate(longPrompt, AnswerLength.LONG)
            val longTime = System.currentTimeMillis() - longStart
            val longEngineState = llm.currentEngineState

            println("\n[REGRESSION_VAL] [MODE: LONG]")
            println("[REGRESSION_VAL] Engine State: $longEngineState")
            println("[REGRESSION_VAL] Generation Latency: ${longTime}ms")
            println("[REGRESSION_VAL] Long Answer:\n$longAnswer")

            val longValidation = CitationValidator.validateWithSourceBlocks(
                answer = longAnswer,
                sourceBlocks = assembled.sourceBlocks,
                query = q.text,
                isBudgetLimited = assembled.isBudgetLimited
            )
            println("[REGRESSION_VAL] Long Citations: ${longValidation.validCitations.map { "P${it.page}" }}")
            println("[REGRESSION_VAL] Long Supported Ratio: ${longValidation.overallSupportedRatio}")
            println("[REGRESSION_VAL] Long Unsupported Claims Count: ${longValidation.claimResults.count { it.status != com.siddu.offlinedocai.ai.validation.ValidationStatus.SUPPORTED }}")

            // Specific Assertions for Question A
            if (q.code == "A") {
                assertFalse(
                    "Short answer for Question A must NOT begin with malformed heading",
                    shortAnswer.trim().startsWith("Failure Likely causes", ignoreCase = true)
                )
                assertFalse(
                    "Short answer for Question A must NOT contain malformed heading",
                    shortAnswer.contains("Failure Likely causes / What to check", ignoreCase = true)
                )
                assertFalse(
                    "Short answer for Question A must NOT start with raw heading",
                    shortAnswer.trim().startsWith("Page 14: Common Causes", ignoreCase = true) ||
                    shortAnswer.trim().startsWith("Section 14:", ignoreCase = true)
                )
                assertFalse(
                    "Long answer for Question A must NOT begin with malformed heading",
                    longAnswer.trim().startsWith("Failure Likely causes", ignoreCase = true)
                )
                assertFalse(
                    "Long answer for Question A must NOT contain malformed heading",
                    longAnswer.contains("Failure Likely causes / What to check", ignoreCase = true)
                )
                assertFalse(
                    "Long answer for Question A must NOT start with raw heading",
                    longAnswer.trim().startsWith("Page 14: Common Causes", ignoreCase = true) ||
                    longAnswer.trim().startsWith("Section 14:", ignoreCase = true)
                )

                // Must contain explanatory causes
                val hasExplanatory = longAnswer.contains("cambial contact", ignoreCase = true) ||
                        longAnswer.contains("desiccation", ignoreCase = true) ||
                        longAnswer.contains("pathogen", ignoreCase = true) ||
                        longAnswer.contains("causes", ignoreCase = true)
                assertTrue("Question A answer must contain explanatory content", hasExplanatory)
            }

            // Specific Assertions for Question D (Tools & Materials relevance)
            if (q.code == "D") {
                assertFalse(
                    "Question D LONG answer must NOT contain 'inspect grafts weekly' (irrelevant aftercare)",
                    longAnswer.contains("inspect grafts weekly", ignoreCase = true)
                )
                assertFalse(
                    "Question D LONG answer must NOT contain 'remove suckers' (irrelevant aftercare)",
                    longAnswer.contains("remove suckers", ignoreCase = true)
                )
                assertFalse(
                    "Question D SHORT answer must NOT contain 'inspect grafts weekly'",
                    shortAnswer.contains("inspect grafts weekly", ignoreCase = true)
                )
                assertTrue(
                    "Question D must contain tools",
                    shortAnswer.contains("knives", ignoreCase = true) || shortAnswer.contains("shears", ignoreCase = true) || shortAnswer.contains("wax", ignoreCase = true)
                )
            }

            // Specific Assertions for Question E (Temperature relevance)
            if (q.code == "E") {
                assertFalse(
                    "Question E LONG answer must NOT contain tools",
                    longAnswer.contains("sharp grafting knives", ignoreCase = true) || longAnswer.contains("pruning shears", ignoreCase = true)
                )
                assertTrue(
                    "Question E answer must contain temperature figures (21C-27C / 70F-80F)",
                    longAnswer.contains("21", ignoreCase = true) || longAnswer.contains("70", ignoreCase = true)
                )
            }

            // Specific Assertions for Question C & F (Grafting methods completeness)
            if (q.code == "C" || q.code == "F") {
                val methodKeywords = listOf("whip", "cleft", "bark", "bridge", "side-veneer", "splice", "saddle", "approach")
                for (m in methodKeywords) {
                    assertTrue("SHORT answer must include method '$m'", shortAnswer.contains(m, ignoreCase = true))
                    assertTrue("LONG answer must include method '$m'", longAnswer.contains(m, ignoreCase = true))
                }
            }

            // Universal Numbering & List Quality Assertions
            assertFalse("Short answer must NOT contain empty number sequence '1. 2. 3. 4.'", shortAnswer.contains("1. 2. 3. 4."))
            assertFalse("Long answer must NOT contain empty number sequence '1. 2. 3. 4.'", longAnswer.contains("1. 2. 3. 4."))
            assertFalse("Long answer must NOT contain empty number sequence '2. 3. 4.'", longAnswer.contains("2. 3. 4."))
            assertFalse("Long answer must NOT contain empty number sequence '1. 2. 3.'", longAnswer.contains("1. 2. 3."))

            // Verify both answers are grounded and not blank
            assertTrue("Short answer must not be blank", shortAnswer.isNotBlank())
            assertTrue("Long answer must not be blank", longAnswer.isNotBlank())

            // Verify no unsupported out-of-syllabus botany inventions
            val outOfSyllabusTerms = listOf("photosynthesis", "chloroplast", "hydroponics", "mitochondria", "falcon heavy")
            for (term in outOfSyllabusTerms) {
                assertFalse("Short answer must not contain out-of-syllabus term: $term", shortAnswer.contains(term, ignoreCase = true))
                assertFalse("Long answer must not contain out-of-syllabus term: $term", longAnswer.contains(term, ignoreCase = true))
            }
        }

        println("\n================================================================================")
        println("[REGRESSION_VAL] ALL 6 BENCHMARK QUESTIONS VALIDATED SUCCESSFULLY ON REAL ANDROID")
        println("================================================================================")
    }
}
