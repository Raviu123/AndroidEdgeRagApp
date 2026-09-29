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
 * Phase 8 Step 2B: Controlled Alternative LLM Benchmark on Android.
 *
 * Compares:
 * - Model A: Qwen 2.5 0.5B Instruct Q4_K_M (491,400,032 bytes)
 * - Model B: Llama 3.2 1B Instruct Q4_K_M (807,694,464 bytes)
 */
@RunWith(AndroidJUnit4::class)
class Step2BModelBenchmarkAndroidTest {

    private lateinit var context: Context
    private lateinit var database: AppDatabase
    private lateinit var onnxEngine: ONNXEmbeddingEngine
    private lateinit var vectorStore: VectorStore
    private lateinit var llm: GGUFLLM
    private lateinit var testPdfFile: File
    private var docId: Long = 0L

    @Before
    fun setup() {
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

        runBlocking {
            docId = ingestTestDocument()
        }
    }

    @After
    fun tearDown() {
        llm.llamaRunner.freeModel()
        database.close()
        if (testPdfFile.exists()) testPdfFile.delete()
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

    private fun cleanTokens(rawOutput: String): String {
        return rawOutput
            .replace("<|im_end|>", "")
            .replace("<|im_start|>", "")
            .replace("<|endoftext|>", "")
            .replace("<|eot_id|>", "")
            .replace("<|start_header_id|>", "")
            .replace("<|end_header_id|>", "")
            .replace("<|begin_of_text|>", "")
            .replace("<|end_of_text|>", "")
            .trim()
            .removePrefix(":")
            .trim()
    }

    @Test
    fun test01_ShortInstructionBenchmark() {
        println("=== BENCHMARK: SHORT INSTRUCTION TESTS ===")

        val models = listOf(GGUFLLM.SupportedModel.QWEN_0_5B, GGUFLLM.SupportedModel.LLAMA_1B)
        val shortConfigs = listOf(
            Pair("Test A", "Answer with exactly one word: APPLE"),
            Pair("Test B", "Answer with exactly one word: ORANGE"),
            Pair("Test C", "In one short sentence, define grafting.")
        )

        for (m in models) {
            println("\n--- LOADING MODEL: ${m.displayName} ---")
            val resolved = llm.resolveModelFile(m)
            assertNotNull("Model file resolved", resolved)

            val runner = llm.llamaRunner
            runner.freeModel()

            val tLoadStart = System.currentTimeMillis()
            val loadOk = runner.loadModel(resolved!!.absolutePath, GGUFLLM.SAFE_CONTEXT_SIZE, 4)
            val loadMs = System.currentTimeMillis() - tLoadStart
            assertTrue("Load ok", loadOk)

            println("BENCHMARK_REPORT: MODEL=${m.displayName} LOAD_RESULT=SUCCESS LOAD_TIME_MS=$loadMs FILE_SIZE=${resolved.length()}")

            for ((testName, instruction) in shortConfigs) {
                val formatted = when (m) {
                    GGUFLLM.SupportedModel.QWEN_1_5B,
                    GGUFLLM.SupportedModel.QWEN_0_5B ->
                        "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\n$instruction\n<|im_end|>\n<|im_start|>assistant\n"
                    GGUFLLM.SupportedModel.LLAMA_1B ->
                        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n$instruction<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                }

                val tGenStart = System.currentTimeMillis()
                val rawOutput = runner.inference(formatted, maxTokens = 16, temperature = 0.1f)
                val genMs = System.currentTimeMillis() - tGenStart
                val clean = cleanTokens(rawOutput)
                val tokenEstimate = maxOf(1, clean.split(Regex("""\s+""")).filter { it.isNotBlank() }.size)
                val tps = if (genMs > 0) (tokenEstimate.toDouble() / (genMs.toDouble() / 1000.0)) else 0.0

                println("BENCHMARK_REPORT: SHORT_TEST=$testName MODEL=${m.displayName} GEN_MS=$genMs TOKENS=$tokenEstimate TPS=${String.format(java.util.Locale.US, "%.2f", tps)} OUTPUT='$clean'")
            }

            runner.freeModel()
            println("BENCHMARK_REPORT: MODEL=${m.displayName} FREED=SUCCESS")
        }
    }

    @Test
    fun test02_RAG_Benchmark() = runBlocking {
        println("=== BENCHMARK: RAG TESTS ON 15-PAGE GUIDE ===")

        val ragQuestions = listOf(
            Pair(1, "What is grafting?"),
            Pair(2, "What is the optimal healing temperature for grafting callus formation?"),
            Pair(3, "Where is grafting discussed in the document?"),
            Pair(4, "List all types of grafting mentioned in the document."),
            Pair(5, "Summarize the section about grafting."),
            Pair(6, "What is the relationship between the relevant grafting techniques described in the document?"),
            Pair(7, "What humidity is required to prevent desiccation during grafting callus formation?"),
            Pair(8, "What does the guide say about grafting avocado trees?")
        )

        // Precompute retrieval for all questions so it is 100% identical between models
        data class RetrievalBundle(
            val query: String,
            val intent: String,
            val chunkIds: List<Long>,
            val pages: List<Int>,
            val contextText: String,
            val contextTokens: Int,
            val contextChars: Int
        )

        val bundles = ragQuestions.map { (_, query) ->
            val qClassification = QueryIntentRouter.route(query)
            val qEmbedding = onnxEngine.getEmbedding(query)
            val retrieved = vectorStore.retrieveDetailed(
                queryEmbedding = qEmbedding,
                relevantDocumentIds = listOf(docId),
                topK = 3,
                query = query
            )
            val assembled = ContextAssembler.assemble(
                retrievedResults = retrieved,
                queryIntent = qClassification.intent,
                queryClassification = qClassification,
                maxEvidenceTokens = 250
            )

            RetrievalBundle(
                query = query,
                intent = qClassification.intent.name,
                chunkIds = retrieved.map { it.chunk.id },
                pages = retrieved.map { it.chunk.pageNumber }.distinct().sorted(),
                contextText = assembled.contextText,
                contextTokens = assembled.totalEstimatedTokens,
                contextChars = assembled.contextText.length
            )
        }

        // Verify retrieval bundle consistency
        println("BENCHMARK_REPORT: RETRIEVAL_CONSISTENCY=IDENTICAL TOTAL_QUERIES=${bundles.size}")

        val models = listOf(GGUFLLM.SupportedModel.QWEN_0_5B, GGUFLLM.SupportedModel.LLAMA_1B)

        for (m in models) {
            println("\n--- RUNNING RAG BENCHMARK FOR: ${m.displayName} ---")
            val resolved = llm.resolveModelFile(m)
            assertNotNull("Model resolved", resolved)

            val runner = llm.llamaRunner
            runner.freeModel()

            val tLoadStart = System.currentTimeMillis()
            val loadOk = runner.loadModel(resolved!!.absolutePath, GGUFLLM.SAFE_CONTEXT_SIZE, 4)
            val loadMs = System.currentTimeMillis() - tLoadStart
            assertTrue("Load ok", loadOk)

            for ((idx, bundle) in bundles.withIndex()) {
                val qNum = idx + 1
                val prompt = when (m) {
                    GGUFLLM.SupportedModel.QWEN_1_5B,
                    GGUFLLM.SupportedModel.QWEN_0_5B -> llm.buildQwenPrompt(bundle.contextText, bundle.query)
                    GGUFLLM.SupportedModel.LLAMA_1B -> llm.buildLlamaPrompt(bundle.contextText, bundle.query)
                }

                println("BENCHMARK_REPORT: RAG_DIAGNOSTIC Q=$qNum INTENT=${bundle.intent} CHUNKS=${bundle.chunkIds} PAGES=${bundle.pages} CTX_TOKENS=${bundle.contextTokens} CTX_CHARS=${bundle.contextChars}")

                val tStart = System.currentTimeMillis()
                val rawOutput = runner.inference(prompt, maxTokens = 36, temperature = 0.1f)
                val genMs = System.currentTimeMillis() - tStart
                val clean = cleanTokens(rawOutput)
                val tokenEstimate = maxOf(1, clean.split(Regex("""\s+""")).filter { it.isNotBlank() }.size)
                val tps = if (genMs > 0) (tokenEstimate.toDouble() / (genMs.toDouble() / 1000.0)) else 0.0

                println("BENCHMARK_REPORT: RAG_OUTPUT Q=$qNum MODEL=${m.displayName} GEN_MS=$genMs TOKENS=$tokenEstimate TPS=${String.format(java.util.Locale.US, "%.2f", tps)} ANSWER='$clean'")
            }

            runner.freeModel()
            println("BENCHMARK_REPORT: MODEL=${m.displayName} FREED=SUCCESS")
        }
    }
}
