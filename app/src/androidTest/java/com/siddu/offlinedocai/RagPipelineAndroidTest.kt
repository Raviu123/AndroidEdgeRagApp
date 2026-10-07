package com.siddu.offlinedocai

import android.content.Context
import android.net.Uri
import androidx.room.Room
import androidx.sqlite.db.SupportSQLiteDatabase
import androidx.sqlite.db.SupportSQLiteOpenHelper
import androidx.sqlite.db.framework.FrameworkSQLiteOpenHelperFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.siddu.offlinedocai.ai.chunking.TextChunker
import com.siddu.offlinedocai.ai.embeddings.ONNXEmbeddingEngine
import com.siddu.offlinedocai.ai.llm.AnswerLength
import com.siddu.offlinedocai.ai.llm.GGUFLLM
import com.siddu.offlinedocai.ai.retrieval.ContextAssembler
import com.siddu.offlinedocai.ai.retrieval.QueryIntentRouter
import com.siddu.offlinedocai.ai.retrieval.SearchQuery
import com.siddu.offlinedocai.ai.retrieval.VectorStore
import com.siddu.offlinedocai.data.AppDatabase
import com.siddu.offlinedocai.data.entities.DocumentEntity
import com.siddu.offlinedocai.domain.repository.DocumentRepository
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * On-device checks of the parts JVM tests cannot exercise: the real ONNX embedder, Room/SQLite
 * (migration, foreign keys) and, when a GGUF model is installed, llama.cpp generation.
 * The sample document is synthetic and spans unrelated topics on purpose.
 */
@RunWith(AndroidJUnit4::class)
class RagPipelineAndroidTest {

    private lateinit var context: Context
    private lateinit var db: AppDatabase

    private val handbook = """
        Employee Handbook

        Leave Policy

        Staff members receive twenty days of paid annual leave in their first year. Unused days may be carried over until March of the following year. Sick leave is granted separately and requires a doctor's note after three consecutive days.

        Information Security

        Laptops must use full-disk encryption. Passwords are rotated every ninety days, and a hardware security key is required for remote access to internal systems. Lost devices must be reported to the help desk within one hour.

        Cafeteria

        The cafeteria serves breakfast from seven to nine and lunch from twelve to two. Vegetarian and vegan options are available every day, and the menu is published on Monday mornings.

        Travel Expenses

        Economy class is required for flights shorter than six hours. Hotel costs are reimbursed up to the city limit listed in the travel portal, and receipts must be submitted within thirty days of returning.
    """.trimIndent()

    @Before
    fun setUp() {
        context = InstrumentationRegistry.getInstrumentation().targetContext
        db = Room.inMemoryDatabaseBuilder(context, AppDatabase::class.java).build()
    }

    @After
    fun tearDown() {
        db.close()
    }

    @Test
    fun importIndexesWithBoundedChunksAndReindexDoesNotDuplicate() = runBlocking {
        val file = File(context.cacheDir, "handbook.txt").apply { writeText(handbook) }
        val repo = DocumentRepository(context, db)
        repo.importDocument(Uri.fromFile(file))

        val doc = waitForStatus(DocumentRepository.STATUS_COMPLETED)
        val engine = ONNXEmbeddingEngine.getInstance(context)
        val chunks = db.chunkDao().getAllChunksForDocument(doc.id)
        assertTrue(chunks.isNotEmpty())
        for (c in chunks) {
            assertTrue(engine.countTokens(c.text) <= TextChunker.DEFAULT_MAX_TOKENS)
            assertEquals(ONNXEmbeddingEngine.MODEL_IDENTIFIER, c.embeddingModelId)
            assertEquals(ONNXEmbeddingEngine.EMBEDDING_DIM, c.embeddingDim)
        }
        assertEquals(DocumentRepository.CURRENT_INDEX_SIGNATURE, doc.indexSignature)

        // Re-indexing replaces chunks instead of appending duplicates.
        repo.reprocessDocument(doc)
        delay(300)
        waitForStatus(DocumentRepository.STATUS_COMPLETED)
        assertEquals(chunks.size, db.chunkDao().countForDocument(doc.id))

        // Deleting the document removes its chunks.
        repo.deleteDocument(doc.id)
        assertEquals(0, db.chunkDao().countForDocument(doc.id))
    }

    @Test
    fun realEmbeddingsAnswerParaphrasedQuestionsAcrossTopics() = runBlocking {
        val docId = indexHandbook()
        val engine = ONNXEmbeddingEngine.getInstance(context)
        val store = VectorStore(db.chunkDao())

        val expectations = mapOf(
            "How much vacation do new hires get?" to "twenty days of paid annual leave",
            "What do I do if my computer goes missing?" to "Lost devices must be reported",
            "When can I eat lunch at work?" to "lunch from twelve to two",
            "Can I fly business class on a short trip?" to "Economy class is required"
        )
        for ((question, evidence) in expectations) {
            val cls = QueryIntentRouter.route(question)
            val results = store.retrieve(SearchQuery(question, engine.getEmbedding(question)), listOf(docId), cls)
            assertTrue("no results for '$question'", results.isNotEmpty())
            assertTrue("'$question' -> ${results.first().chunk.text}", results.take(2).any { it.chunk.text.contains(evidence) })
        }
    }

    @Test
    fun migrationFromV2RemovesOrphansAndDuplicatesAndEnforcesForeignKeys() {
        val name = "migration-test.db"
        context.deleteDatabase(name)
        createV2Database(name).use { legacy ->
            legacy.execSQL("INSERT INTO documents (id, uri, name, type, pageCount, addedAt, status) VALUES (1, 'content://a', 'a.pdf', 'pdf', 3, 0, 'COMPLETED')")
            val blob = ByteArray(384 * 4)
            for (i in 0 until 2) { // the old no-op delete left duplicated chunks behind
                legacy.execSQL("INSERT INTO chunks (documentId, pageNumber, text, embedding, chunkIndex, pageSpan, sectionTitle, paragraphIndex) VALUES (1, 1, 'alpha', ?, 0, '1', NULL, 0)", arrayOf<Any>(blob))
                legacy.execSQL("INSERT INTO chunks (documentId, pageNumber, text, embedding, chunkIndex, pageSpan, sectionTitle, paragraphIndex) VALUES (1, 2, 'beta', ?, 1, '2', NULL, 1)", arrayOf<Any>(blob))
            }
            // Orphan: its document was deleted.
            legacy.execSQL("INSERT INTO chunks (documentId, pageNumber, text, embedding, chunkIndex, pageSpan, sectionTitle, paragraphIndex) VALUES (99, 1, 'orphan', NULL, 0, '1', NULL, 0)")
        }

        val migrated = Room.databaseBuilder(context, AppDatabase::class.java, name)
            .addMigrations(AppDatabase.MIGRATION_1_2, AppDatabase.MIGRATION_2_3)
            .build()
        try {
            runBlocking {
                val chunks = migrated.chunkDao().getAllChunksForDocument(1)
                assertEquals(listOf("alpha", "beta"), chunks.map { it.text })
                assertTrue(chunks.all { it.embeddingModelId == ONNXEmbeddingEngine.MODEL_IDENTIFIER && it.embeddingDim == 384 })
                assertEquals(0, migrated.chunkDao().countForDocument(99))
                val doc = migrated.documentDao().getById(1)!!
                assertEquals(null, doc.indexSignature) // forces re-indexing with the new pipeline

                migrated.documentDao().delete(doc)
                assertEquals("ON DELETE CASCADE", 0, migrated.chunkDao().countForDocument(1))
            }
        } finally {
            migrated.close()
            context.deleteDatabase(name)
        }
    }

    @Test
    fun groundedAnswerFromInstalledModel() = runBlocking {
        val modelFile = File(context.filesDir, "models/llm").listFiles()?.firstOrNull { it.extension == "gguf" }
        assumeTrue("No GGUF model installed on this device; skipping LLM test", modelFile != null)

        val docId = indexHandbook()
        val engine = ONNXEmbeddingEngine.getInstance(context)
        val modelManager = com.siddu.offlinedocai.data.ai.ModelManager.getInstance(context)
        modelManager.refresh()
        modelManager.setActiveModel(modelFile!!.name)
        val llm = GGUFLLM.getInstance(context)
        val question = "How many days of paid leave do staff get in their first year?"
        val cls = QueryIntentRouter.route(question)
        val results = VectorStore(db.chunkDao()).retrieve(SearchQuery(question, engine.getEmbedding(question)), listOf(docId), cls)
        val budget = minOf(llm.availableEvidenceTokens(question, AnswerLength.SHORT), 1800)
        val assembled = ContextAssembler.assemble(results, cls, budget, llm::countTokens)
        val answer = llm.generateGrounded(assembled, question, AnswerLength.SHORT)
        llm.release()

        assertTrue(answer, answer.contains("twenty", ignoreCase = true) || answer.contains("20"))
        assertTrue("answer should cite a source: $answer", answer.contains("[Source"))
    }

    // ---------------------------------------------------------------------------------------------

    private suspend fun indexHandbook(): Long {
        val docId = db.documentDao().insert(DocumentEntity(uri = "test://handbook", name = "handbook.txt", type = "text/plain", status = "COMPLETED"))
        val engine = ONNXEmbeddingEngine.getInstance(context)
        val chunks = TextChunker.chunkDocument(
            listOf(com.siddu.offlinedocai.domain.rag.ExtractedPage(1, handbook)),
            tokenCounter = engine::countTokens
        )
        val entities = chunks.map { c ->
            val vector = engine.getEmbedding(listOfNotNull(c.sectionTitle, c.text).joinToString("\n"))
            com.siddu.offlinedocai.data.entities.ChunkEntity(
                documentId = docId, pageNumber = c.pageNumber, text = c.text,
                embedding = com.siddu.offlinedocai.data.FloatArrayConverter.fromFloatArray(vector),
                chunkIndex = c.index, pageSpan = c.pageSpan, sectionTitle = c.sectionTitle,
                embeddingModelId = engine.modelId, embeddingDim = vector.size
            )
        }
        db.chunkDao().replaceChunksForDocument(docId, entities)
        return docId
    }

    private suspend fun waitForStatus(status: String): DocumentEntity = withTimeout(120_000) {
        while (true) {
            val doc = db.documentDao().getAllDocumentsOnce().firstOrNull()
            if (doc?.status == status) return@withTimeout doc
            if (doc?.status == DocumentRepository.STATUS_FAILED) throw AssertionError("Import failed: ${doc.errorMessage}")
            delay(200)
        }
        @Suppress("UNREACHABLE_CODE") throw IllegalStateException()
    }

    /** Builds a database exactly as schema v2 (the previous release) created it. */
    private fun createV2Database(name: String): SupportSQLiteDatabase {
        val config = SupportSQLiteOpenHelper.Configuration.builder(context)
            .name(name)
            .callback(object : SupportSQLiteOpenHelper.Callback(2) {
                override fun onCreate(db: SupportSQLiteDatabase) {
                    db.execSQL("CREATE TABLE IF NOT EXISTS `documents` (`id` INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, `uri` TEXT NOT NULL, `name` TEXT NOT NULL, `type` TEXT NOT NULL, `pageCount` INTEGER NOT NULL, `addedAt` INTEGER NOT NULL, `status` TEXT NOT NULL)")
                    db.execSQL("CREATE TABLE IF NOT EXISTS `chunks` (`id` INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, `documentId` INTEGER NOT NULL, `pageNumber` INTEGER NOT NULL, `text` TEXT NOT NULL, `embedding` BLOB, `chunkIndex` INTEGER NOT NULL, `pageSpan` TEXT NOT NULL, `sectionTitle` TEXT, `paragraphIndex` INTEGER NOT NULL)")
                    db.execSQL("CREATE TABLE IF NOT EXISTS `conversations` (`id` INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, `title` TEXT NOT NULL, `createdAt` INTEGER NOT NULL, `documentId` INTEGER)")
                    db.execSQL("CREATE TABLE IF NOT EXISTS `messages` (`id` INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, `conversationId` INTEGER NOT NULL, `role` TEXT NOT NULL, `content` TEXT NOT NULL, `timestamp` INTEGER NOT NULL, `sources` TEXT)")
                }

                override fun onUpgrade(db: SupportSQLiteDatabase, oldVersion: Int, newVersion: Int) = Unit
            })
            .build()
        return FrameworkSQLiteOpenHelperFactory().create(config).writableDatabase
    }
}
