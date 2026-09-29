package com.siddu.offlinedocai

import android.content.Context
import androidx.room.Room
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.siddu.offlinedocai.ai.embeddings.ONNXEmbeddingEngine
import com.siddu.offlinedocai.ai.llm.GGUFLLM
import com.siddu.offlinedocai.ai.retrieval.VectorStore
import com.siddu.offlinedocai.data.AppDatabase
import com.siddu.offlinedocai.data.FloatArrayConverter
import com.siddu.offlinedocai.data.entities.ChunkEntity
import com.siddu.offlinedocai.data.entities.DocumentEntity
import com.siddu.offlinedocai.domain.repository.ChatRepository
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Diagnostic trace test for the single query "what is grafting"
 * through the exact production ChatRepository and GGUF pipeline.
 */
@RunWith(AndroidJUnit4::class)
class SingleProductionQueryTraceTest {

    private lateinit var context: Context
    private lateinit var database: AppDatabase
    private lateinit var vectorStore: VectorStore
    private lateinit var chatRepository: ChatRepository
    private var docId: Long = 0L

    @Before
    fun setup() {
        context = InstrumentationRegistry.getInstrumentation().targetContext
        PDFBoxResourceLoader.init(context)

        database = Room.inMemoryDatabaseBuilder(context, AppDatabase::class.java)
            .allowMainThreadQueries()
            .build()

        vectorStore = VectorStore(database.chunkDao())
        chatRepository = ChatRepository(context, database, vectorStore)

        runBlocking {
            docId = ingestGraftingDocument()
        }
    }

    @After
    fun tearDown() {
        (chatRepository.llm as? GGUFLLM)?.llamaRunner?.freeModel()
        database.close()
    }

    private suspend fun ingestGraftingDocument(): Long {
        val docEntity = DocumentEntity(
            id = 0,
            name = "Grafting_Guide.pdf",
            uri = "content://dummy/grafting.pdf",
            type = "application/pdf",
            pageCount = 2,
            status = "READY"
        )
        val generatedDocId = database.documentDao().insert(docEntity)

        val embeddingEngine = ONNXEmbeddingEngine(context)

        val page1Text = "Grafting is a horticultural technique whereby tissues of plants are joined so as to continue their growth together. The upper part of the combined plant is called the scion while the lower part is called the rootstock."
        val page2Text = "For successful grafting to occur, the vascular cambium tissues of the rootstock and scion must be placed in contact with each other."

        val chunks = listOf(
            ChunkEntity(
                id = 0,
                documentId = generatedDocId,
                pageNumber = 1,
                chunkIndex = 0,
                text = page1Text,
                sectionTitle = "Introduction to Grafting",
                pageSpan = "1",
                embedding = FloatArrayConverter.fromFloatArray(embeddingEngine.getEmbedding(page1Text))
            ),
            ChunkEntity(
                id = 0,
                documentId = generatedDocId,
                pageNumber = 2,
                chunkIndex = 1,
                text = page2Text,
                sectionTitle = "Vascular Cambium",
                pageSpan = "2",
                embedding = FloatArrayConverter.fromFloatArray(embeddingEngine.getEmbedding(page2Text))
            )
        )
        database.chunkDao().insertAll(chunks)
        embeddingEngine.close()
        return generatedDocId
    }

    @Test
    fun traceProductionQuery_what_is_grafting() = runBlocking {
        println("=== STARTING PRODUCTION TRACE FOR: 'what is grafting' ===")
        val query = "what is grafting"
        val convId = chatRepository.startConversation("Trace Chat", docId)

        val stateTransitions = mutableListOf<Pair<Long, ChatRepository.ChatState>>()
        val startTime = System.currentTimeMillis()

        val monitorJob = launch(Dispatchers.Default) {
            chatRepository.chatState.collectLatest { state ->
                val now = System.currentTimeMillis() - startTime
                println("CHAT_STATE_TRANSITION at +${now}ms: $state")
                stateTransitions.add(Pair(now, state))
            }
        }

        println("SUBMITTING QUERY via chatRepository.sendMessage()...")
        chatRepository.sendMessage(convId, query, listOf(docId))

        // Wait up to 180 seconds for resolution or observe hang
        var terminalReached = false
        for (i in 0 until 180) { // 180 seconds max
            delay(1000)
            val currentState = chatRepository.chatState.value
            if (currentState is ChatRepository.ChatState.Idle && stateTransitions.size > 2) {
                terminalReached = true
                println("TERMINAL STATE REACHED: Idle after ${System.currentTimeMillis() - startTime}ms")
                break
            }
            if (currentState is ChatRepository.ChatState.Error) {
                terminalReached = true
                println("TERMINAL STATE REACHED: Error(${currentState.message}) after ${System.currentTimeMillis() - startTime}ms")
                break
            }
        }

        monitorJob.cancel()
        val totalMs = System.currentTimeMillis() - startTime
        println("=== TRACE COMPLETE in ${totalMs}ms. TerminalReached=$terminalReached ===")
        println("State transitions count: ${stateTransitions.size}")
        stateTransitions.forEach { (t, s) ->
            println("  +${t}ms: $s")
        }

        val messages = database.messageDao().getMessagesForConversation(convId).first()
        println("Final Messages in Conversation (${messages.size}):")
        messages.forEach { m ->
            println("  [${m.role}] ${m.content} (sources=${m.sources != null})")
        }

        assertTrue("Terminal state (Idle or Error) must be reached", terminalReached)
        val assistantMessages = messages.filter { it.role == "assistant" && it.content.isNotBlank() && !it.content.startsWith("👋") }
        assertTrue("Assistant answer must be produced and persisted", assistantMessages.isNotEmpty())
        val answer = assistantMessages.last().content
        println("ASSERTED ANSWER: $answer")
        assertTrue("Answer must mention grafting or technique", answer.contains("grafting", ignoreCase = true) || answer.contains("technique", ignoreCase = true))
    }
}
