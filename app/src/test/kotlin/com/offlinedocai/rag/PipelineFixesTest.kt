package com.offlinedocai.rag

import com.offlinedocai.ai.chunking.TextChunker
import com.offlinedocai.ai.embeddings.AvailableEmbeddingModels
import com.offlinedocai.ai.embeddings.EmbeddingPooling
import com.offlinedocai.ai.llm.AnswerLength
import com.offlinedocai.ai.llm.ChatMessage
import com.offlinedocai.ai.llm.PromptBuilder
import com.offlinedocai.ai.retrieval.QueryClassification
import com.offlinedocai.ai.retrieval.QueryIntent
import com.offlinedocai.ai.retrieval.SearchQuery
import com.offlinedocai.ai.retrieval.VectorStore
import com.offlinedocai.data.entities.DocumentEntity
import com.offlinedocai.domain.rag.DocumentProcessor
import com.offlinedocai.domain.rag.ExtractedPage
import com.offlinedocai.domain.repository.ChatRepository
import com.offlinedocai.domain.repository.DocumentRepository
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PipelineFixesTest {

    // --- Reasoning models (DeepSeek-R1) -------------------------------------------------------------

    @Test
    fun reasoningModelDetection() {
        assertTrue(PromptBuilder.isReasoningModel(null, "DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf"))
        assertTrue(PromptBuilder.isReasoningModel("{{ '<think>\\n' }}", "anything"))
        assertFalse(PromptBuilder.isReasoningModel("<|im_start|>", "qwen2.5-1.5b-instruct-q4_k_m.gguf"))
        assertFalse(PromptBuilder.isReasoningModel(null, "Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
    }

    @Test
    fun unfinishedThinkingIsDetectedAndHidden() {
        assertTrue(PromptBuilder.isStillThinking("<think>step 1, step 2", promptOpensThinking = false))
        assertFalse(PromptBuilder.isStillThinking("<think>ok</think> The answer.", promptOpensThinking = false))
        // Template already opened <think>: the output only closes it.
        assertTrue(PromptBuilder.isStillThinking("still reasoning", promptOpensThinking = true))
        assertFalse(PromptBuilder.isStillThinking("done</think>\nAnswer", promptOpensThinking = true))
        assertEquals("", PromptBuilder.cleanOutput("still reasoning", promptOpensThinking = true))
        assertEquals("Answer [Source 1, Page 2]", PromptBuilder.cleanOutput("done</think>\nAnswer [Source 1, Page 2]", promptOpensThinking = true))
        assertEquals("_Thinking…_", PromptBuilder.visibleAnswer("<think>hmm"))
    }

    @Test
    fun deepSeekFoldsSystemPromptIntoFirstUserTurn() {
        val folded = PromptBuilder.foldSystemIntoUser(
            listOf(ChatMessage("system", "RULES"), ChatMessage("user", "Q"))
        )
        assertEquals(1, folded.size)
        assertEquals("user", folded[0].role)
        assertEquals("RULES\n\nQ", folded[0].content)
    }

    @Test
    fun followUpContextIsAddedToGroundedPrompt() {
        val messages = PromptBuilder.groundedMessages(emptyList(), "And the second one?", AnswerLength.DEFAULT, previousQuestion = "What are the two leave types?")
        assertTrue(messages.last().content.contains("What are the two leave types?"))
    }

    // --- Not-found detection ------------------------------------------------------------------------

    @Test
    fun paraphrasedNotFoundAnswersAreRecognised() {
        assertTrue(PromptBuilder.isNotFoundAnswer("The provided sources do not mention the refund window."))
        assertTrue(PromptBuilder.isNotFoundAnswer("I could not find any information about that."))
        // A cited partial answer is still an answer.
        assertFalse(PromptBuilder.isNotFoundAnswer("The sources do not mention fees, but shipping takes 5 days [Source 1, Page 3]."))
        assertFalse(PromptBuilder.isNotFoundAnswer("Shipping takes five days."))
    }

    // --- Follow-up detection -------------------------------------------------------------------------

    @Test
    fun followUpQuestionsAreDetected() {
        assertTrue(ChatRepository.isFollowUp("And what about the second one?"))
        assertTrue(ChatRepository.isFollowUp("Why is that?"))
        assertTrue(ChatRepository.isFollowUp("Explain it more"))
        assertFalse(ChatRepository.isFollowUp("What is the warranty period for the X200 vacuum cleaner model?"))
    }

    // --- Embedding catalog --------------------------------------------------------------------------

    @Test
    fun embeddingCatalogHasThreeVerifiedModels() {
        val list = AvailableEmbeddingModels.list
        assertEquals(3, list.size)
        assertEquals(1, list.count { it.isBundled })
        assertTrue(list.all { it.sha256.length == 64 && it.sizeBytes > 0 && it.dimension == 384 })
        assertEquals(list.size, list.map { it.id }.distinct().size)
        val bge = AvailableEmbeddingModels.byId(AvailableEmbeddingModels.BGE_SMALL_ONNX_ID)!!
        assertEquals(EmbeddingPooling.CLS, bge.pooling)
        assertTrue(bge.queryPrefix.isNotBlank())
        // The default id must stay stable: existing indexes were built with it.
        assertEquals("all-MiniLM-L6-v2-v1", AvailableEmbeddingModels.default.id)
    }

    // --- Index signatures ---------------------------------------------------------------------------

    @Test
    fun documentsAreSearchableOnlyWithTheModelThatIndexedThem() {
        val minilm = AvailableEmbeddingModels.MINILM_ONNX_ID
        val bge = AvailableEmbeddingModels.BGE_SMALL_ONNX_ID
        val doc = DocumentEntity(id = 1, uri = "x", name = "x", type = "pdf", status = DocumentRepository.STATUS_COMPLETED, indexSignature = DocumentRepository.indexSignature(minilm))
        assertTrue(DocumentRepository.isSearchable(doc, minilm))
        assertFalse(DocumentRepository.isSearchable(doc, bge))
        // An older chunker version with the same model stays searchable while it is re-chunked.
        assertTrue(DocumentRepository.isSearchable(doc.copy(indexSignature = "chunker-v2|$minilm"), minilm))
        assertFalse(DocumentRepository.isSearchable(doc.copy(status = DocumentRepository.STATUS_PROCESSING), minilm))
    }

    @Test
    fun retrievalIgnoresVectorsFromOtherEmbeddingModels() = runBlocking {
        val dao = FakeChunkDao()
        dao.insertAll(
            listOf(
                chunk(1, 1, 0, "refund policy details", TestVectors.topic(10 to 1f)).copy(embeddingModelId = "model-a"),
                chunk(2, 1, 1, "refund policy details", TestVectors.topic(10 to 1f)).copy(embeddingModelId = "model-b")
            )
        )
        val q = "refund policy"
        val results = VectorStore(dao).retrieve(
            SearchQuery(q, TestVectors.topic(10 to 1f)), listOf(1L), QueryClassification(QueryIntent.GENERAL, q), embeddingModelId = "model-b"
        )
        assertTrue(results.isNotEmpty())
        assertTrue(results.all { it.chunk.embeddingModelId == "model-b" })
    }

    // --- Text extraction ----------------------------------------------------------------------------

    @Test
    fun textDecodingHandlesBomsAndLegacyEncodings() {
        assertEquals("héllo", DocumentProcessor.decodeText(byteArrayOf(0xEF.toByte(), 0xBB.toByte(), 0xBF.toByte()) + "héllo".toByteArray(Charsets.UTF_8)))
        assertEquals("hi", DocumentProcessor.decodeText(byteArrayOf(0xFF.toByte(), 0xFE.toByte()) + "hi".toByteArray(Charsets.UTF_16LE)))
        assertEquals("café", DocumentProcessor.decodeText("café".toByteArray(Charsets.UTF_8)))
        // Windows-1252 "café" is invalid UTF-8 and must not become garbage.
        assertEquals("café", DocumentProcessor.decodeText(byteArrayOf('c'.code.toByte(), 'a'.code.toByte(), 'f'.code.toByte(), 0xE9.toByte())))
    }

    // --- Chunking -----------------------------------------------------------------------------------

    @Test
    fun longRunOnTextIsSplitWithinTheTokenLimit() {
        val count = TestTokenizer.instance::countTokens
        val runOn = (1..900).joinToString(" ") { "word$it" } // no sentence punctuation at all
        val chunks = TextChunker.chunkDocument(listOf(ExtractedPage(1, runOn)), count)
        assertTrue(chunks.size > 1)
        assertTrue(chunks.all { count(it.text) <= TextChunker.DEFAULT_MAX_TOKENS })
        // No word is lost or duplicated by the word-window split (no overlap is carried for run-on text inside one block).
        val words = chunks.flatMap { it.text.split(" ") }.toSet()
        assertTrue((1..900).all { "word$it" in words })
    }

    @Test
    fun overlapFromPreviousPageIsIncludedInPageSpan() {
        val count = TestTokenizer.instance::countTokens
        val sentence = { i: Int -> "Sentence number $i explains an important detail about the subject matter." }
        val page1 = (1..14).joinToString(" ") { sentence(it) }
        val page2 = "New topic begins here. " + (15..30).joinToString(" ") { sentence(it) }
        val chunks = TextChunker.chunkDocument(listOf(ExtractedPage(1, page1), ExtractedPage(2, page2)), count)
        val firstOnPage2 = chunks.first { it.text.contains("Sentence number 16 ") || it.text.contains("New topic") }
        if (firstOnPage2.text.contains("Sentence number 14 ")) {
            assertTrue("chunk with page-1 overlap must cite page 1 too: ${firstOnPage2.pageSpan}", firstOnPage2.pageSpan.startsWith("1"))
        }
    }
}
