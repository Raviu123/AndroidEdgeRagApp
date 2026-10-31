package com.offlinedocai.rag

import com.offlinedocai.ai.chunking.TextChunker
import com.offlinedocai.ai.llm.AnswerLength
import com.offlinedocai.ai.llm.ChatTurn
import com.offlinedocai.ai.llm.GenerationRequest
import com.offlinedocai.ai.llm.ModelTier
import com.offlinedocai.ai.llm.PromptBuilder
import com.offlinedocai.ai.llm.RepetitionGuard
import com.offlinedocai.ai.retrieval.QueryIntent
import com.offlinedocai.domain.rag.ExtractedPage
import com.offlinedocai.domain.repository.ChatRepository
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Answer-quality safeguards: model tiers, loop handling, running headers, follow-up scoping. */
class AnswerQualityTest {

    // --- Model tiers --------------------------------------------------------------------------------

    @Test
    fun modelSizesAreReadFromMetadataAndNames() {
        assertEquals(ModelTier.TINY, ModelTier.detect("494032768", null, ""))
        assertEquals(ModelTier.STANDARD, ModelTier.detect("1543714304", null, ""))
        assertEquals(ModelTier.TINY, ModelTier.detect(null, "0.5B", ""))
        assertEquals(ModelTier.SMALL, ModelTier.detect(null, "1B", ""))
        assertEquals(ModelTier.TINY, ModelTier.detect(null, null, "qwen2.5-0.5b-instruct-q4_k_m.gguf"))
        assertEquals(ModelTier.SMALL, ModelTier.detect(null, null, "Llama 3.2 1B Instruct Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
        assertEquals(ModelTier.STANDARD, ModelTier.detect(null, null, "Qwen2.5 1.5B Instruct"))
        assertEquals(ModelTier.TINY, ModelTier.detect(null, null, "SmolLM2-360M-Instruct"))
        // Unknown size: never shrink the evidence of a model that may be large.
        assertEquals(ModelTier.STANDARD, ModelTier.detect(null, null, "my-model.gguf"))
    }

    @Test
    fun smallerModelsSeeLessEvidence() {
        val tiny = ChatRepository.evidenceCap(QueryIntent.GENERAL, ModelTier.TINY)
        val small = ChatRepository.evidenceCap(QueryIntent.GENERAL, ModelTier.SMALL)
        val standard = ChatRepository.evidenceCap(QueryIntent.GENERAL, ModelTier.STANDARD)
        assertTrue(tiny < small && small < standard)
        assertEquals(2000, standard)
        assertTrue(ChatRepository.evidenceCap(QueryIntent.ENUMERATION, ModelTier.TINY) > tiny)
    }

    // --- Prompts ------------------------------------------------------------------------------------

    @Test
    fun tinyPromptAsksToIgnoreUnrelatedSourcesAndReportGaps() {
        val system = PromptBuilder.groundedSystemPrompt(AnswerLength.DEFAULT, false, ModelTier.TINY)
        assertTrue(system.contains("ignore the rest"))
        assertTrue(system.contains("does not cover"))
        assertFalse("tiny models must not be pushed to pad answers", system.contains("single-line"))
    }

    @Test
    fun questionIsStatedBeforeAndAfterTheSources() {
        val user = PromptBuilder.groundedMessages(emptyList(), "How long is the trial?", AnswerLength.DEFAULT).last().content
        assertTrue(user.startsWith("Question to answer: How long is the trial?"))
        assertTrue(user.trimEnd().endsWith("Question: How long is the trial?"))
    }

    // --- Follow-ups ---------------------------------------------------------------------------------

    @Test
    fun longWhyQuestionsAreNewQuestions() {
        assertFalse(ChatRepository.isFollowUp("Why should applications validate data received from external sources?"))
        assertTrue(ChatRepository.isFollowUp("Why is that?"))
        assertTrue(ChatRepository.isFollowUp("Why?"))
    }

    @Test
    fun previousTurnCanBeDroppedFromARequest() {
        val request = GenerationRequest(previousTurn = ChatTurn("q", "a"))
        val stripped = request.withPreviousTurn(null)
        assertNull(stripped.previousTurn)
        assertTrue(stripped.token === request.token)
    }

    // --- Repetition ---------------------------------------------------------------------------------

    private val loopingList = buildString {
        append("Poor storage can cause several problems:\n\n")
        append("1. Data leakage: If files are written to shared folders, other apps may read them.\n\n")
        for (i in 2..9) {
            val label = if (i == 3) "Data corruption" else "Data tampering"
            append("$i. $label: If files are written to private folders that other apps can reach, they may be changed or deleted.\n\n")
        }
        append("10. Data tampering: If")
    }

    @Test
    fun loopingAnswersAreDetected() {
        assertTrue(RepetitionGuard.isLooping(loopingList))
        assertTrue(RepetitionGuard.isLooping("The answer is " + "the app and ".repeat(12)))
        assertFalse(RepetitionGuard.isLooping("The trial lasts 30 days. After the trial, a plan must be chosen. Plans can be changed at any time."))
    }

    @Test
    fun repeatedItemsAreRemovedAndListsRenumbered() {
        val cleaned = RepetitionGuard.clean(RepetitionGuard.dropUnfinishedTail(loopingList))
        val items = cleaned.lines().filter { Regex("""^\d+\.""").containsMatchIn(it) }
        assertEquals(2, items.size)
        assertTrue(items[0].startsWith("1. Data leakage"))
        assertTrue(items[1].startsWith("2. Data tampering"))
        assertFalse(cleaned.contains("If\n") || cleaned.endsWith("If"))
    }

    @Test
    fun normalAnswersAreLeftUnchanged() {
        val answer = "The trial lasts 30 days [Source 1, Page 2].\n\n1. Choose a plan.\n2. Enter payment details.\n3. Confirm the order."
        assertEquals(answer, RepetitionGuard.clean(answer))
    }

    // --- Running headers and footers ----------------------------------------------------------------

    @Test
    fun runningHeadersFootersAndPageNumbersAreRemoved() {
        val pages = (1..6).map { p ->
            ExtractedPage(
                p,
                "Acme Field Manual v3.2\n\nTopic $p is introduced here.\nBody text of page $p describes topic number $p in detail.\n" +
                    "A closing sentence that describes what topic number $p leads to next.\nPage $p of 6"
            )
        }
        val cleaned = TextChunker.stripRunningHeadersAndFooters(pages)
        for ((i, page) in cleaned.withIndex()) {
            assertFalse(page.text.contains("Acme Field Manual"))
            assertFalse(page.text.contains("of 6"))
            assertTrue(page.text.contains("Body text of page ${i + 1}"))
            // A long edge line that differs only by a number is content, not a footer.
            assertTrue(page.text.contains("what topic number ${i + 1} leads to next"))
        }
        val chunks = TextChunker.chunkDocument(pages)
        assertTrue(chunks.none { it.text.contains("Acme Field Manual") })
    }

    @Test
    fun linesThatRepeatOnFewPagesAreKept() {
        val pages = (1..10).map { p ->
            val top = if (p <= 2) "Shared introduction line" else "Unique opening line for page $p with words"
            ExtractedPage(p, "$top\nMiddle content for page $p.\nClosing remark that is specific to page $p only")
        }
        val cleaned = TextChunker.stripRunningHeadersAndFooters(pages)
        assertTrue(cleaned[0].text.contains("Shared introduction line"))
        assertTrue(cleaned[5].text.contains("Closing remark"))
    }

    @Test
    fun shortDocumentsOnlyLosePageNumbers() {
        val pages = listOf(ExtractedPage(1, "Title line\nSome content here.\n1"), ExtractedPage(2, "Title line\nMore content.\n2"))
        val cleaned = TextChunker.stripRunningHeadersAndFooters(pages)
        assertTrue(cleaned.all { it.text.startsWith("Title line") })
        assertTrue(cleaned.none { it.text.endsWith("1") || it.text.endsWith("2") })
    }
}
