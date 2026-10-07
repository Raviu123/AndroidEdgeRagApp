package com.siddu.offlinedocai

import com.siddu.offlinedocai.ai.export.ChatExporter
import com.siddu.offlinedocai.data.entities.MessageEntity
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class ChatExporterTest {

    private val sampleMessages = listOf(
        MessageEntity(
            id = 1L,
            conversationId = 100L,
            role = "assistant",
            content = "Hello! How can I help you?",
            timestamp = 1000L
        ),
        MessageEntity(
            id = 2L,
            conversationId = 100L,
            role = "user",
            content = "What is the refund policy?",
            timestamp = 2000L
        ),
        MessageEntity(
            id = 3L,
            conversationId = 100L,
            role = "assistant",
            content = "Unused items can be returned within 30 days for a full refund.",
            timestamp = 3000L,
            sources = "Page 1: Unused items can be returned within 30 days...|||Page 2: Refunds go to the original payment method"
        )
    )

    @Test
    fun testExportWithChunks() {
        val jsonStr = ChatExporter.buildExportJson(
            conversationId = 100L,
            conversationTitle = "Test Chat",
            documentId = 1L,
            messages = sampleMessages,
            includeChunks = true
        )

        val obj = JSONObject(jsonStr)
        assertEquals(100L, obj.getLong("conversationId"))
        assertTrue("includeChunks should be true", obj.getBoolean("includeChunks"))

        val qaTurns = obj.getJSONArray("qaTurns")
        assertEquals(2, qaTurns.length())

        val turn2 = qaTurns.getJSONObject(1)
        assertEquals("What is the refund policy?", turn2.getString("question"))
        assertTrue("Should have retrieved chunks", turn2.has("retrievedChunks"))
        assertEquals(2, turn2.getJSONArray("retrievedChunks").length())
    }

    @Test
    fun testExportWithoutChunks_QnAOnly() {
        val jsonStr = ChatExporter.buildExportJson(
            conversationId = 100L,
            conversationTitle = "Test Chat",
            documentId = 1L,
            messages = sampleMessages,
            includeChunks = false
        )

        val obj = JSONObject(jsonStr)
        assertFalse("includeChunks should be false", obj.getBoolean("includeChunks"))

        val qaTurns = obj.getJSONArray("qaTurns")
        assertEquals(2, qaTurns.length())

        val turn2 = qaTurns.getJSONObject(1)
        assertEquals("What is the refund policy?", turn2.getString("question"))
        assertEquals("Unused items can be returned within 30 days for a full refund.", turn2.getString("response"))
        assertFalse("Should NOT have retrievedChunks in Q&A only mode", turn2.has("retrievedChunks"))

        val rawMessages = obj.getJSONArray("rawMessages")
        for (i in 0 until rawMessages.length()) {
            val msg = rawMessages.getJSONObject(i)
            assertFalse("Raw messages should NOT contain sources in Q&A only mode", msg.has("sources"))
        }
    }

    @Test
    fun testBuildPlainTextQnA() {
        val text = ChatExporter.buildPlainTextQnA(
            conversationTitle = "Store Policies",
            messages = sampleMessages
        )

        assertTrue(text.contains("# 💬 Store Policies"))
        assertTrue(text.contains("### Q1: What is the refund policy?"))
        assertTrue(text.contains("**Answer:**"))
        assertTrue(text.contains("Unused items can be returned within 30 days for a full refund."))
        assertFalse("Plain text should not contain raw chunks markup", text.contains("Page 1 [SCR"))
    }
}
