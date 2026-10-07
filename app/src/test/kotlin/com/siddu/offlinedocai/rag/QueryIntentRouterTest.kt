package com.siddu.offlinedocai.rag

import com.siddu.offlinedocai.ai.retrieval.QueryIntent
import com.siddu.offlinedocai.ai.retrieval.QueryIntentRouter
import org.junit.Assert.assertEquals
import org.junit.Test

class QueryIntentRouterTest {

    private fun intent(q: String) = QueryIntentRouter.route(q).intent

    @Test
    fun pageQuestions() {
        val single = QueryIntentRouter.route("What is on page 12?")
        assertEquals(QueryIntent.PAGE_LOOKUP, single.intent)
        assertEquals(12, single.pageStart)

        val range = QueryIntentRouter.route("Summarize pages 3 to 7")
        assertEquals(QueryIntent.PAGE_RANGE, range.intent)
        assertEquals(3 to 7, range.pageStart to range.pageEnd)

        // Mentioning a page inside a real question is not a page lookup.
        assertEquals(QueryIntent.GENERAL, intent("Why does the author say on page 4 that costs rose?"))
    }

    @Test
    fun documentSummaries() {
        listOf("Summarize this document", "summary", "What is this paper about?", "Give me an overview of the report", "tl;dr")
            .forEach { assertEquals(it, QueryIntent.SUMMARY, intent(it)) }
        // A topical summary request is an ordinary search, not a whole-document summary.
        assertEquals(QueryIntent.GENERAL, intent("Summarize the section on interest rates"))
    }

    @Test
    fun comparisons() {
        val c = QueryIntentRouter.route("What is the difference between a Roth IRA and a traditional IRA?")
        assertEquals(QueryIntent.COMPARISON, c.intent)
        assertEquals(listOf("Roth IRA", "traditional IRA"), c.subQueries)
        assertEquals(QueryIntent.COMPARISON, intent("Compare TCP with UDP"))
        assertEquals(QueryIntent.COMPARISON, intent("Python vs Java"))
    }

    @Test
    fun enumerations() {
        listOf("List all the side effects", "What are the main risks?", "Name every author mentioned", "What are the steps to reset the router?", "How many employees does the company have?")
            .forEach { assertEquals(it, QueryIntent.ENUMERATION, intent(it)) }
    }

    @Test
    fun everythingElseIsAGeneralSearch() {
        listOf(
            "What is photosynthesis?", "Who is the CEO?", "When was the contract signed?",
            "How does the refund process work?", "What is the phone number for support?", "explain thermodynamics"
        ).forEach { assertEquals(it, QueryIntent.GENERAL, intent(it)) }
    }
}
