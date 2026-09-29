package com.siddu.offlinedocai

import com.siddu.offlinedocai.ai.retrieval.FactType
import com.siddu.offlinedocai.ai.retrieval.QueryIntent
import com.siddu.offlinedocai.ai.retrieval.QueryIntentRouter
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class QueryIntentRouterTest {

    // 1. PAGE_LOOKUP
    @Test
    fun testPageLookupQueries() {
        val q1 = QueryIntentRouter.route("What is on page 10?")
        assertEquals(QueryIntent.PAGE_LOOKUP, q1.intent)
        assertEquals(10, q1.pageStart)
        assertEquals(10, q1.pageEnd)

        val q2 = QueryIntentRouter.route("tell me about page 4")
        assertEquals(QueryIntent.PAGE_LOOKUP, q2.intent)
        assertEquals(4, q2.pageStart)

        val q3 = QueryIntentRouter.route("what does the 3rd page say?")
        assertEquals(QueryIntent.PAGE_LOOKUP, q3.intent)
        assertEquals(3, q3.pageStart)

        val q4 = QueryIntentRouter.route("page no. 7")
        assertEquals(QueryIntent.PAGE_LOOKUP, q4.intent)
        assertEquals(7, q4.pageStart)

        val q5 = QueryIntentRouter.route("details on p. 12")
        assertEquals(QueryIntent.PAGE_LOOKUP, q5.intent)
        assertEquals(12, q5.pageStart)
    }

    // 2. PAGE_RANGE
    @Test
    fun testPageRangeQueries() {
        val q1 = QueryIntentRouter.route("Summarize pages 10-15.")
        assertEquals(QueryIntent.PAGE_RANGE, q1.intent)
        assertEquals(10, q1.pageStart)
        assertEquals(15, q1.pageEnd)

        val q2 = QueryIntentRouter.route("What happens from page 5 through page 9?")
        assertEquals(QueryIntent.PAGE_RANGE, q2.intent)
        assertEquals(5, q2.pageStart)
        assertEquals(9, q2.pageEnd)

        val q3 = QueryIntentRouter.route("between page 2 and 6")
        assertEquals(QueryIntent.PAGE_RANGE, q3.intent)
        assertEquals(2, q3.pageStart)
        assertEquals(6, q3.pageEnd)

        val q4 = QueryIntentRouter.route("pages 1 to 3")
        assertEquals(QueryIntent.PAGE_RANGE, q4.intent)
        assertEquals(1, q4.pageStart)
        assertEquals(3, q4.pageEnd)

        // Strict priority check: PAGE_RANGE must take precedence over PAGE_LOOKUP
        val qPriority = QueryIntentRouter.route("Summarize pages 10-15.")
        assertTrue("Must be PAGE_RANGE not PAGE_LOOKUP", qPriority.intent == QueryIntent.PAGE_RANGE)
    }

    // 3. EXACT_FACT
    @Test
    fun testExactFactQueries() {
        val qPhone = QueryIntentRouter.route("What is the phone number?")
        assertEquals(QueryIntent.EXACT_FACT, qPhone.intent)
        assertEquals(FactType.PHONE, qPhone.factType)

        val qEmail = QueryIntentRouter.route("What is the email address?")
        assertEquals(QueryIntent.EXACT_FACT, qEmail.intent)
        assertEquals(FactType.EMAIL, qEmail.factType)

        val qUrl = QueryIntentRouter.route("What is the website URL?")
        assertEquals(QueryIntent.EXACT_FACT, qUrl.intent)
        assertEquals(FactType.URL, qUrl.factType)

        val qDate = QueryIntentRouter.route("What date was this document signed?")
        assertEquals(QueryIntent.EXACT_FACT, qDate.intent)
        assertEquals(FactType.DATE, qDate.factType)

        val qId = QueryIntentRouter.route("What is the employee ID?")
        assertEquals(QueryIntent.EXACT_FACT, qId.intent)
        assertEquals(FactType.ID, qId.factType)

        val qAmount = QueryIntentRouter.route("What is the salary budget?")
        assertEquals(QueryIntent.EXACT_FACT, qAmount.intent)
        assertEquals(FactType.AMOUNT, qAmount.factType)

        val qPercent = QueryIntentRouter.route("What percentage of users passed?")
        assertEquals(QueryIntent.EXACT_FACT, qPercent.intent)
        assertEquals(FactType.PERCENTAGE, qPercent.factType)

        val qNumber = QueryIntentRouter.route("How many participants attended?")
        assertEquals(QueryIntent.EXACT_FACT, qNumber.intent)
        assertEquals(FactType.NUMBER, qNumber.factType)

        val qValue = QueryIntentRouter.route("What is the value of parameter K?")
        assertEquals(QueryIntent.EXACT_FACT, qValue.intent)
        assertEquals(FactType.EXACT_VALUE, qValue.factType)
    }

    // 4. ENTITY_LOOKUP
    @Test
    fun testEntityLookupQueries() {
        val q1 = QueryIntentRouter.route("What is Project Alpha?")
        assertEquals(QueryIntent.ENTITY_LOOKUP, q1.intent)
        assertEquals("Project Alpha", q1.targetTerm)

        val q2 = QueryIntentRouter.route("Tell me about John Smith.")
        assertEquals(QueryIntent.ENTITY_LOOKUP, q2.intent)
        assertEquals("John Smith", q2.targetTerm)

        val q3 = QueryIntentRouter.route("Who is Alan Turing?")
        assertEquals(QueryIntent.ENTITY_LOOKUP, q3.intent)
        assertEquals("Alan Turing", q3.targetTerm)

        val q4 = QueryIntentRouter.route("Explain the Apollo mission.")
        assertEquals(QueryIntent.ENTITY_LOOKUP, q4.intent)
        assertEquals("Apollo", q4.targetTerm)
    }

    // 5. SECTION_QUERY
    @Test
    fun testSectionQueries() {
        val q1 = QueryIntentRouter.route("What are the objectives?")
        assertEquals(QueryIntent.SECTION_QUERY, q1.intent)
        assertEquals("objectives", q1.targetTerm)

        val q2 = QueryIntentRouter.route("What are the requirements?")
        assertEquals(QueryIntent.SECTION_QUERY, q2.intent)
        assertEquals("requirements", q2.targetTerm)

        val q3 = QueryIntentRouter.route("Explain the methodology section.")
        assertEquals(QueryIntent.SECTION_QUERY, q3.intent)
        assertTrue(q3.targetTerm?.contains("methodology") == true)

        val q4 = QueryIntentRouter.route("What does the introduction section say?")
        assertEquals(QueryIntent.SECTION_QUERY, q4.intent)
        assertEquals("introduction", q4.targetTerm)

        val q5 = QueryIntentRouter.route("Explain Chapter 3.")
        assertEquals(QueryIntent.SECTION_QUERY, q5.intent)
        assertEquals("Chapter 3", q5.targetTerm)
    }

    // 6. EXHAUSTIVE_LIST
    @Test
    fun testExhaustiveListQueries() {
        val q1 = QueryIntentRouter.route("List all projects.")
        assertEquals(QueryIntent.EXHAUSTIVE_LIST, q1.intent)
        assertEquals("projects", q1.targetTerm)

        val q2 = QueryIntentRouter.route("List all technologies.")
        assertEquals(QueryIntent.EXHAUSTIVE_LIST, q2.intent)
        assertEquals("technologies", q2.targetTerm)

        val q3 = QueryIntentRouter.route("Give me the complete list of requirements.")
        assertEquals(QueryIntent.EXHAUSTIVE_LIST, q3.intent)
        assertEquals("requirements", q3.targetTerm)

        val q4 = QueryIntentRouter.route("What are all the tools mentioned?")
        assertEquals(QueryIntent.EXHAUSTIVE_LIST, q4.intent)
        assertEquals("tools", q4.targetTerm)

        val q5 = QueryIntentRouter.route("What are the projects?")
        assertEquals(QueryIntent.EXHAUSTIVE_LIST, q5.intent)
        assertEquals("projects", q5.targetTerm)
    }

    // 7. FIND_ALL
    @Test
    fun testFindAllQueries() {
        val q1 = QueryIntentRouter.route("Find every mention of authentication.")
        assertEquals(QueryIntent.FIND_ALL, q1.intent)
        assertEquals("authentication", q1.targetTerm)

        val q2 = QueryIntentRouter.route("Where is authentication mentioned?")
        assertEquals(QueryIntent.FIND_ALL, q2.intent)
        assertEquals("authentication", q2.targetTerm)

        val q3 = QueryIntentRouter.route("Find all occurrences of memory leak.")
        assertEquals(QueryIntent.FIND_ALL, q3.intent)
        assertEquals("memory leak", q3.targetTerm)

        val q4 = QueryIntentRouter.route("Show every place where privacy is discussed.")
        assertEquals(QueryIntent.FIND_ALL, q4.intent)
        assertEquals("privacy", q4.targetTerm)
    }

    // 8. COMPARISON
    @Test
    fun testComparisonQueries() {
        val q1 = QueryIntentRouter.route("Compare X and Y.")
        assertEquals(QueryIntent.COMPARISON, q1.intent)
        assertEquals("X", q1.entityA)
        assertEquals("Y", q1.entityB)

        val q2 = QueryIntentRouter.route("What is the difference between Kotlin and Java?")
        assertEquals(QueryIntent.COMPARISON, q2.intent)
        assertEquals("Kotlin", q2.entityA)
        assertEquals("Java", q2.entityB)

        val q3 = QueryIntentRouter.route("How does REST differ from GraphQL?")
        assertEquals(QueryIntent.COMPARISON, q3.intent)
        assertEquals("REST", q3.entityA)
        assertEquals("GraphQL", q3.entityB)

        val q4 = QueryIntentRouter.route("Qwen vs LLaMA")
        assertEquals(QueryIntent.COMPARISON, q4.intent)
        assertEquals("Qwen", q4.entityA)
        assertEquals("LLaMA", q4.entityB)
    }

    // 9. GLOBAL_SUMMARY
    @Test
    fun testGlobalSummaryQueries() {
        val q1 = QueryIntentRouter.route("What is this document about?")
        assertEquals(QueryIntent.GLOBAL_SUMMARY, q1.intent)

        val q2 = QueryIntentRouter.route("Summarize the document.")
        assertEquals(QueryIntent.GLOBAL_SUMMARY, q2.intent)

        val q3 = QueryIntentRouter.route("Give me an overview of this file.")
        assertEquals(QueryIntent.GLOBAL_SUMMARY, q3.intent)

        val q4 = QueryIntentRouter.route("summary")
        assertEquals(QueryIntent.GLOBAL_SUMMARY, q4.intent)
    }

    // 10. CONCEPTUAL_QUERY & Ambiguous Fallbacks
    @Test
    fun testConceptualAndAmbiguousQueries() {
        val q1 = QueryIntentRouter.route("Explain authentication.")
        assertEquals(QueryIntent.CONCEPTUAL_QUERY, q1.intent)

        val q2 = QueryIntentRouter.route("How does authentication work?")
        assertEquals(QueryIntent.CONCEPTUAL_QUERY, q2.intent)

        val q3 = QueryIntentRouter.route("Describe the data replication process.")
        assertEquals(QueryIntent.CONCEPTUAL_QUERY, q3.intent)

        // Ambiguous query: safe fallback
        val qAmbiguous1 = QueryIntentRouter.route("Is all of this correct?")
        assertEquals(QueryIntent.CONCEPTUAL_QUERY, qAmbiguous1.intent)
        assertTrue("Ambiguous queries must flag isAmbiguous=true", qAmbiguous1.isAmbiguous)

        val qAmbiguous2 = QueryIntentRouter.route("what about that?")
        assertEquals(QueryIntent.CONCEPTUAL_QUERY, qAmbiguous2.intent)
        assertTrue("Ambiguous queries must flag isAmbiguous=true", qAmbiguous2.isAmbiguous)
    }
}
