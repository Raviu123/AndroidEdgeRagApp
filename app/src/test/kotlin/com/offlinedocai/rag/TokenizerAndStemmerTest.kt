package com.offlinedocai.rag

import com.offlinedocai.ai.retrieval.PorterStemmer
import com.offlinedocai.ai.retrieval.TextAnalyzer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class TokenizerAndStemmerTest {

    private val tokenizer = TestTokenizer.instance

    @Test
    fun wordPiece_matchesBertUncasedBehaviour() {
        val ids = tokenizer.tokenizeToIds("Unaffable café, naïve résumé!")
        // Accents are stripped and text lower-cased exactly like BertTokenizer(do_lower_case=True).
        assertEquals(tokenizer.tokenizeToIds("unaffable cafe, naive resume!"), ids)
        assertTrue(ids.none { it == tokenizer.unkTokenId })
    }

    @Test
    fun encode_truncatesButAlwaysKeepsClsAndSep() {
        val longText = List(600) { "token$it" }.joinToString(" ")
        val ids = tokenizer.encode(longText, 256)
        assertEquals(256, ids.size)
        assertEquals(tokenizer.clsTokenId, ids.first())
        assertEquals(tokenizer.sepTokenId, ids.last())
    }

    @Test
    fun encode_shortTextIsNotPadded() {
        val ids = tokenizer.encode("what is the refund policy", 256)
        assertEquals(tokenizer.countTokens("what is the refund policy") + 2, ids.size)
    }

    @Test
    fun unicodePunctuationIsSplitFromWords() {
        // The em dash and curly quotes are separate tokens, so the words around them stay intact.
        assertEquals(3, tokenizer.countTokens("cost—benefit"))
        assertEquals(tokenizer.tokenizeToIds("cost"), tokenizer.tokenizeToIds("“cost”").drop(1).dropLast(1))
    }

    @Test
    fun porterStemmer_referenceOutputs() {
        val expected = mapOf(
            "caresses" to "caress", "ponies" to "poni", "cats" to "cat", "agreed" to "agre",
            "plastered" to "plaster", "motoring" to "motor", "sized" to "size", "hopping" to "hop",
            "happy" to "happi", "relational" to "relat", "generalization" to "gener",
            "oscillators" to "oscil", "connection" to "connect", "connected" to "connect",
            "connecting" to "connect", "connections" to "connect", "rate" to "rate", "accurate" to "accur"
        )
        for ((word, stem) in expected) assertEquals("stem($word)", stem, PorterStemmer.stem(word))
    }

    @Test
    fun analyzer_neverMatchesBySubstring() {
        val rate = TextAnalyzer.terms("rate").single()
        assertFalse(rate in TextAnalyzer.terms("The accurate and separate generator"))
        val data = TextAnalyzer.terms("data").single()
        assertFalse(data in TextAnalyzer.terms("The database stores records"))
        assertTrue(TextAnalyzer.terms("Interest rates rose").contains(rate))
    }

    @Test
    fun analyzer_dropsQuestionScaffoldingButKeepsContentWords() {
        assertEquals(listOf("refund", "polici"), TextAnalyzer.terms("What is the refund policy?"))
        assertEquals(listOf("2024", "budget"), TextAnalyzer.terms("the 2024 budget"))
    }
}
