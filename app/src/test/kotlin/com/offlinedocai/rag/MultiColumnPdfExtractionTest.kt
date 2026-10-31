package com.offlinedocai.rag

import com.offlinedocai.domain.rag.DocumentProcessor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Verifies spatial multi-column layout reconstruction for PDF extraction,
 * guaranteeing that vertically split (two-column) documents are extracted in
 * proper reading order (Column 1 top-to-bottom, then Column 2 top-to-bottom)
 * without horizontal line interleaving.
 */
class MultiColumnPdfExtractionTest {

    @Test
    fun testSingleColumnPdfReconstruction() {
        val tokens = listOf(
            DocumentProcessor.PositionedToken("Introduction", 50f, 150f, 50f, 60f, 12f),
            DocumentProcessor.PositionedToken("to", 155f, 170f, 50f, 60f, 12f),
            DocumentProcessor.PositionedToken("Edge", 175f, 210f, 50f, 60f, 12f),
            DocumentProcessor.PositionedToken("AI", 215f, 235f, 50f, 60f, 12f),
            DocumentProcessor.PositionedToken("On-device", 50f, 120f, 75f, 85f, 10f),
            DocumentProcessor.PositionedToken("processing", 125f, 200f, 75f, 85f, 10f),
            DocumentProcessor.PositionedToken("preserves", 205f, 270f, 75f, 85f, 10f),
            DocumentProcessor.PositionedToken("privacy.", 275f, 330f, 75f, 85f, 10f)
        )

        val result = DocumentProcessor.reconstructPageText(tokens, "fallback")
        assertTrue(result.contains("Introduction to Edge AI"))
        assertTrue(result.contains("On-device processing preserves privacy."))
    }

    @Test
    fun testTwoColumnPdfReconstructionNoInterleaving() {
        // Simulates a 2-column page (Page width ~ 550pt).
        // Left Column: X in [50, 250], Y from 100 to 200
        // Right Column: X in [300, 500], Y from 100 to 200
        // In visual-order PDF streams, Left Line 1 and Right Line 1 appear at same Y (~100f).
        
        val tokens = mutableListOf<DocumentProcessor.PositionedToken>()
        
        // Left Column Line 1 & Right Column Line 1 (same Y = 100)
        tokens.add(DocumentProcessor.PositionedToken("Left column sentence one.", 50f, 240f, 90f, 100f, 10f))
        tokens.add(DocumentProcessor.PositionedToken("Right column sentence one.", 310f, 500f, 90f, 100f, 10f))

        // Left Column Line 2 & Right Column Line 2 (same Y = 120)
        tokens.add(DocumentProcessor.PositionedToken("Left column sentence two.", 50f, 240f, 110f, 120f, 10f))
        tokens.add(DocumentProcessor.PositionedToken("Right column sentence two.", 310f, 500f, 110f, 120f, 10f))

        // Left Column Line 3 & Right Column Line 3 (same Y = 140)
        tokens.add(DocumentProcessor.PositionedToken("Left column sentence three.", 50f, 240f, 130f, 140f, 10f))
        tokens.add(DocumentProcessor.PositionedToken("Right column sentence three.", 310f, 500f, 130f, 140f, 10f))

        // Left Column Line 4 & Right Column Line 4 (same Y = 160)
        tokens.add(DocumentProcessor.PositionedToken("Left column sentence four.", 50f, 240f, 150f, 160f, 10f))
        tokens.add(DocumentProcessor.PositionedToken("Right column sentence four.", 310f, 500f, 150f, 160f, 10f))

        val result = DocumentProcessor.reconstructPageText(tokens, "fallback")

        // In correct reading order, all Left column sentences MUST appear before Right column sentences
        val leftIdx1 = result.indexOf("Left column sentence one.")
        val leftIdx4 = result.indexOf("Left column sentence four.")
        val rightIdx1 = result.indexOf("Right column sentence one.")
        val rightIdx4 = result.indexOf("Right column sentence four.")

        assertTrue("Left col 1 must exist", leftIdx1 >= 0)
        assertTrue("Left col 4 must exist", leftIdx4 >= 0)
        assertTrue("Right col 1 must exist", rightIdx1 >= 0)
        assertTrue("Right col 4 must exist", rightIdx4 >= 0)

        // Verify un-interleaved sequence: Left 1 < Left 4 < Right 1 < Right 4
        assertTrue("Left 1 must come before Left 4", leftIdx1 < leftIdx4)
        assertTrue("Left 4 must come before Right 1 (Columns un-interleaved!)", leftIdx4 < rightIdx1)
        assertTrue("Right 1 must come before Right 4", rightIdx1 < rightIdx4)
    }

    @Test
    fun testTwoColumnWithTitleHeaderAndFooter() {
        val tokens = mutableListOf<DocumentProcessor.PositionedToken>()

        // Full-width title spanning X in [50, 500], Y = 40
        tokens.add(DocumentProcessor.PositionedToken("OfflineDocAI: Full Paper Title Spanning Both Columns", 50f, 500f, 25f, 40f, 14f))

        // Two columns in the middle (Y in 80..180)
        tokens.add(DocumentProcessor.PositionedToken("Col 1 para start.", 50f, 230f, 70f, 80f, 10f))
        tokens.add(DocumentProcessor.PositionedToken("Col 2 para start.", 310f, 490f, 70f, 80f, 10f))
        tokens.add(DocumentProcessor.PositionedToken("Col 1 para end.", 50f, 230f, 95f, 105f, 10f))
        tokens.add(DocumentProcessor.PositionedToken("Col 2 para end.", 310f, 490f, 95f, 105f, 10f))
        tokens.add(DocumentProcessor.PositionedToken("Col 1 more text.", 50f, 230f, 120f, 130f, 10f))
        tokens.add(DocumentProcessor.PositionedToken("Col 2 more text.", 310f, 490f, 120f, 130f, 10f))
        tokens.add(DocumentProcessor.PositionedToken("Col 1 final text.", 50f, 230f, 145f, 155f, 10f))
        tokens.add(DocumentProcessor.PositionedToken("Col 2 final text.", 310f, 490f, 145f, 155f, 10f))

        // Full-width footer at bottom (Y = 220)
        tokens.add(DocumentProcessor.PositionedToken("Page 1 Footer Note Spanning Both Columns", 50f, 500f, 210f, 220f, 9f))

        val result = DocumentProcessor.reconstructPageText(tokens, "fallback")

        val titleIdx = result.indexOf("Full Paper Title")
        val col1Idx = result.indexOf("Col 1 para start.")
        val col2Idx = result.indexOf("Col 2 para start.")
        val footerIdx = result.indexOf("Page 1 Footer Note")

        assertTrue("Title must come first", titleIdx < col1Idx)
        assertTrue("Col 1 must come before Col 2", col1Idx < col2Idx)
        assertTrue("Footer must come last", col2Idx < footerIdx)
    }
}
