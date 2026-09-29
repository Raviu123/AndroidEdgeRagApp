package com.siddu.offlinedocai

import com.siddu.offlinedocai.ai.llm.ParsedSourceBlock
import com.siddu.offlinedocai.ai.validation.CitationValidator
import com.siddu.offlinedocai.ai.validation.FallbackLevel
import com.siddu.offlinedocai.ai.validation.RawCitation
import com.siddu.offlinedocai.ai.validation.ValidationStatus
import org.junit.Assert.*
import org.junit.Test

/**
 * Comprehensive verification test suite for Phase 6: CitationValidator.kt — Evidence and Claim Validation.
 *
 * Verifies:
 * Tests A through U:
 * A. VALID CITATION (Source exists and page matches)
 * B. INVALID SOURCE (Fabricated source ID rejected)
 * C. INVALID PAGE (Mismatched page rejected)
 * D. PAGE SPAN (10-11 accepts 10 and 11, rejects 12)
 * E. EXACT NUMBER (3200 mAh supported)
 * F. WRONG NUMBER (5000 mAh rejected)
 * G. EXACT DATE (Correct date accepted)
 * H. WRONG DATE (Incorrect date rejected)
 * I. EXACT URL (Correct URL accepted)
 * J. WRONG URL (Incorrect URL rejected)
 * K. PARAPHRASE (Reasonable paraphrase accepted via stem normalization)
 * L. UNSUPPORTED ADDITION (Supported clause preserved; unsupported addition removed)
 * M. MULTI-SOURCE (Claim supported across multiple blocks verified)
 * N. UNCITED CLAIM (Uncited hallucination rejected)
 * O. MULTI-DOCUMENT (Cross-document citation rejected)
 * P. EXHAUSTIVE COMPLETE (All items verified when retrieval unconstrained)
 * Q. EXHAUSTIVE BUDGET LIMITED (Budget-limited completeness claim softened)
 * R. FIND_ALL (Every distinct occurrence validated independently)
 * S. COMPARISON (Independent entity validation in comparative claims)
 * T. FALLBACK (Zero valid claims triggers Level 3 grounded extractive fallback)
 * U. NO-HALLUCINATION (Absent fact triggers Level 4 not-found response)
 *
 * Real Document Tests: 9 Queries on NASA Systems Engineering Handbook corpus.
 */
class CitationValidatorTest {

    // =========================================================================
    // PART 1: TESTS A THROUGH U
    // =========================================================================

    @Test
    fun testA_ValidCitation_Accepted() {
        val blocks = listOf(
            ParsedSourceBlock(sourceId = 42L, page = 12, pageSpan = "12", section = "Design", chunkIndex = 5, text = "The engine operates at 2400 RPM.", documentId = 1L)
        )
        val raw = RawCitation(rawText = "[Source 42, Page 12]", sourceId = 42L, page = 12, documentId = 1L)
        val validated = CitationValidator.validateCitation(raw, blocks)

        assertNotNull("Citation must be valid", validated)
        assertEquals(42L, validated!!.sourceId)
        assertEquals(12, validated.page)
    }

    @Test
    fun testB_InvalidSource_Rejected() {
        val blocks = listOf(
            ParsedSourceBlock(sourceId = 42L, page = 12, pageSpan = "12", section = "Design", chunkIndex = 5, text = "The engine operates at 2400 RPM.", documentId = 1L)
        )
        val raw = RawCitation(rawText = "[Source 999, Page 12]", sourceId = 999L, page = 12)
        val validated = CitationValidator.validateCitation(raw, blocks)

        assertNull("Fabricated source 999 must be rejected", validated)
    }

    @Test
    fun testC_InvalidPage_Rejected() {
        val blocks = listOf(
            ParsedSourceBlock(sourceId = 42L, page = 12, pageSpan = "12", section = "Design", chunkIndex = 5, text = "The engine operates at 2400 RPM.", documentId = 1L)
        )
        val raw = RawCitation(rawText = "[Source 42, Page 99]", sourceId = 42L, page = 99)
        val validated = CitationValidator.validateCitation(raw, blocks)

        assertNull("Mismatched page 99 must be rejected", validated)
    }

    @Test
    fun testD_PageSpan_SpanAcceptsAndRejects() {
        val blocks = listOf(
            ParsedSourceBlock(sourceId = 50L, page = 10, pageSpan = "10-11", section = "Telemetry", chunkIndex = 2, text = "Telemetry packet size is 512 bytes.", documentId = 1L)
        )
        val validPage10 = CitationValidator.validateCitation(RawCitation("[Source 50, Page 10]", 50L, 10), blocks)
        val validPage11 = CitationValidator.validateCitation(RawCitation("[Source 50, Page 11]", 50L, 11), blocks)
        val invalidPage12 = CitationValidator.validateCitation(RawCitation("[Source 50, Page 12]", 50L, 12), blocks)

        assertNotNull("Page 10 in 10-11 must be accepted", validPage10)
        assertNotNull("Page 11 in 10-11 must be accepted", validPage11)
        assertNull("Page 12 outside 10-11 must be rejected", invalidPage12)
    }

    @Test
    fun testE_ExactNumber_Supported() {
        val blocks = listOf(
            ParsedSourceBlock(sourceId = 42L, page = 8, pageSpan = "8", section = "Power", chunkIndex = 1, text = "Battery capacity is 3200 mAh.", documentId = 1L)
        )
        val answer = "The battery capacity is 3200 mAh. [Source 42, Page 8]"
        val result = CitationValidator.validate(answer, blocks)

        assertEquals("Must be fully supported", FallbackLevel.L1_FULLY_SUPPORTED, result.fallbackLevel)
        assertTrue("Must retain 3200 mAh", result.validatedAnswer.contains("3200 mAh"))
        assertEquals(ValidationStatus.SUPPORTED, result.claimResults.first().status)
    }

    @Test
    fun testF_WrongNumber_Rejected() {
        val blocks = listOf(
            ParsedSourceBlock(sourceId = 42L, page = 8, pageSpan = "8", section = "Power", chunkIndex = 1, text = "Battery capacity is 3200 mAh.", documentId = 1L)
        )
        val answer = "The battery capacity is 5000 mAh. [Source 42, Page 8]"
        val result = CitationValidator.validate(answer, blocks)

        assertNotEquals("5000 mAh cannot be fully supported", FallbackLevel.L1_FULLY_SUPPORTED, result.fallbackLevel)
        assertEquals(ValidationStatus.UNSUPPORTED, result.claimResults.first().status)
        assertTrue(result.claimResults.first().reason.contains("5000") || result.claimResults.first().reason.contains("metric"))
    }

    @Test
    fun testG_ExactDate_Supported() {
        val blocks = listOf(
            ParsedSourceBlock(sourceId = 10L, page = 1, pageSpan = "1", section = "Milestones", chunkIndex = 0, text = "The project launched on 2024-05-01.", documentId = 1L)
        )
        val answer = "The project launched on 2024-05-01. [Source 10, Page 1]"
        val result = CitationValidator.validate(answer, blocks)

        assertEquals(FallbackLevel.L1_FULLY_SUPPORTED, result.fallbackLevel)
        assertEquals(ValidationStatus.SUPPORTED, result.claimResults.first().status)
    }

    @Test
    fun testH_WrongDate_Rejected() {
        val blocks = listOf(
            ParsedSourceBlock(sourceId = 10L, page = 1, pageSpan = "1", section = "Milestones", chunkIndex = 0, text = "The project launched on 2024-05-01.", documentId = 1L)
        )
        val answer = "The project launched on 2025-05-01. [Source 10, Page 1]"
        val result = CitationValidator.validate(answer, blocks)

        assertNotEquals(FallbackLevel.L1_FULLY_SUPPORTED, result.fallbackLevel)
        assertEquals(ValidationStatus.UNSUPPORTED, result.claimResults.first().status)
    }

    @Test
    fun testI_ExactUrl_Supported() {
        val blocks = listOf(
            ParsedSourceBlock(sourceId = 20L, page = 2, pageSpan = "2", section = "Code", chunkIndex = 0, text = "Repository is located at https://github.com/nasa/se-handbook.", documentId = 1L)
        )
        val answer = "Repository is located at https://github.com/nasa/se-handbook. [Source 20, Page 2]"
        val result = CitationValidator.validate(answer, blocks)

        assertEquals(FallbackLevel.L1_FULLY_SUPPORTED, result.fallbackLevel)
        assertEquals(ValidationStatus.SUPPORTED, result.claimResults.first().status)
    }

    @Test
    fun testJ_WrongUrl_Rejected() {
        val blocks = listOf(
            ParsedSourceBlock(sourceId = 20L, page = 2, pageSpan = "2", section = "Code", chunkIndex = 0, text = "Repository is located at https://github.com/nasa/se-handbook.", documentId = 1L)
        )
        val answer = "Repository is located at https://github.com/fake/repo. [Source 20, Page 2]"
        val result = CitationValidator.validate(answer, blocks)

        assertNotEquals(FallbackLevel.L1_FULLY_SUPPORTED, result.fallbackLevel)
        assertEquals(ValidationStatus.UNSUPPORTED, result.claimResults.first().status)
    }

    @Test
    fun testK_Paraphrase_Accepted() {
        val blocks = listOf(
            ParsedSourceBlock(sourceId = 30L, page = 5, pageSpan = "5", section = "Verification", chunkIndex = 0,
                text = "The system performs verification through test, analysis, inspection, or demonstration.", documentId = 1L)
        )
        val answer = "The system can be verified using testing, analysis, inspection, or demonstration. [Source 30, Page 5]"
        val result = CitationValidator.validate(answer, blocks)

        assertEquals(FallbackLevel.L1_FULLY_SUPPORTED, result.fallbackLevel)
        assertEquals(ValidationStatus.SUPPORTED, result.claimResults.first().status)
    }

    @Test
    fun testL_UnsupportedAddition_DetectedAndPruned() {
        val blocks = listOf(
            ParsedSourceBlock(sourceId = 40L, page = 6, pageSpan = "6", section = "Phases", chunkIndex = 0,
                text = "Phase A establishes the initial system concept.", documentId = 1L)
        )
        val answer = "Phase A establishes the initial system concept and guarantees the final production design. [Source 40, Page 6]"
        val result = CitationValidator.validate(answer, blocks)

        val claimResult = result.claimResults.first()
        assertEquals("Must detect partially supported compound sentence", ValidationStatus.PARTIALLY_SUPPORTED, claimResult.status)
        assertTrue("Must retain supported portion", claimResult.validatedText.contains("Phase A establishes the initial system concept"))
        assertFalse("Must strip unsupported addition", claimResult.validatedText.contains("guarantees the final production design"))
    }

    @Test
    fun testM_MultiSource_JointlySupported() {
        val blocks = listOf(
            ParsedSourceBlock(sourceId = 101L, page = 1, pageSpan = "1", section = "PhaseA", chunkIndex = 0, text = "Phase A establishes system requirements.", documentId = 1L),
            ParsedSourceBlock(sourceId = 102L, page = 2, pageSpan = "2", section = "PhaseB", chunkIndex = 1, text = "Phase B freezes baseline requirements.", documentId = 1L)
        )
        val answer = "Phase A establishes system requirements, while Phase B freezes the baseline requirements. [Source 101, Page 1] [Source 102, Page 2]"
        val result = CitationValidator.validate(answer, blocks)

        assertEquals(FallbackLevel.L1_FULLY_SUPPORTED, result.fallbackLevel)
        val claim = result.claimResults.first()
        assertEquals(ValidationStatus.SUPPORTED, claim.status)
        assertEquals("Must cite both sources", 2, claim.citations.size)
    }

    @Test
    fun testN_UncitedClaim_UnsupportedFactRejected() {
        val blocks = listOf(
            ParsedSourceBlock(sourceId = 50L, page = 1, pageSpan = "1", section = "Power", chunkIndex = 0, text = "Power is provided by solar panels and lithium batteries.", documentId = 1L)
        )
        val answer = "The engine runs on liquid hydrogen."
        val result = CitationValidator.validate(answer, blocks)

        assertNotEquals(FallbackLevel.L1_FULLY_SUPPORTED, result.fallbackLevel)
        assertEquals(ValidationStatus.UNSUPPORTED, result.claimResults.first().status)
    }

    @Test
    fun testO_MultiDocument_CrossDocumentCitationRejected() {
        val blocks = listOf(
            ParsedSourceBlock(sourceId = 101L, page = 5, pageSpan = "5", section = "Handbook", chunkIndex = 0, text = "Phase A establishes system concepts.", documentId = 1L),
            ParsedSourceBlock(sourceId = 201L, page = 2, pageSpan = "2", section = "Software", chunkIndex = 0, text = "Phase A creates UI wireframes.", documentId = 2L)
        )
        // Citation falsely attributes Doc 1 / Source 101 to Doc 2 wireframes claim
        val answer = "Phase A creates UI wireframes. [Doc 1, Source 101, Page 5]"
        val result = CitationValidator.validate(answer, blocks)

        assertNotEquals(FallbackLevel.L1_FULLY_SUPPORTED, result.fallbackLevel)
        assertEquals(ValidationStatus.UNSUPPORTED, result.claimResults.first().status)
    }

    @Test
    fun testP_ExhaustiveComplete_AllItemsVerified() {
        val blocks = listOf(
            ParsedSourceBlock(sourceId = 1L, page = 1, pageSpan = "1", section = "Projects", chunkIndex = 0,
                text = "Active missions: 1. Apollo, 2. Gemini, 3. Mercury, 4. Artemis.", documentId = 1L)
        )
        val answer = "Based on the document, the items are:\n• Apollo (Page 1)\n• Gemini (Page 1)\n• Mercury (Page 1)\n• Artemis (Page 1)"
        val result = CitationValidator.validate(answer, blocks, isBudgetLimited = false)

        assertEquals(FallbackLevel.L1_FULLY_SUPPORTED, result.fallbackLevel)
        assertTrue(result.validatedAnswer.contains("Apollo"))
        assertTrue(result.validatedAnswer.contains("Artemis"))
    }

    @Test
    fun testQ_ExhaustiveBudgetLimited_SoftenedCompletenessClaim() {
        val blocks = listOf(
            ParsedSourceBlock(sourceId = 1L, page = 1, pageSpan = "1", section = "Projects", chunkIndex = 0,
                text = "Subsystem components include Optical Sensor and Infrared Camera.", documentId = 1L)
        )
        val answer = "These are all projects mentioned: Optical Sensor, Infrared Camera."
        val result = CitationValidator.validate(answer, blocks, isBudgetLimited = true)

        assertFalse("Must not claim absolute completeness when budget-limited", result.validatedAnswer.contains("These are all"))
        assertTrue("Must include softened retrieved framing", result.validatedAnswer.contains("the retrieved") || result.validatedAnswer.contains("retrieved"))
    }

    @Test
    fun testR_FindAll_SeparateOccurrencesIndependentlyValidated() {
        val blocks = listOf(
            ParsedSourceBlock(sourceId = 1L, page = 3, pageSpan = "3", section = "Sec1", chunkIndex = 0, text = "Authentication uses biometric sensors.", documentId = 1L),
            ParsedSourceBlock(sourceId = 2L, page = 12, pageSpan = "12", section = "Sec2", chunkIndex = 0, text = "Authentication session keys expire in 1 hour.", documentId = 1L)
        )
        val answer = "• Authentication uses biometric sensors. [Source 1, Page 3]\n• Authentication session keys expire in 1 hour. [Source 2, Page 12]"
        val result = CitationValidator.validate(answer, blocks)

        assertEquals(FallbackLevel.L1_FULLY_SUPPORTED, result.fallbackLevel)
        assertEquals(2, result.claimResults.size)
        assertTrue(result.claimResults.all { it.status == ValidationStatus.SUPPORTED })
    }

    @Test
    fun testS_Comparison_BothEntitiesIndependentlyValidated() {
        val blocks = listOf(
            ParsedSourceBlock(sourceId = 1L, page = 5, pageSpan = "5", section = "PhaseA", chunkIndex = 0, text = "Phase A develops the system concept.", documentId = 1L),
            ParsedSourceBlock(sourceId = 2L, page = 10, pageSpan = "10", section = "PhaseB", chunkIndex = 0, text = "Phase B establishes preliminary design.", documentId = 1L)
        )
        val answer = "• Phase A develops the system concept. (Page 5)\n• Phase B establishes preliminary design. (Page 10)"
        val result = CitationValidator.validate(answer, blocks)

        assertEquals(FallbackLevel.L1_FULLY_SUPPORTED, result.fallbackLevel)
        assertEquals(2, result.claimResults.size)
        assertEquals(ValidationStatus.SUPPORTED, result.claimResults[0].status)
        assertEquals(ValidationStatus.SUPPORTED, result.claimResults[1].status)
    }

    @Test
    fun testT_Fallback_ZeroValidClaimsExtractsSnippet() {
        val blocks = listOf(
            ParsedSourceBlock(sourceId = 1L, page = 1, pageSpan = "1", section = "Power", chunkIndex = 0,
                text = "The satellite is powered by solar arrays generating 500 Watts.", documentId = 1L)
        )
        val hallucinatedAnswer = "The spacecraft is propelled by antimatter warp drives launched in 2999."
        val result = CitationValidator.validate(hallucinatedAnswer, blocks, query = "How is it powered?")

        assertEquals(FallbackLevel.L3_EXTRACTIVE_FALLBACK, result.fallbackLevel)
        assertTrue("Must extract grounded evidence snippet", result.validatedAnswer.contains("solar arrays") || result.validatedAnswer.contains("500 Watts"))
        assertTrue("Must cite Page 1", result.validatedAnswer.contains("Page 1"))
    }

    @Test
    fun testU_NoHallucination_AbsentFactReturnsNotFound() {
        val blocks = listOf(
            ParsedSourceBlock(sourceId = 1L, page = 1, pageSpan = "1", section = "Power", chunkIndex = 0, text = "Solar arrays generate power.", documentId = 1L)
        )
        val notFoundAnswer = "I searched the provided document(s), but could not find information answering \"What is the secret launch code?\"."
        val result = CitationValidator.validate(notFoundAnswer, blocks, query = "What is the secret launch code?")

        assertEquals(FallbackLevel.L4_NOT_FOUND, result.fallbackLevel)
        assertTrue(result.validatedAnswer.contains("could not find information") || result.validatedAnswer.contains("I couldn't find"))
    }

    // =========================================================================
    // PART 2: REAL DOCUMENT VALIDATION (9 NASA SE HANDBOOK QUERIES)
    // =========================================================================

    private val nasaEvidenceBlocks = listOf(
        ParsedSourceBlock(sourceId = 601L, page = 3, pageSpan = "3", section = "SE Engine", chunkIndex = 0,
            text = "The Systems Engineering (SE) engine is a cyclical, iterative process consisting of 17 processes categorized into System Design Processes, Product Realization Processes, and Technical Management Processes.",
            documentId = 10L),
        ParsedSourceBlock(sourceId = 602L, page = 10, pageSpan = "10", section = "Cryptography", chunkIndex = 1,
            text = "Quantum Encryption Protocols: Page 10 outlines post-quantum lattice-based cryptography, detailing Kyber key encapsulation and Dilithium digital signatures for next-generation aerospace telemetry.",
            documentId = 10L),
        ParsedSourceBlock(sourceId = 603L, page = 15, pageSpan = "15-16", section = "Fundamentals", chunkIndex = 2,
            text = "Chapter 2: Fundamentals of Systems Engineering. Systems engineering provides the technical foundation for the successful realization of complex aerospace systems.",
            documentId = 10L),
        ParsedSourceBlock(sourceId = 604L, page = 18, pageSpan = "18", section = "Life Cycle", chunkIndex = 3,
            text = "The systems engineering life cycle spans seven distinct project phases, progressing from concept studies (Pre-Phase A) through concept development (Phase A), preliminary design (Phase B), final design (Phase C), fabrication and integration (Phase D), operations (Phase E), to closeout (Phase F).",
            documentId = 10L),
        ParsedSourceBlock(sourceId = 605L, page = 21, pageSpan = "21", section = "Phase A", chunkIndex = 4,
            text = "Phase A (Concept and Technology Development) determines the feasible system concept, defines mission architecture, establishes top-level system requirements, and develops initial risk management plans.",
            documentId = 10L),
        ParsedSourceBlock(sourceId = 606L, page = 25, pageSpan = "25", section = "Phase B", chunkIndex = 5,
            text = "Phase B (Preliminary Design and Technology Completion) establishes the preliminary design, freezes the baseline requirements, demonstrates critical technology maturity at TRL 6, and culminates in the Preliminary Design Review (PDR).",
            documentId = 10L),
        ParsedSourceBlock(sourceId = 607L, page = 88, pageSpan = "88", section = "Verification", chunkIndex = 6,
            text = "Section 4.3: System Verification Process provides evidence that the delivered system complies with its specified requirements through test, analysis, inspection, or demonstration.",
            documentId = 10L),
        ParsedSourceBlock(sourceId = 608L, page = 104, pageSpan = "104", section = "Validation", chunkIndex = 7,
            text = "Section 5.4: System Validation Process demonstrates that the realized system functions as intended in its operational environment to fulfill stakeholder expectations and mission objectives.",
            documentId = 10L),
        ParsedSourceBlock(sourceId = 609L, page = 120, pageSpan = "120", section = "Technical Management", chunkIndex = 8,
            text = "Technical Management Processes include: 1. Technical Planning, 2. Requirements Management, 3. Interface Management, 4. Technical Risk Management, 5. Configuration Management, 6. Technical Data Management, 7. Technical Assessment, and 8. Decision Analysis.",
            documentId = 10L)
    )

    private fun printValidationReport(queryType: String, claim: String, citation: String, evidence: String, status: ValidationStatus) {
        println("--------------------------------------------------")
        println("QUERY TYPE: $queryType")
        println("GENERATED CLAIM: $claim")
        println("→ CITATION: $citation")
        println("→ SOURCE EVIDENCE: $evidence")
        println("→ VALIDATION RESULT: $status")
        println("--------------------------------------------------\n")
    }

    @Test
    fun testRealDoc1_PageLookup() {
        val answer = "Page 10 outlines post-quantum lattice-based cryptography with Kyber and Dilithium. [Source 602, Page 10] (Page 10)"
        val result = CitationValidator.validate(answer, nasaEvidenceBlocks)

        val claim = result.claimResults.first()
        printValidationReport("1. PAGE LOOKUP", claim.originalClaim, "[Source 602, Page 10]", nasaEvidenceBlocks[1].text, claim.status)
        assertEquals(ValidationStatus.SUPPORTED, claim.status)
    }

    @Test
    fun testRealDoc2_PageRange() {
        val answer = "Chapter 2 covers fundamentals of systems engineering for complex aerospace systems. [Source 603, Page 15] (Page 15)"
        val result = CitationValidator.validate(answer, nasaEvidenceBlocks)

        val claim = result.claimResults.first()
        printValidationReport("2. PAGE RANGE", claim.originalClaim, "[Source 603, Page 15]", nasaEvidenceBlocks[2].text, claim.status)
        assertEquals(ValidationStatus.SUPPORTED, claim.status)
    }

    @Test
    fun testRealDoc3_ExactSpecification() {
        val answer = "Phase B demonstrates critical technology maturity at TRL 6 and culminates in the Preliminary Design Review (PDR). [Source 606, Page 25] (Page 25)"
        val result = CitationValidator.validate(answer, nasaEvidenceBlocks)

        val claim = result.claimResults.first()
        printValidationReport("3. EXACT SPECIFICATION", claim.originalClaim, "[Source 606, Page 25]", nasaEvidenceBlocks[5].text, claim.status)
        assertEquals(ValidationStatus.SUPPORTED, claim.status)
        assertTrue(claim.originalClaim.contains("TRL 6"))
    }

    @Test
    fun testRealDoc4_Comparison() {
        val answer = "• Phase A determines the feasible system concept. [Source 605, Page 21] (Page 21)\n• Phase B establishes preliminary design and demonstrates TRL 6. [Source 606, Page 25] (Page 25)"
        val result = CitationValidator.validate(answer, nasaEvidenceBlocks)

        assertEquals(FallbackLevel.L1_FULLY_SUPPORTED, result.fallbackLevel)
        assertEquals(2, result.claimResults.size)
        printValidationReport("4. COMPARISON", result.claimResults[0].originalClaim, "[Source 605, Page 21]", nasaEvidenceBlocks[4].text, result.claimResults[0].status)
        printValidationReport("4. COMPARISON", result.claimResults[1].originalClaim, "[Source 606, Page 25]", nasaEvidenceBlocks[5].text, result.claimResults[1].status)
        assertEquals(ValidationStatus.SUPPORTED, result.claimResults[0].status)
        assertEquals(ValidationStatus.SUPPORTED, result.claimResults[1].status)
    }

    @Test
    fun testRealDoc5_SectionQuery() {
        val answer = "Section 4.3 provides evidence that the system complies with specified requirements through test, analysis, inspection, or demonstration. [Source 607, Page 88] (Page 88)"
        val result = CitationValidator.validate(answer, nasaEvidenceBlocks)

        val claim = result.claimResults.first()
        printValidationReport("5. SECTION QUERY", claim.originalClaim, "[Source 607, Page 88]", nasaEvidenceBlocks[6].text, claim.status)
        assertEquals(ValidationStatus.SUPPORTED, claim.status)
    }

    @Test
    fun testRealDoc6_FindAll() {
        val answer = "• Verification: Section 4.3 provides evidence through test, analysis, inspection, or demonstration. [Source 607, Page 88] (Page 88)\n• Validation: Section 5.4 demonstrates that the system functions in its operational environment. [Source 608, Page 104] (Page 104)"
        val result = CitationValidator.validate(answer, nasaEvidenceBlocks)

        assertEquals(FallbackLevel.L1_FULLY_SUPPORTED, result.fallbackLevel)
        assertEquals(2, result.claimResults.size)
        printValidationReport("6. FIND ALL", result.claimResults[0].originalClaim, "[Source 607, Page 88]", nasaEvidenceBlocks[6].text, result.claimResults[0].status)
        printValidationReport("6. FIND ALL", result.claimResults[1].originalClaim, "[Source 608, Page 104]", nasaEvidenceBlocks[7].text, result.claimResults[1].status)
        assertEquals(ValidationStatus.SUPPORTED, result.claimResults[0].status)
        assertEquals(ValidationStatus.SUPPORTED, result.claimResults[1].status)
    }

    @Test
    fun testRealDoc7_AbsentFact() {
        val answer = "I searched the provided document(s), but could not find information answering \"What is the warp drive frequency for starship Enterprise?\"."
        val result = CitationValidator.validate(answer, nasaEvidenceBlocks, query = "What is the warp drive frequency for starship Enterprise?")

        printValidationReport("7. ABSENT FACT", answer, "None", "No matching evidence", ValidationStatus.UNSUPPORTED)
        assertEquals(FallbackLevel.L4_NOT_FOUND, result.fallbackLevel)
        assertTrue(result.validatedAnswer.contains("could not find information"))
    }

    @Test
    fun testRealDoc8_GlobalSummary() {
        val answer = "📄 **Document Overview & Key Summary:**\n\n• The Systems Engineering engine consists of 17 cyclical iterative processes. [Source 601, Page 3] (Page 3)\n• The systems engineering life cycle spans seven distinct project phases from Pre-Phase A to Phase F. [Source 604, Page 18] (Page 18)"
        val result = CitationValidator.validate(answer, nasaEvidenceBlocks)

        assertEquals(FallbackLevel.L1_FULLY_SUPPORTED, result.fallbackLevel)
        assertTrue(result.validatedAnswer.contains("Document Overview"))
        printValidationReport("8. GLOBAL SUMMARY", result.claimResults[0].originalClaim, "[Source 601, Page 3]", nasaEvidenceBlocks[0].text, result.claimResults[0].status)
        printValidationReport("8. GLOBAL SUMMARY", result.claimResults[1].originalClaim, "[Source 604, Page 18]", nasaEvidenceBlocks[3].text, result.claimResults[1].status)
        assertEquals(ValidationStatus.SUPPORTED, result.claimResults[0].status)
        assertEquals(ValidationStatus.SUPPORTED, result.claimResults[1].status)
    }

    @Test
    fun testRealDoc9_ExhaustiveList() {
        val answer = "Based on the document, the items are:\n• Technical Planning (Page 120)\n• Requirements Management (Page 120)\n• Configuration Management (Page 120)\n• Decision Analysis (Page 120)"
        val result = CitationValidator.validate(answer, nasaEvidenceBlocks)

        assertEquals(FallbackLevel.L1_FULLY_SUPPORTED, result.fallbackLevel)
        printValidationReport("9. EXHAUSTIVE LIST", result.claimResults.first().originalClaim, "(Page 120)", nasaEvidenceBlocks[8].text, result.claimResults.first().status)
        assertTrue(result.claimResults.all { it.status == ValidationStatus.SUPPORTED })
    }
}
