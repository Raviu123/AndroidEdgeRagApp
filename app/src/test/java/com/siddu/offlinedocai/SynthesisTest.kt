package com.siddu.offlinedocai

import android.content.Context
import com.siddu.offlinedocai.ai.llm.GGUFLLM
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test
import java.io.File

class SynthesisTest {

    // Realistic resume containing 3 distinct projects (single-line continuous text from TextChunker)
    private val sampleResumeWith3Projects = """
        SIDDHARTHA BHAT S +91 7907921234 Kasaragod, Kerala Email: siddusbs2810@gmail.com LinkedIn: https://linkedin.com/in/siddharthabhat GitHub: https://github.com/siddharthabhat TECHNICAL SKILLS C, C++, Python, Kotlin, SQL, PHP, C#, JavaScript, React, Node.js, Express.js, CSS, ML/AI, OpenCV, PyTorch, Android Development PROJECTS OfflineDocAI - Complete On-Device Document Assistant with local vector search and LLM synthesis. Technologies: Kotlin, C++, ONNX, Jetpack Compose. DueGuard - Intelligent offline payment and subscription tracker with edge notifications. Technologies: Kotlin, Room, WorkManager. KisanMitra - AI-powered agricultural advisory app with multilingual voice support. Technologies: Python, PyTorch, FastAPI, Android. EDUCATION Government College of Engineering, Kannur - B.Tech in Computer Science and Engineering, CGPA: 8.5 (2020-2024). CERTIFICATIONS Deep Learning Specialization by DeepLearning.AI. AWS Certified Cloud Practitioner.
    """.trimIndent().replace("\n", " ")

    // Multi-page document where Projects span across Page 1 and Page 2
    private val sampleMultiPageContext = """
        [Page 1] SIDDHARTHA BHAT S +91 7907921234 Email: siddusbs2810@gmail.com TECHNICAL SKILLS C++, Kotlin, Python PROJECTS Project 1: OfflineDocAI - Complete On-Device Document Assistant with local vector search and LLM synthesis. Technologies: Kotlin, C++, ONNX.
        [Page 2] Project 2: DueGuard - Intelligent offline payment and subscription tracker. Technologies: Kotlin, Room. Project 3: KisanMitra - AI-powered agricultural advisory platform. Technologies: Python, PyTorch. EDUCATION Government College of Engineering, Kannur - B.Tech in CSE.
    """.trimIndent()

    private val sampleTechnicalDoc = """
        System Architecture and Security Specification. Overview: This document describes the OfflineDocAI local authentication and storage engine. Authentication: User authentication is handled entirely locally using biometric fingerprint prompts and AES-256 encrypted session keys stored in Android Keystore. Vector Store: Retrieval uses cosine similarity on GGUF embeddings with 384 dimensions.
    """.trimIndent().replace("\n", " ")

    private fun createMockContext(): Context {
        val tempDir = File(System.getProperty("java.io.tmpdir"), "offlinedocai_test")
        tempDir.mkdirs()
        return object : android.content.ContextWrapper(null) {
            override fun getFilesDir(): File = tempDir
        }
    }

    private fun runQuery(contextText: String, question: String, page: Int = 1): String = runBlocking {
        val llm = GGUFLLM(createMockContext())
        val prompt = if (contextText.contains("[Page ")) {
            """
                You are an offline AI assistant. Use the following extracted document context to answer the user's question.

                Context:
                $contextText

                Question: $question

                Answer concisely, referencing the page numbers when appropriate.
            """.trimIndent()
        } else {
            """
                You are an offline AI assistant. Use the following extracted document context to answer the user's question.

                Context:
                [Page $page] $contextText

                Question: $question

                Answer concisely, referencing the page numbers when appropriate.
            """.trimIndent()
        }
        llm.generate(prompt)
    }

    @Test
    fun testA_Projects_AllProjectsIncluded() {
        val answer = runQuery(sampleResumeWith3Projects, "projects")
        println("QUERY: projects\nANSWER:\n$answer\n")
        assertTrue("Project 1 (OfflineDocAI) must be present", answer.contains("OfflineDocAI"))
        assertTrue("Project 2 (DueGuard) must be present", answer.contains("DueGuard"))
        assertTrue("Project 3 (KisanMitra) must be present", answer.contains("KisanMitra"))
        assertFalse("Unrelated education section must NOT appear", answer.contains("Government College") || answer.contains("CGPA"))
        assertFalse("Unrelated phone must NOT appear", answer.contains("7907921234"))
        assertFalse("Unrelated email must NOT appear", answer.contains("siddusbs2810@gmail.com"))
        assertTrue("Page citation must appear", answer.contains("(Page 1)"))
    }

    @Test
    fun testB_WhatProjectsAreMentioned() {
        val answer = runQuery(sampleResumeWith3Projects, "What projects are mentioned?")
        println("QUERY: What projects are mentioned?\nANSWER:\n$answer\n")
        assertTrue("OfflineDocAI must be present", answer.contains("OfflineDocAI"))
        assertTrue("DueGuard must be present", answer.contains("DueGuard"))
        assertTrue("KisanMitra must be present", answer.contains("KisanMitra"))
        assertFalse("Unrelated education must NOT appear", answer.contains("Government College"))
    }

    @Test
    fun testC_TellMeAboutTheProjects() {
        val answer = runQuery(sampleResumeWith3Projects, "Tell me about the projects")
        println("QUERY: Tell me about the projects\nANSWER:\n$answer\n")
        assertTrue("OfflineDocAI must be present", answer.contains("OfflineDocAI"))
        assertTrue("DueGuard must be present", answer.contains("DueGuard"))
        assertTrue("KisanMitra must be present", answer.contains("KisanMitra"))
    }

    @Test
    fun testD_WhatAreAllTheProjects_MultiPage() {
        val answer = runQuery(sampleMultiPageContext, "What are all the projects?")
        println("QUERY: What are all the projects? (MultiPage)\nANSWER:\n$answer\n")
        assertTrue("Project 1 from Page 1 must be present", answer.contains("OfflineDocAI"))
        assertTrue("Project 2 from Page 2 must be present", answer.contains("DueGuard"))
        assertTrue("Project 3 from Page 2 must be present", answer.contains("KisanMitra"))
        assertTrue("Must cite Page 1", answer.contains("(Page 1)"))
        assertTrue("Must cite Page 2", answer.contains("(Page 2)"))
        assertFalse("Unrelated education must NOT appear", answer.contains("Government College"))
    }

    @Test
    fun testE_Skills() {
        val answer = runQuery(sampleResumeWith3Projects, "skills")
        println("QUERY: skills\nANSWER:\n$answer\n")
        assertTrue("Should contain Python", answer.contains("Python"))
        assertTrue("Should contain Kotlin", answer.contains("Kotlin"))
        assertTrue("Should contain C++", answer.contains("C++"))
        assertFalse("Phone number must NOT appear in skills", answer.contains("7907921234"))
        assertFalse("Email must NOT appear in skills", answer.contains("siddusbs2810@gmail.com"))
        assertFalse("Education must NOT appear in skills", answer.contains("Government College") || answer.contains("CGPA"))
        assertFalse("Projects must NOT appear in skills", answer.contains("OfflineDocAI - Complete") || answer.contains("DueGuard"))
    }

    @Test
    fun testF_Certifications() {
        val answer = runQuery(sampleResumeWith3Projects, "certifications")
        println("QUERY: certifications\nANSWER:\n$answer\n")
        assertTrue("Should contain Deep Learning Specialization", answer.contains("Deep Learning Specialization"))
        assertTrue("Should contain AWS Certified", answer.contains("AWS Certified"))
        assertFalse("Phone must NOT appear", answer.contains("7907921234"))
        assertFalse("Email must NOT appear", answer.contains("siddusbs2810@gmail.com"))
        assertFalse("Skills list must NOT appear", answer.contains("C, C++, Python"))
    }

    @Test
    fun testG_PhoneNumber() {
        val answer = runQuery(sampleResumeWith3Projects, "phone number")
        println("QUERY: phone number\nANSWER:\n$answer\n")
        assertTrue(answer.contains("+91 7907921234") || answer.contains("7907921234"))
        assertFalse("Unrelated email should not appear", answer.contains("siddusbs2810@gmail.com"))
        assertFalse("Unrelated projects should not appear", answer.contains("OfflineDocAI"))
    }

    @Test
    fun testH_Email() {
        val answer = runQuery(sampleResumeWith3Projects, "email")
        println("QUERY: email\nANSWER:\n$answer\n")
        assertTrue(answer.contains("siddusbs2810@gmail.com"))
        assertFalse("Unrelated phone should not appear", answer.contains("7907921234"))
        assertFalse("Unrelated projects should not appear", answer.contains("OfflineDocAI"))
    }

    @Test
    fun testI_WhatIsThisDocumentAbout() {
        val answer = runQuery(sampleResumeWith3Projects, "What is this document about?")
        println("QUERY: What is this document about?\nANSWER:\n$answer\n")
        assertTrue("Should contain Document Overview / Summary", answer.contains("Document Overview") || answer.contains("Summary"))
    }

    @Test
    fun testJ_HowDoesAuthenticationWork() {
        val answer = runQuery(sampleTechnicalDoc, "How does authentication work?", page = 2)
        println("QUERY: How does authentication work?\nANSWER:\n$answer\n")
        assertTrue("Should explain biometric/AES-256 authentication", answer.contains("biometric") || answer.contains("AES-256") || answer.contains("Keystore"))
        assertTrue("Should cite Page 2", answer.contains("(Page 2)"))
        assertFalse("Should not dump unrelated vector store section", answer.contains("Vector Store: Retrieval uses cosine"))
    }
}
