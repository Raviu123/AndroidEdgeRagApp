package com.siddu.offlinedocai

import android.content.Context
import com.siddu.offlinedocai.ai.llm.GGUFLLM
import com.siddu.offlinedocai.ai.llm.LlamaRunner
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test
import java.io.File

class LlamaRunnerContractTest {

    private fun createMockContext(): Context {
        val tempDir = File(System.getProperty("java.io.tmpdir"), "offlinedocai_runner_test")
        tempDir.mkdirs()
        return object : android.content.ContextWrapper(null) {
            override fun getFilesDir(): File = tempDir
        }
    }

    @Test
    fun testLlamaRunner_GracefulHandlingOnJVM() {
        val runner = LlamaRunner(createMockContext())
        // On Windows desktop JVM without Android .so, native library is not loaded
        println("LlamaRunner.isNativeLoaded on host JVM = ${runner.isNativeLoaded}")
        assertFalse("Native Android .so should not load on Windows desktop JVM", runner.isNativeLoaded)
        assertFalse("Model cannot be loaded when native library is not loaded", runner.isModelLoaded())
        assertNull("Loaded model path should be null", runner.getLoadedModelPath())

        val loadResult = runner.loadModel("non_existent_model.gguf")
        assertFalse("loadModel should fail safely without crashing", loadResult)

        // Idempotent cleanup should not crash
        runner.freeModel()
        assertFalse(runner.isModelLoaded())
    }

    @Test
    fun testGGUFLLM_EngineStateTracking_DeterministicFallbackOnJVM() = runBlocking {
        val llm = GGUFLLM(createMockContext())
        println("Initial GGUFLLM engine state: ${llm.currentEngineState}")
        assertEquals(GGUFLLM.EngineState.DETERMINISTIC_FALLBACK, llm.currentEngineState)

        val prompt = """
            You are an offline AI assistant. Use the following extracted document context to answer the user's question.

            Context:
            [Source 101, Page 3] Optical Sensor operates at 4K 60fps HDR with 3.5W power.

            Question: What is the optical sensor power?
        """.trimIndent()

        val answer = llm.generate(prompt)
        println("Generated answer: $answer")
        println("Engine state after generation: ${llm.currentEngineState}")

        assertEquals("Engine state must be explicitly DETERMINISTIC_FALLBACK on JVM", GGUFLLM.EngineState.DETERMINISTIC_FALLBACK, llm.currentEngineState)
        assertTrue("Answer must be grounded", answer.contains("3.5W"))
    }
}
