package com.siddu.offlinedocai

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.siddu.offlinedocai.ai.llm.GGUFLLM
import com.siddu.offlinedocai.ai.llm.LlamaRunner
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * On-Device Android Instrumentation Test for actual GGUF / llama.cpp execution.
 *
 * Exercises the complete native bridge:
 * Android Runtime -> libllama_jni.so -> llama.cpp -> Qwen 2.5 0.5B GGUF -> Native Tokens.
 */
@RunWith(AndroidJUnit4::class)
class LlamaNativeInferenceAndroidTest {

    @Test
    fun testRealGGUF_NativeExecutionOnAndroidDevice() = runBlocking {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val llm = GGUFLLM(appContext)

        // 1. Verify native library loaded
        assertTrue("Native library llama_jni must be loaded on Android runtime", llm.llamaRunner.isNativeLoaded)

        // 2. Resolve or copy GGUF model
        val modelFile = llm.resolveModelFile()
        assertNotNull("GGUF model file must exist in assets or filesDir", modelFile)
        assertTrue("Model file must be readable on disk", modelFile!!.exists() && modelFile.canRead())
        assertTrue("Model file size must be > 100 MB", modelFile.length() > 100 * 1024 * 1024)

        println("=== NATIVE GGUF INFERENCE VERIFICATION ===")
        println("Device Model File: ${modelFile.absolutePath} (${modelFile.length()} bytes)")

        // 3. Load model into native memory
        val runner = llm.llamaRunner
        val loadStartTime = System.currentTimeMillis()
        val loaded = runner.loadModel(
            modelPath = modelFile.absolutePath,
            contextSize = 1024,
            threadCount = 4
        )
        val loadDuration = System.currentTimeMillis() - loadStartTime
        assertTrue("Native llama_init_from_model must succeed", loaded)
        assertTrue("Runner must report model is loaded", runner.isModelLoaded())
        println("1. Native Model Load Time: $loadDuration ms")

        try {
            // 4. Test Native Prompt A
            val promptA = "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\nAnswer with exactly one word: APPLE\n<|im_end|>\n<|im_start|>assistant\n"
            val genStartTimeA = System.currentTimeMillis()
            val outputA = runner.inference(promptA, maxTokens = 16, temperature = 0.0f)
            val genDurationA = System.currentTimeMillis() - genStartTimeA
            println("2. Native Prompt A Output: '$outputA' (in $genDurationA ms)")
            assertTrue("Native inference output A must be non-empty", outputA.isNotBlank())

            // 5. Test Native Prompt B
            val promptB = "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\nAnswer with exactly one word: ORANGE\n<|im_end|>\n<|im_start|>assistant\n"
            val genStartTimeB = System.currentTimeMillis()
            val outputB = runner.inference(promptB, maxTokens = 16, temperature = 0.0f)
            val genDurationB = System.currentTimeMillis() - genStartTimeB
            println("3. Native Prompt B Output: '$outputB' (in $genDurationB ms)")
            assertTrue("Native inference output B must be non-empty", outputB.isNotBlank())

            // 6. Verify outputs respond distinctly to distinct prompts
            assertNotEquals("Distinct prompts must yield distinct neural outputs", outputA.trim().lowercase(), outputB.trim().lowercase())

            // 7. Test End-to-End Grounded Generation via GGUFLLM
            val ragPrompt = """
                Context:
                [Source 201, Page 1] Hardware Device Specification. Helpline Contact: +1-800-555-0199. Support Email: support@offlinedocai.com.

                Question: What is the support email?
            """.trimIndent()

            val ragAnswer = llm.generate(ragPrompt)
            println("4. GGUFLLM Grounded Answer: $ragAnswer")
            println("5. Final Engine State: ${llm.currentEngineState}")
            assertEquals("Engine state must be GGUF_NATIVE on Android", GGUFLLM.EngineState.GGUF_NATIVE, llm.currentEngineState)
            assertTrue("Answer must contain grounded support email", ragAnswer.contains("support@offlinedocai.com"))

        } finally {
            // 8. Release native model memory
            runner.freeModel()
            assertFalse("Model must be marked freed", runner.isModelLoaded())
            println("6. Native model context freed successfully.")
        }
    }
}
