package com.siddu.offlinedocai

import android.content.Context
import com.siddu.offlinedocai.ai.llm.GGUFLLM
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import java.io.File

class MultiModelBenchmarkTest {

    private lateinit var context: Context
    private lateinit var ggufLlm: GGUFLLM

    private fun createMockContext(): Context {
        val tempDir = File(System.getProperty("java.io.tmpdir"), "offlinedocai_multi_model_test")
        tempDir.mkdirs()
        return object : android.content.ContextWrapper(null) {
            override fun getFilesDir(): File = tempDir
        }
    }

    @Before
    fun setup() {
        context = createMockContext()
        ggufLlm = GGUFLLM(context)
    }

    @Test
    fun testSupportedModelEnums() {
        val qwen = GGUFLLM.SupportedModel.QWEN_0_5B
        assertEquals("qwen2.5-0.5b-instruct-q4_k_m.gguf", qwen.filename)
        assertEquals("Qwen 2.5 0.5B Instruct", qwen.displayName)
        assertEquals("qwen2", qwen.architecture)

        val llama = GGUFLLM.SupportedModel.LLAMA_1B
        assertEquals("Llama-3.2-1B-Instruct-Q4_K_M.gguf", llama.filename)
        assertEquals("Llama 3.2 1B Instruct", llama.displayName)
        assertEquals("llama", llama.architecture)

        assertEquals(qwen, GGUFLLM.SupportedModel.fromFilename("qwen2.5-0.5b-instruct-q4_k_m.gguf"))
        assertEquals(llama, GGUFLLM.SupportedModel.fromFilename("Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
        assertNull(GGUFLLM.SupportedModel.fromFilename("unknown-model.gguf"))
    }

    @Test
    fun testQwenPromptFormatting() {
        val contextText = "[Source 1, Page 3]\nApple tree grafting requires clean cuts."
        val question = "What does apple tree grafting require?"
        val prompt = ggufLlm.buildQwenPrompt(contextText, question)

        assertTrue("Must contain Qwen system start token", prompt.contains("<|im_start|>system\n"))
        assertTrue("Must contain Qwen system end token", prompt.contains("<|im_end|>\n<|im_start|>user\n"))
        assertTrue("Must contain context text", prompt.contains(contextText))
        assertTrue("Must contain question", prompt.contains(question))
        assertTrue("Must end with Qwen assistant prefix", prompt.endsWith("<|im_start|>assistant\n"))
        assertFalse("Must not contain Llama tokens", prompt.contains("<|start_header_id|>"))
    }

    @Test
    fun testLlamaPromptFormatting() {
        val contextText = "[Source 1, Page 3]\nApple tree grafting requires clean cuts."
        val question = "What does apple tree grafting require?"
        val prompt = ggufLlm.buildLlamaPrompt(contextText, question)

        assertTrue("Must start with Llama begin_of_text", prompt.startsWith("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"))
        assertTrue("Must contain Llama eot_id", prompt.contains("<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"))
        assertTrue("Must contain context text", prompt.contains(contextText))
        assertTrue("Must contain question", prompt.contains(question))
        assertTrue("Must end with Llama assistant header", prompt.endsWith("<|start_header_id|>assistant<|end_header_id|>\n\n"))
        assertFalse("Must not contain Qwen tokens", prompt.contains("<|im_start|>"))
    }

    @Test
    fun testBuildPromptForModelDispatch() {
        val contextText = "[Source 1, Page 1]\nTest context."
        val question = "Test question?"

        val qwen15Prompt = ggufLlm.buildPromptForModel(GGUFLLM.SupportedModel.QWEN_1_5B, contextText, question)
        assertTrue(qwen15Prompt.startsWith("<|im_start|>system\n"))

        val qwenPrompt = ggufLlm.buildPromptForModel(GGUFLLM.SupportedModel.QWEN_0_5B, contextText, question)
        assertTrue(qwenPrompt.startsWith("<|im_start|>system\n"))

        val llamaPrompt = ggufLlm.buildPromptForModel(GGUFLLM.SupportedModel.LLAMA_1B, contextText, question)
        assertTrue(llamaPrompt.startsWith("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"))
    }

    @Test
    fun testModelSelectionActiveModel() {
        assertEquals(GGUFLLM.SupportedModel.QWEN_1_5B, ggufLlm.getActiveModel())

        ggufLlm.selectModel(GGUFLLM.SupportedModel.LLAMA_1B)
        assertEquals(GGUFLLM.SupportedModel.LLAMA_1B, ggufLlm.getActiveModel())

        ggufLlm.selectModel(GGUFLLM.SupportedModel.QWEN_0_5B)
        assertEquals(GGUFLLM.SupportedModel.QWEN_0_5B, ggufLlm.getActiveModel())

        ggufLlm.selectModel(GGUFLLM.SupportedModel.QWEN_1_5B)
        assertEquals(GGUFLLM.SupportedModel.QWEN_1_5B, ggufLlm.getActiveModel())
    }

    @Test
    fun testArithmeticAidInBothPrompts() {
        val contextText = "Product A costs 100 USD. Product B costs 150 USD."
        val question = "What is the difference in cost between Product B and Product A?"
        val aid = ggufLlm.detectAndComputeArithmetic(question, contextText)
        assertNotNull("Should detect arithmetic aid", aid)

        val qwenPrompt = ggufLlm.buildQwenPrompt(contextText, question, aid)
        assertTrue(qwenPrompt.contains("Calculation Aid (verified arithmetic):"))
        assertTrue(qwenPrompt.contains(aid!!))

        val llamaPrompt = ggufLlm.buildLlamaPrompt(contextText, question, aid)
        assertTrue(llamaPrompt.contains("Calculation Aid (verified arithmetic):"))
        assertTrue(llamaPrompt.contains(aid))
    }
}
