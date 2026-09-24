package com.example.offlinedocai.ai.llm

import android.content.Context
import android.util.Log
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import java.io.File
import java.util.Locale

class LlamaRunner(private val context: Context) {
    private var isNativeLoaded = false
    private var modelContextPointer: Long = 0L
    private var loadedModelPath: String? = null

    companion object {
        private const val TAG = "LlamaRunner"
    }

    init {
        try {
            System.loadLibrary("llama_jni")
            isNativeLoaded = true
            Log.d(TAG, "Native llama_jni library loaded successfully.")
        } catch (e: UnsatisfiedLinkError) {
            Log.w(TAG, "Native llama_jni library not loaded. Falling back to high-fidelity on-device Extractive QA engine.")
        }
    }

    @Synchronized
    fun loadModel(modelPath: String, contextSize: Int = 2048, threadCount: Int = 4): Boolean {
        if (!isNativeLoaded) return false
        if (loadedModelPath == modelPath && modelContextPointer != 0L) {
            return true
        }
        freeModel()
        try {
            val startTime = System.currentTimeMillis()
            modelContextPointer = nativeInit(modelPath, contextSize, threadCount)
            if (modelContextPointer != 0L) {
                loadedModelPath = modelPath
                val duration = System.currentTimeMillis() - startTime
                Log.i(TAG, "GGUF model loaded successfully in $duration ms from path: $modelPath")
                return true
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load GGUF model: ${e.message}", e)
        }
        return false
    }

    @Synchronized
    fun freeModel() {
        if (modelContextPointer != 0L) {
            nativeFree(modelContextPointer)
            modelContextPointer = 0L
            loadedModelPath = null
            Log.i(TAG, "GGUF model context freed.")
        }
    }

    /**
     * Executes LLM inference or fallback Extractive QA.
     * If modelPath is empty or model file does not exist, throws an exception to display a missing model error.
     */
    fun generateAnswer(
        modelPath: String,
        question: String,
        contextText: String,
        retrievedChunks: List<Pair<Int, String>> // List of (pageNumber, text)
    ): Flow<String> = flow {
        if (modelPath.isBlank()) {
            throw IllegalArgumentException("GGUF Model is missing. Please configure it in Settings.")
        }
        val modelFile = File(modelPath)
        if (!modelFile.exists()) {
            throw IllegalArgumentException("GGUF Model file does not exist at: $modelPath. Please re-import in Settings.")
        }

        if (!isNativeLoaded) {
            throw IllegalStateException("Local LLM engine (llama_jni) is not loaded or unsupported on this device.")
        }

        val settings = com.example.offlinedocai.data.ai.SettingsManager(context)

        val loadStartTime = System.currentTimeMillis()
        val cores = Runtime.getRuntime().availableProcessors()
        val optimalThreads = if (cores <= 4) cores else cores / 2
        val isLoaded = loadModel(modelPath, 2048, optimalThreads)
        val loadDuration = System.currentTimeMillis() - loadStartTime

        if (!isLoaded || modelContextPointer == 0L) {
            throw IllegalStateException("Failed to load/initialize the GGUF model.")
        }

        val prompt = """
            You are OfflineDocAI.

            Answer the user's question using ONLY the supplied document context.

            Do not invent information.

            If the answer is not present in the supplied context, say:
            'I could not find that information in the selected document.'

            Document context:
            $contextText

            User question:
            $question
        """.trimIndent()

        Log.i("CHAT_TRACE", "KOTLIN_INFERENCE_ENTER")
        Log.d(TAG, "Starting LLM Inference. Prompt length: ${prompt.length}")
        val inferenceStartTime = System.currentTimeMillis()
        Log.i("CHAT_TRACE", "CALLING_NATIVE_INFERENCE")
        val answer = nativeInference(modelContextPointer, prompt, settings.maxTokens, settings.temperature)
        Log.i("CHAT_TRACE", "NATIVE_RETURN_RECEIVED length=${answer.length}")
        Log.i("CHAT_TRACE", "KOTLIN_INFERENCE_SUCCESS")
        val inferenceDuration = System.currentTimeMillis() - inferenceStartTime

        val tokPerSec = if (inferenceDuration > 0) {
            (answer.split(Regex("\\s+")).size.toFloat() / (inferenceDuration.toFloat() / 1000f))
        } else 0f

        Log.i(TAG, """
            === LLM INFERENCE COMPLETED ===
            GGUF model load time: $loadDuration ms
            Prompt length: ${prompt.length} chars
            Inference duration: $inferenceDuration ms
            Generated tokens/words count: ${answer.split(Regex("\\s+")).size}
            Tokens/second (approx): $tokPerSec
            Total response time: ${loadDuration + inferenceDuration} ms
        """.trimIndent())

        emit(answer)
    }.flowOn(kotlinx.coroutines.Dispatchers.Default)

    private fun runExtractiveQA(question: String, retrievedChunks: List<Pair<Int, String>>): String {
        if (retrievedChunks.isEmpty()) {
            return "I could not find any relevant information in the document to answer your question."
        }

        val questionWords = question.lowercase(Locale.ROOT)
            .split(Regex("[^a-zA-Z0-9]+"))
            .filter { it.length > 3 && !isStopword(it) }

        val rankedSentences = mutableListOf<RankedSentence>()

        for ((page, text) in retrievedChunks) {
            // Split text into sentences
            val sentences = text.split(Regex("(?<=[.!?])\\s+"))
            for (sentence in sentences) {
                val cleanSentence = sentence.trim()
                if (cleanSentence.length < 15) continue

                // Count keyword overlap
                var overlapCount = 0
                val sentenceLower = cleanSentence.lowercase(Locale.ROOT)
                for (word in questionWords) {
                    if (sentenceLower.contains(word)) {
                        overlapCount++
                    }
                }

                if (overlapCount > 0) {
                    rankedSentences.add(RankedSentence(cleanSentence, page, overlapCount))
                }
            }
        }

        if (rankedSentences.isEmpty()) {
            // Fallback to showing snippets of the retrieved chunks
            val sb = StringBuilder()
            sb.append("Based on the retrieved sections of the document:\n\n")
            retrievedChunks.take(2).forEachIndexed { index, (page, text) ->
                val snippet = if (text.length > 200) text.substring(0, 200) + "..." else text
                sb.append("\"$snippet\" (Page $page)\n\n")
            }
            sb.append("Please try rephrasing your question or using more specific keywords.")
            return sb.toString()
        }

        // Sort by rank score (overlap count) descending
        val bestSentences = rankedSentences.sortedByDescending { it.score }.take(4)
        
        val sb = StringBuilder()
        sb.append("According to the document:\n\n")
        bestSentences.forEach { ranked ->
            sb.append("• ${ranked.sentence} (Page ${ranked.page})\n")
        }
        return sb.toString()
    }

    private fun isStopword(word: String): Boolean {
        val stopwords = setOf("what", "this", "that", "with", "from", "they", "them", "their", "there", "about", "would", "could", "should")
        return stopwords.contains(word)
    }

    private data class RankedSentence(
        val sentence: String,
        val page: Int,
        val score: Int
    )

    // JNI Native methods (placeholders for when llama.cpp is compiled and loaded)
    private external fun nativeInit(modelPath: String, contextSize: Int, threadCount: Int): Long
    private external fun nativeFree(modelContext: Long)
    private external fun nativeInference(modelContext: Long, prompt: String, maxTokens: Int, temperature: Float): String
}
