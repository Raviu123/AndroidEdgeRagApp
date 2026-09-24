package com.example.offlinedocai.data.ai

import android.content.Context
import android.util.Log
import com.example.offlinedocai.OfflineDocAIApplication
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

class BundledModelManager(
    private val context: Context,
    private val config: ModelConfig = ModelConfig()
) {
    private val TAG = "BundledModelManager"
    private val scope = CoroutineScope(Dispatchers.Main)

    private val _isEmbeddingReady = MutableStateFlow(false)
    val isEmbeddingReady: StateFlow<Boolean> = _isEmbeddingReady.asStateFlow()

    private val _isLlmReady = MutableStateFlow(false)
    val isLlmReady: StateFlow<Boolean> = _isLlmReady.asStateFlow()

    private val _isAiReady = MutableStateFlow(false)
    val isAiReady: StateFlow<Boolean> = _isAiReady.asStateFlow()

    private val _initializationState = MutableStateFlow<InitializationState>(InitializationState.Idle)
    val initializationState: StateFlow<InitializationState> = _initializationState.asStateFlow()

    sealed interface InitializationState {
        data object Idle : InitializationState
        data class Copying(val progress: Float, val message: String) : InitializationState
        data object Ready : InitializationState
        data class Failed(val error: String) : InitializationState
    }

    fun startInitialization() {
        scope.launch {
            _initializationState.value = InitializationState.Copying(0f, "Checking local models...")
            val success = withContext(Dispatchers.IO) {
                try {
                    // Check if assets are present
                    if (!verifyAssetExists(config.embeddingAssetPath) ||
                        !verifyAssetExists(config.embeddingVocabAssetPath) ||
                        !verifyAssetExists(config.llmAssetPath)
                    ) {
                        _initializationState.value = InitializationState.Failed(
                            "Required model binaries are missing from the app assets. " +
                            "Please place all-MiniLM-L6-v2.onnx and vocab.txt in assets/models/embedding/, " +
                            "and llama-3.2-1b-instruct-q4_k_m.gguf in assets/models/llm/ during build time."
                        )
                        return@withContext false
                    }

                    // Prepare target paths
                    val embeddingDir = File(context.filesDir, "models/embedding").apply { mkdirs() }
                    val llmDir = File(context.filesDir, "models/llm").apply { mkdirs() }

                    val targetOnnxFile = File(embeddingDir, File(config.embeddingAssetPath).name)
                    val targetVocabFile = File(embeddingDir, File(config.embeddingVocabAssetPath).name)
                    val targetLlmFile = File(llmDir, File(config.llmAssetPath).name)

                    // Copy Embedding ONNX Model
                    if (!targetOnnxFile.exists()) {
                        _initializationState.value = InitializationState.Copying(0.1f, "Preparing embedding model...")
                        copyAssetToFile(config.embeddingAssetPath, targetOnnxFile)
                    }
                    _isEmbeddingReady.value = true

                    // Copy Embedding Vocab
                    if (!targetVocabFile.exists()) {
                        copyAssetToFile(config.embeddingVocabAssetPath, targetVocabFile)
                    }

                    // Load embedding model
                    val app = context.applicationContext as OfflineDocAIApplication
                    app.embeddingModel.loadModel(targetOnnxFile.absolutePath)
                    val settings = SettingsManager(context)
                    settings.onnxModelPath = targetOnnxFile.absolutePath

                    // Copy GGUF LLM Model
                    if (!targetLlmFile.exists()) {
                        _initializationState.value = InitializationState.Copying(0.4f, "Copying GGUF LLM model (may take a few minutes)...")
                        copyAssetToFile(config.llmAssetPath, targetLlmFile)
                    }
                    settings.ggufModelPath = targetLlmFile.absolutePath
                    _isLlmReady.value = true

                    _isAiReady.value = true
                    _initializationState.value = InitializationState.Ready
                    true
                } catch (e: Exception) {
                    Log.e(TAG, "Initialization failed", e)
                    _initializationState.value = InitializationState.Failed("Model preparation failed: ${e.message}")
                    false
                }
            }
        }
    }

    private fun verifyAssetExists(assetPath: String): Boolean {
        return try {
            context.assets.open(assetPath).close()
            true
        } catch (e: Exception) {
            false
        }
    }

    private fun copyAssetToFile(assetPath: String, targetFile: File) {
        context.assets.open(assetPath).use { input ->
            FileOutputStream(targetFile).use { output ->
                input.copyTo(output)
            }
        }
        Log.d(TAG, "Successfully copied $assetPath to ${targetFile.absolutePath}")
    }
}
