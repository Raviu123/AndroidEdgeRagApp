package com.example.offlinedocai.ui.settings

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.offlinedocai.data.ai.SettingsManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

class SettingsViewModel(application: Application) : AndroidViewModel(application) {
    private val context = application.applicationContext
    val settingsManager = SettingsManager(context)

    private val _importState = MutableStateFlow<ImportStatus>(ImportStatus.Idle)
    val importState: StateFlow<ImportStatus> = _importState

    sealed interface ImportStatus {
        data object Idle : ImportStatus
        data object Processing : ImportStatus
        data class Success(val fileName: String, val isGguf: Boolean) : ImportStatus
        data class Error(val message: String) : ImportStatus
    }

    fun importModel(uri: Uri, isGguf: Boolean) {
        viewModelScope.launch {
            _importState.value = ImportStatus.Processing
            val success = withContext(Dispatchers.IO) {
                try {
                    val contentResolver = context.contentResolver
                    val fileName = getFileName(uri) ?: if (isGguf) "model.gguf" else "model.onnx"
                    val modelsDir = File(context.filesDir, "models").apply { mkdirs() }
                    val destFile = File(modelsDir, fileName)

                    contentResolver.openInputStream(uri)?.use { input ->
                        FileOutputStream(destFile).use { output ->
                            input.copyTo(output)
                        }
                    }

                    if (isGguf) {
                        settingsManager.ggufModelPath = destFile.absolutePath
                    } else {
                        settingsManager.onnxModelPath = destFile.absolutePath
                        // Write a basic default vocab.txt if it doesn't exist in that folder
                        val vocabFile = File(modelsDir, "vocab.txt")
                        if (!vocabFile.exists()) {
                            writeDefaultVocab(vocabFile)
                        }
                    }
                    true
                } catch (e: Exception) {
                    false
                }
            }

            if (success) {
                _importState.value = ImportStatus.Success(if (isGguf) "GGUF Model" else "ONNX Model", isGguf)
            } else {
                _importState.value = ImportStatus.Error("Failed to copy model file to internal storage.")
            }
        }
    }

    private fun getFileName(uri: Uri): String? {
        return try {
            val cursor = context.contentResolver.query(uri, null, null, null, null)
            cursor?.use {
                if (it.moveToFirst()) {
                    val nameIndex = it.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                    if (nameIndex != -1) it.getString(nameIndex) else null
                } else null
            }
        } catch (e: Exception) {
            null
        }
    }

    private fun writeDefaultVocab(file: File) {
        try {
            file.printWriter().use { out ->
                // Write standard BERT control tokens
                out.println("[PAD]")
                out.println("[UNK]")
                out.println("[CLS]")
                out.println("[SEP]")
                out.println("[MASK]")
                // Write some basic English subwords and words to avoid crashes during tokenization
                val commonTokens = listOf(
                    "the", "of", "and", "in", "to", "a", "is", "for", "on", "that", "by", "this", "with", "as", "are", "it", "an", "be",
                    "##s", "##d", "##ed", "##ing", "##er", "##ly", "##y", "##e", "##a", "##t", "##n", "##r", "##o", "##i", "##c", "##l",
                    "document", "pdf", "text", "ai", "offline", "local", "model", "chat", "vector", "search", "rag", "system", "user"
                )
                commonTokens.forEach { out.println(it) }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun resetImportState() {
        _importState.value = ImportStatus.Idle
    }
}
