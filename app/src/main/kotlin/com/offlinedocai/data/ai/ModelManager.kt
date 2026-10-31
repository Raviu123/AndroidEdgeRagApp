package com.offlinedocai.data.ai

import android.content.Context
import android.content.SharedPreferences
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Log
import com.offlinedocai.ai.llm.GGUFMetadata
import com.offlinedocai.ai.llm.GGUFMetadataParser
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.atomic.AtomicBoolean

enum class ModelStatus {
    NOT_INSTALLED,
    IMPORTING,
    INSTALLED,
    LOADING,
    LOADED,
    ERROR
}

data class ModelItem(
    val id: String,
    val filename: String,
    val displayName: String,
    val architecture: String,
    val parameterCount: String,
    val quantization: String,
    val contextLength: Int,
    val sizeBytes: Long,
    val importDate: Long,
    val file: File,
    val status: ModelStatus,
    val isActive: Boolean,
    val errorMessage: String? = null
)

data class ImportProgress(
    val filename: String,
    val bytesCopied: Long,
    val totalBytes: Long,
    val percent: Int
)

/**
 * Production-ready Model Manager for local GGUF models.
 *
 * Responsibilities:
 * - SAF-based model importing with atomic copy and live progress
 * - Binary GGUF validation before installation
 * - Scoped-storage compliant persistent application directory
 * - Metadata persistence across application restarts
 * - Safe model deletion (preventing deletion during active generation)
 * - Active model selection and graceful fallback
 */
class ModelManager(private val context: Context, refreshOnInit: Boolean = true) {

    private val TAG = "ModelManager"
    private val prefs: SharedPreferences = context.getSharedPreferences("offline_doc_ai_models", Context.MODE_PRIVATE)
    private val metadataFile: File = File(context.filesDir, "models/llm_metadata.json")

    private val _models = MutableStateFlow<List<ModelItem>>(emptyList())
    val models: StateFlow<List<ModelItem>> = _models.asStateFlow()

    private val _activeModel = MutableStateFlow<ModelItem?>(null)
    val activeModel: StateFlow<ModelItem?> = _activeModel.asStateFlow()

    private val _importProgress = MutableStateFlow<ImportProgress?>(null)
    val importProgress: StateFlow<ImportProgress?> = _importProgress.asStateFlow()

    private val _importError = MutableStateFlow<String?>(null)
    val importError: StateFlow<String?> = _importError.asStateFlow()

    private val isGenerating = AtomicBoolean(false)

    companion object {
        const val TARGET_MODEL_FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
        private const val KEY_ACTIVE_MODEL_FILENAME = "active_model_filename"

        @Volatile
        private var instance: ModelManager? = null

        fun getInstance(context: Context): ModelManager {
            return instance ?: synchronized(this) {
                instance ?: ModelManager(context.applicationContext, refreshOnInit = false).also { instance = it }
            }
        }
    }

    init {
        // The app-wide instance refreshes on a background thread instead (see refresh()).
        if (refreshOnInit) refreshModels()
    }

    /** Re-scans installed models off the main thread (GGUF headers are read from disk). */
    suspend fun refresh() = withContext(Dispatchers.IO) { refreshModels() }

    fun getModelDirectory(): File {
        val dir = File(context.filesDir, "models/llm")
        if (!dir.exists()) dir.mkdirs()
        return dir
    }

    fun setGenerating(generating: Boolean) {
        isGenerating.set(generating)
    }

    fun isGenerationActive(): Boolean = isGenerating.get()

    /**
     * Scans local application model storage and test tmp storage,
     * validates GGUF files, loads metadata, and restores active model.
     */
    @Synchronized
    fun refreshModels() {
        val modelDir = getModelDirectory()
        val storedMetadata = loadPersistedMetadata()

        val foundFiles = mutableListOf<File>()

        // 1. Files in app-internal persistent storage
        val localFiles = modelDir.listFiles()?.filter {
            it.isFile && it.extension.equals("gguf", ignoreCase = true) && !it.name.endsWith(".part") && it.length() > 1024
        } ?: emptyList()
        foundFiles.addAll(localFiles)

        // 2. Debug builds only: /data/local/tmp/models/llm (adb push convenience)
        val tmpDir = File("/data/local/tmp/models/llm")
        if (com.offlinedocai.BuildConfig.DEBUG && tmpDir.exists() && tmpDir.isDirectory) {
            val tmpFiles = tmpDir.listFiles()?.filter {
                it.isFile && it.extension.equals("gguf", ignoreCase = true) && it.length() > 1024
            } ?: emptyList()
            for (tf in tmpFiles) {
                if (foundFiles.none { it.name.equals(tf.name, ignoreCase = true) }) {
                    foundFiles.add(tf)
                }
            }
        }

        // Restore active selection from SharedPreferences
        val savedActiveFilename = prefs.getString(KEY_ACTIVE_MODEL_FILENAME, null)

        val updatedItems = mutableListOf<ModelItem>()

        for (file in foundFiles) {
            val cachedMeta = storedMetadata[file.name]
            val meta = if (cachedMeta != null && cachedMeta.sizeBytes == file.length() && cachedMeta.isValid) {
                cachedMeta
            } else {
                val parsed = GGUFMetadataParser.parse(file)
                parsed
            }

            val importDate = file.lastModified()
            val item = ModelItem(
                id = file.name,
                filename = file.name,
                displayName = meta.modelName.ifBlank { GGUFMetadataParser.deriveDisplayName(file.name, meta.architecture, meta.parameterCount) },
                architecture = meta.architecture,
                parameterCount = meta.parameterCount,
                quantization = meta.quantization.ifBlank { GGUFMetadataParser.extractQuantFromFilename(file.name) },
                contextLength = meta.contextLength,
                sizeBytes = file.length(),
                importDate = importDate,
                file = file,
                status = if (meta.isValid) ModelStatus.INSTALLED else ModelStatus.ERROR,
                isActive = false,
                errorMessage = meta.errorMessage
            )
            updatedItems.add(item)
        }

        // Determine active model
        var active = updatedItems.firstOrNull { it.filename.equals(savedActiveFilename, ignoreCase = true) && it.status == ModelStatus.INSTALLED }
        if (active == null) {
            // Default preference: Qwen 1.5B if installed, otherwise first installed
            active = updatedItems.firstOrNull { it.filename.equals(TARGET_MODEL_FILENAME, ignoreCase = true) && it.status == ModelStatus.INSTALLED }
                ?: updatedItems.firstOrNull { it.status == ModelStatus.INSTALLED }
        }

        val finalizedList = updatedItems.map {
            if (active != null && it.id == active.id) it.copy(isActive = true) else it
        }

        _models.value = finalizedList
        _activeModel.value = finalizedList.firstOrNull { it.isActive }

        if (_activeModel.value != null) {
            prefs.edit().putString(KEY_ACTIVE_MODEL_FILENAME, _activeModel.value!!.filename).apply()
        }

        savePersistedMetadata(finalizedList)
    }

    /**
     * Safely selects an installed model as the active LLM and persists choice.
     */
    @Synchronized
    fun setActiveModel(modelId: String): Boolean {
        val currentList = _models.value
        val target = currentList.firstOrNull { it.id == modelId && it.status == ModelStatus.INSTALLED } ?: return false

        val updated = currentList.map {
            it.copy(isActive = (it.id == target.id))
        }
        _models.value = updated
        _activeModel.value = updated.firstOrNull { it.isActive }

        prefs.edit().putString(KEY_ACTIVE_MODEL_FILENAME, target.filename).apply()
        Log.i(TAG, "Active local GGUF model set to: ${target.filename} (${target.displayName})")
        return true
    }

    /**
     * Safely removes a model file and its metadata.
     * Prevents deletion if generation is currently in progress.
     */
    @Synchronized
    fun removeModel(modelId: String): Boolean {
        if (isGenerating.get()) {
            Log.w(TAG, "Cannot remove model $modelId: active generation process is running.")
            return false
        }

        val currentList = _models.value
        val target = currentList.firstOrNull { it.id == modelId } ?: return false

        val wasActive = target.isActive

        // Delete physical file if inside internal filesDir
        val modelDir = getModelDirectory()
        if (target.file.parentFile?.absolutePath == modelDir.absolutePath) {
            if (target.file.exists()) {
                val deleted = target.file.delete()
                Log.i(TAG, "Deleted model file: ${target.file.absolutePath} (success=$deleted)")
            }
        } else {
            Log.i(TAG, "Model file is outside app directory (${target.file.absolutePath}), removing from registry.")
        }

        val remaining = currentList.filter { it.id != modelId }

        // If the active model was deleted, gracefully select another installed model
        val newActive = if (wasActive) {
            remaining.firstOrNull { it.status == ModelStatus.INSTALLED }
        } else {
            remaining.firstOrNull { it.isActive }
        }

        val finalized = remaining.map {
            if (newActive != null && it.id == newActive.id) it.copy(isActive = true) else it.copy(isActive = false)
        }

        _models.value = finalized
        _activeModel.value = finalized.firstOrNull { it.isActive }

        if (_activeModel.value != null) {
            prefs.edit().putString(KEY_ACTIVE_MODEL_FILENAME, _activeModel.value!!.filename).apply()
        } else {
            prefs.edit().remove(KEY_ACTIVE_MODEL_FILENAME).apply()
        }

        savePersistedMetadata(finalized)
        return true
    }

    /**
     * Imports a GGUF model from a SAF content URI.
     * Copies safely into a .part temporary file, validates GGUF header/metadata,
     * atomically renames to the destination file, registers the model, and activates it.
     */
    suspend fun importModelFromUri(uri: Uri, beforeReplace: suspend (File) -> Unit = {}): Result<ModelItem> = withContext(Dispatchers.IO) {
        _importError.value = null
        try {
            val resolver = context.contentResolver

            // 1. Resolve source filename and size from ContentResolver
            var sourceName = "model.gguf"
            var sourceSize = 0L

            resolver.query(uri, null, null, null, null)?.use { cursor ->
                val nameIdx = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                val sizeIdx = cursor.getColumnIndex(OpenableColumns.SIZE)
                if (cursor.moveToFirst()) {
                    if (nameIdx != -1) {
                        val n = cursor.getString(nameIdx)
                        if (!n.isNullOrBlank()) sourceName = n
                    }
                    if (sizeIdx != -1) {
                        sourceSize = cursor.getLong(sizeIdx)
                    }
                }
            }

            // Ensure .gguf extension
            val finalFilename = if (sourceName.endsWith(".gguf", ignoreCase = true)) {
                sourceName
            } else {
                "$sourceName.gguf"
            }

            Log.i(TAG, "Starting import of GGUF model from URI: $uri -> $finalFilename (declared size: $sourceSize bytes)")

            val modelDir = getModelDirectory()
            val tempFile = File(modelDir, "import_${System.currentTimeMillis()}_$finalFilename.part")
            val destFile = File(modelDir, finalFilename)

            // 2. Stream copy with live progress
            _importProgress.value = ImportProgress(finalFilename, 0L, sourceSize, 0)

            resolver.openInputStream(uri)?.use { input ->
                FileOutputStream(tempFile).use { output ->
                    val buffer = ByteArray(64 * 1024) // 64 KB buffer for fast mobile I/O
                    var bytesCopied = 0L
                    var lastProgressUpdate = System.currentTimeMillis()
                    var read: Int

                    while (input.read(buffer).also { read = it } != -1) {
                        output.write(buffer, 0, read)
                        bytesCopied += read

                        val now = System.currentTimeMillis()
                        if (now - lastProgressUpdate > 250 || bytesCopied == sourceSize) {
                            val percent = if (sourceSize > 0) ((bytesCopied * 100) / sourceSize).toInt().coerceIn(0, 100) else 0
                            _importProgress.value = ImportProgress(finalFilename, bytesCopied, sourceSize, percent)
                            lastProgressUpdate = now
                        }
                    }
                    output.flush()
                }
            } ?: run {
                tempFile.delete()
                _importProgress.value = null
                _importError.value = "Unable to open input stream from selected URI"
                return@withContext Result.failure(IllegalStateException("Unable to open input stream from selected URI"))
            }

            _importProgress.value = ImportProgress(finalFilename, tempFile.length(), tempFile.length(), 100)

            // 3. Binary GGUF validation
            val meta = GGUFMetadataParser.parse(tempFile)
            if (!meta.isValid) {
                Log.e(TAG, "Import validation failed: ${meta.errorMessage}")
                tempFile.delete()
                _importProgress.value = null
                val userError = "Invalid or unsupported GGUF model"
                _importError.value = userError
                return@withContext Result.failure(IllegalArgumentException(userError))
            }

            // 4. Atomically move/rename temp file to destination file (releasing it first if it is the loaded model)
            if (destFile.exists()) {
                beforeReplace(destFile)
                destFile.delete()
            }
            val renamed = tempFile.renameTo(destFile)
            if (!renamed) {
                // Fallback copy if rename fails
                tempFile.copyTo(destFile, overwrite = true)
                tempFile.delete()
            }

            Log.i(TAG, "Model successfully imported and verified: ${destFile.absolutePath} (${destFile.length()} bytes)")
            _importProgress.value = null

            // 5. Refresh models and set newly imported model active
            refreshModels()
            setActiveModel(destFile.name)

            val importedItem = _models.value.firstOrNull { it.id == destFile.name }
                ?: ModelItem(
                    id = destFile.name,
                    filename = destFile.name,
                    displayName = meta.modelName.ifBlank { GGUFMetadataParser.deriveDisplayName(destFile.name, meta.architecture, meta.parameterCount) },
                    architecture = meta.architecture,
                    parameterCount = meta.parameterCount,
                    quantization = meta.quantization,
                    contextLength = meta.contextLength,
                    sizeBytes = destFile.length(),
                    importDate = System.currentTimeMillis(),
                    file = destFile,
                    status = ModelStatus.INSTALLED,
                    isActive = true
                )

            Result.success(importedItem)
        } catch (e: Exception) {
            Log.e(TAG, "Exception during model import: ${e.message}", e)
            _importProgress.value = null
            _importError.value = "Model import failed: ${e.message}"
            Result.failure(e)
        }
    }

    /**
     * Gets the currently active model file for inference, verifying existence and readability.
     */
    fun getActiveModelFile(): File? {
        val active = _activeModel.value ?: run {
            refreshModels()
            _activeModel.value
        }
        val file = active?.file ?: return null
        return if (file.exists() && file.canRead() && file.length() > 1024) file else null
    }

    fun listInstalledModels(): List<ModelItem> = _models.value

    fun getActiveModel(): ModelItem? = _activeModel.value

    fun updateModelStatus(modelId: String, status: ModelStatus, errorMsg: String? = null) {
        _models.value = _models.value.map {
            if (it.id == modelId) it.copy(status = status, errorMessage = errorMsg) else it
        }
        if (_activeModel.value?.id == modelId) {
            _activeModel.value = _activeModel.value?.copy(status = status, errorMessage = errorMsg)
        }
    }

    fun clearImportError() {
        _importError.value = null
    }

    private fun loadPersistedMetadata(): Map<String, GGUFMetadata> {
        val map = mutableMapOf<String, GGUFMetadata>()
        if (!metadataFile.exists()) return map
        try {
            val jsonStr = metadataFile.readText(Charsets.UTF_8)
            val jsonArray = JSONArray(jsonStr)
            for (i in 0 until jsonArray.length()) {
                val obj = jsonArray.getJSONObject(i)
                val fn = obj.optString("filename")
                if (fn.isNotBlank()) {
                    map[fn] = GGUFMetadata(
                        filename = fn,
                        architecture = obj.optString("architecture", "qwen2"),
                        modelName = obj.optString("modelName", ""),
                        parameterCount = obj.optString("parameterCount", ""),
                        quantization = obj.optString("quantization", ""),
                        contextLength = obj.optInt("contextLength", 2048),
                        sizeBytes = obj.optLong("sizeBytes", 0L),
                        isValid = obj.optBoolean("isValid", true),
                        errorMessage = if (obj.has("errorMessage") && !obj.isNull("errorMessage")) obj.getString("errorMessage").takeIf { it.isNotBlank() } else null
                    )
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed reading model metadata JSON: ${e.message}")
        }
        return map
    }

    private fun savePersistedMetadata(items: List<ModelItem>) {
        try {
            val jsonArray = JSONArray()
            for (item in items) {
                val obj = JSONObject()
                obj.put("filename", item.filename)
                obj.put("architecture", item.architecture)
                obj.put("modelName", item.displayName)
                obj.put("parameterCount", item.parameterCount)
                obj.put("quantization", item.quantization)
                obj.put("contextLength", item.contextLength)
                obj.put("sizeBytes", item.sizeBytes)
                obj.put("isValid", item.status != ModelStatus.ERROR)
                // Bug #8 fix: persist errorMessage so it survives across app restarts
                if (!item.errorMessage.isNullOrBlank()) {
                    obj.put("errorMessage", item.errorMessage)
                }
                jsonArray.put(obj)
            }
            metadataFile.writeText(jsonArray.toString(2), Charsets.UTF_8)
        } catch (e: Exception) {
            Log.w(TAG, "Failed saving model metadata JSON: ${e.message}")
        }
    }
}
