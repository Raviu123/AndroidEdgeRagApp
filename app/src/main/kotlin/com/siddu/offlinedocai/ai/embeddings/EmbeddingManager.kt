package com.siddu.offlinedocai.ai.embeddings

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File

/** What the Settings screen shows for each embedding model. */
sealed class EmbeddingModelStatus {
    object NotInstalled : EmbeddingModelStatus()
    object Installed : EmbeddingModelStatus()
    object Loading : EmbeddingModelStatus()
    object Ready : EmbeddingModelStatus()
    data class Unsupported(val reason: String) : EmbeddingModelStatus()
}

/**
 * Owns the three selectable embedding models: which are installed, which one is active, loading and
 * the per-device health check. It is itself the [EmbeddingModel] used by indexing and search, always
 * delegating to the active model, so a switch takes effect immediately everywhere.
 *
 * Switching models changes [modelId]; documents indexed with another model then have a stale index
 * signature and are re-indexed by DocumentRepository. Vectors of different models are never mixed.
 */
class EmbeddingManager private constructor(private val context: Context) : EmbeddingModel {

    private val TAG = "EmbeddingManager"

    companion object {
        private const val PREFS = "embedding_models"
        private const val KEY_ACTIVE = "active_embedding_model"

        @Volatile private var instance: EmbeddingManager? = null

        fun getInstance(context: Context): EmbeddingManager =
            instance ?: synchronized(this) {
                instance ?: EmbeddingManager(context.applicationContext).also { instance = it }
            }

        /**
         * Semantic sanity probe: on a working model, two paraphrases are much closer than an unrelated
         * sentence. A broken backend (wrong kernels, bad file) fails this even if it returns vectors.
         */
        private val PROBE = listOf(
            "The cat is sitting on the mat.",
            "A kitten rests on the rug.",
            "Quarterly revenue increased by ten percent."
        )
    }

    private val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
    private val switchMutex = Mutex()

    private val _activeId = MutableStateFlow(AvailableEmbeddingModels.byId(prefs.getString(KEY_ACTIVE, null))?.id ?: AvailableEmbeddingModels.default.id)
    val activeId: StateFlow<String> = _activeId.asStateFlow()

    private val _statuses = MutableStateFlow<Map<String, EmbeddingModelStatus>>(emptyMap())
    val statuses: StateFlow<Map<String, EmbeddingModelStatus>> = _statuses.asStateFlow()

    @Volatile private var engine: EmbeddingModel? = null
    @Volatile private var engineId: String? = null

    /** Shared WordPiece vocabulary used to size chunks (identical for all three catalog models). */
    val chunkTokenizer: WordPieceTokenizer by lazy {
        context.assets.open(AvailableEmbeddingModels.VOCAB_ASSET).use { WordPieceTokenizer(it) }
    }

    val activeSpec: EmbeddingModelSpec get() = AvailableEmbeddingModels.byId(_activeId.value) ?: AvailableEmbeddingModels.default

    override val modelId: String get() = activeSpec.id

    init {
        refresh()
    }

    fun downloadDirectory(): File = File(context.filesDir, "models/embeddings").apply { mkdirs() }

    fun fileFor(spec: EmbeddingModelSpec): File = File(downloadDirectory(), spec.fileName)

    fun isInstalled(spec: EmbeddingModelSpec): Boolean =
        spec.isBundled || fileFor(spec).let { it.exists() && it.length() == spec.sizeBytes }

    /** Re-scans installed files (after a download or delete). Keeps known Ready/Unsupported states. */
    fun refresh() {
        val current = _statuses.value
        _statuses.value = AvailableEmbeddingModels.list.associate { spec ->
            val previous = current[spec.id]
            spec.id to when {
                !isInstalled(spec) -> EmbeddingModelStatus.NotInstalled
                previous is EmbeddingModelStatus.Ready || previous is EmbeddingModelStatus.Unsupported || previous is EmbeddingModelStatus.Loading -> previous
                else -> EmbeddingModelStatus.Installed
            }
        }
        if (!isInstalled(activeSpec)) {
            // The active model's file disappeared: fall back to the built-in model.
            setActive(AvailableEmbeddingModels.default.id)
        }
    }

    private fun setStatus(id: String, status: EmbeddingModelStatus) {
        _statuses.value = _statuses.value + (id to status)
    }

    private fun setActive(id: String) {
        prefs.edit().putString(KEY_ACTIVE, id).apply()
        _activeId.value = id
    }

    private fun createEngine(spec: EmbeddingModelSpec): EmbeddingModel = when {
        spec.isBundled -> ONNXEmbeddingEngine.getInstance(context)
        spec.backend == EmbeddingBackend.ONNX -> ONNXEmbeddingEngine.forFile(context, spec, fileFor(spec))
        else -> LlamaEmbeddingEngine(context, spec, fileFor(spec))
    }

    private fun closeEngine(e: EmbeddingModel?) {
        when (e) {
            is LlamaEmbeddingEngine -> e.close()
            is ONNXEmbeddingEngine -> if (!e.spec.isBundled) e.close()
        }
    }

    /** Loads [spec] and checks that it produces sensible vectors on this device. Returns the engine or a failure reason. */
    private suspend fun loadAndCheck(spec: EmbeddingModelSpec): Result<EmbeddingModel> = withContext(Dispatchers.Default) {
        if (!isInstalled(spec)) return@withContext Result.failure(IllegalStateException("Model is not downloaded."))
        setStatus(spec.id, EmbeddingModelStatus.Loading)
        val candidate = createEngine(spec)
        val result = runCatching {
            val vectors = candidate.embedDocuments(PROBE)
            check(vectors.all { it.size == spec.dimension }) { "Unexpected embedding size ${vectors.first().size}" }
            val similar = dot(vectors[0], vectors[1])
            val unrelated = dot(vectors[0], vectors[2])
            check(similar > unrelated + 0.1f) {
                "The model loaded but produced unreliable results on this device (similarity check failed)."
            }
            candidate
        }
        result.onSuccess { setStatus(spec.id, EmbeddingModelStatus.Ready) }
            .onFailure { e ->
                closeEngine(candidate)
                val reason = e.message ?: "Unknown error"
                Log.w(TAG, "Embedding model ${spec.id} is not usable: $reason")
                setStatus(spec.id, EmbeddingModelStatus.Unsupported(reason))
            }
        result
    }

    /** Makes sure the active model is loaded and healthy. Returns false (status Unsupported) otherwise. */
    suspend fun ensureActiveReady(): Boolean = switchMutex.withLock {
        val spec = activeSpec
        if (engine != null && engineId == spec.id) return@withLock true
        val loaded = loadAndCheck(spec).getOrNull() ?: return@withLock false
        closeEngine(engine)
        engine = loaded
        engineId = spec.id
        true
    }

    /**
     * Switches the active model after it passes the health check on this device. On failure the
     * previous model stays active and the reason is returned.
     */
    suspend fun activate(id: String): Result<Unit> = switchMutex.withLock {
        val spec = AvailableEmbeddingModels.byId(id) ?: return@withLock Result.failure(IllegalArgumentException("Unknown model $id"))
        if (engineId == spec.id && engine != null) {
            setActive(spec.id)
            return@withLock Result.success(Unit)
        }
        val checked: Result<EmbeddingModel> = loadAndCheck(spec)
        val loaded = checked.getOrNull() ?: return@withLock Result.failure(checked.exceptionOrNull() ?: IllegalStateException("Model failed to load"))
        closeEngine(engine)
        engine = loaded
        engineId = spec.id
        setActive(spec.id)
        Log.i(TAG, "Active embedding model: ${spec.id}")
        Result.success(Unit)
    }

    /** Deletes a downloaded model. The active and the built-in model cannot be deleted. */
    suspend fun delete(id: String): Result<Unit> = switchMutex.withLock {
        val spec = AvailableEmbeddingModels.byId(id) ?: return@withLock Result.failure(IllegalArgumentException("Unknown model"))
        if (spec.isBundled) return@withLock Result.failure(IllegalStateException("The built-in model cannot be deleted."))
        if (spec.id == activeSpec.id) return@withLock Result.failure(IllegalStateException("Switch to another search model before deleting this one."))
        withContext(Dispatchers.IO) { fileFor(spec).delete() }
        _statuses.value = _statuses.value + (spec.id to EmbeddingModelStatus.NotInstalled)
        Result.success(Unit)
    }

    /** The loaded engine of the active model (its [EmbeddingModel.modelId] is the id to store with its vectors). */
    suspend fun currentEngine(): EmbeddingModel {
        engine?.takeIf { engineId == activeSpec.id }?.let { return it }
        if (!ensureActiveReady()) {
            val status = _statuses.value[activeSpec.id]
            val reason = (status as? EmbeddingModelStatus.Unsupported)?.reason ?: "unknown error"
            throw IllegalStateException("The on-device embedding model failed to load ($reason). Choose another search model in Settings.")
        }
        return engine!!
    }

    override suspend fun getEmbedding(text: String): FloatArray = currentEngine().getEmbedding(text)

    override suspend fun embedQuery(text: String): FloatArray = currentEngine().embedQuery(text)

    override suspend fun embedDocuments(texts: List<String>): List<FloatArray> = currentEngine().embedDocuments(texts)

    private fun dot(a: FloatArray, b: FloatArray): Float {
        var s = 0f
        for (i in a.indices) s += a[i] * b[i]
        return s
    }
}
