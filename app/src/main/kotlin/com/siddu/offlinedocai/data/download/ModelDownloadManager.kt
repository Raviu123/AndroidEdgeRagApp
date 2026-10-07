package com.siddu.offlinedocai.data.download

import android.content.Context
import android.os.StatFs
import android.util.Log
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkInfo
import androidx.work.WorkManager
import androidx.work.workDataOf
import com.siddu.offlinedocai.ai.embeddings.AvailableEmbeddingModels
import com.siddu.offlinedocai.ai.embeddings.EmbeddingManager
import com.siddu.offlinedocai.ai.llm.AvailableLlmModels
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit
import kotlin.coroutines.coroutineContext

/** Per-model download state shown on that model's card. */
sealed class DownloadStatus {
    data class Queued(val note: String? = null) : DownloadStatus()
    data class Downloading(val bytes: Long, val total: Long) : DownloadStatus() {
        val percent: Int get() = if (total > 0) ((bytes * 100) / total).toInt().coerceIn(0, 100) else 0
    }
    object Verifying : DownloadStatus()
    object Completed : DownloadStatus()
    data class Failed(val message: String) : DownloadStatus()
}

sealed class DownloadEvent {
    data class Completed(val id: String, val isEmbedding: Boolean) : DownloadEvent()
    data class Failed(val id: String, val message: String) : DownloadEvent()
}

enum class EnqueueResult { STARTED, ALREADY_ACTIVE, ALREADY_INSTALLED, NOT_ENOUGH_SPACE, UNKNOWN_MODEL }

/**
 * Downloads language and embedding models from the catalogs.
 *
 * - Every model has its **own** status entry, so a card never shows another model's progress.
 * - A model that is already queued or downloading cannot be started again.
 * - Downloads run in a WorkManager foreground worker ([ModelDownloadWorker]), so they survive leaving
 *   the app, wait for a network connection and resume a partial file instead of starting over.
 * - Only one file is transferred at a time; others wait in the queue.
 */
class ModelDownloadManager private constructor(private val context: Context) {

    private val TAG = "ModelDownloadManager"

    companion object {
        private const val WORK_PREFIX = "model-download-"
        private const val SPACE_MARGIN_BYTES = 50L * 1024 * 1024
        private const val MAX_NETWORK_ATTEMPTS = 5

        @Volatile private var instance: ModelDownloadManager? = null

        fun getInstance(context: Context): ModelDownloadManager =
            instance ?: synchronized(this) {
                instance ?: ModelDownloadManager(context.applicationContext).also { instance = it }
            }

        fun workName(id: String) = WORK_PREFIX + id
    }

    private val _statuses = MutableStateFlow<Map<String, DownloadStatus>>(emptyMap())
    val statuses: StateFlow<Map<String, DownloadStatus>> = _statuses.asStateFlow()

    private val _events = MutableSharedFlow<DownloadEvent>(extraBufferCapacity = 8)
    val events: SharedFlow<DownloadEvent> = _events.asSharedFlow()

    private val transferMutex = Mutex()
    private val userCancelled = ConcurrentHashMap.newKeySet<String>()
    private val downloader = ModelFileDownloader()
    private val workManager: WorkManager get() = WorkManager.getInstance(context)

    /** Catalog lookup for both kinds of model (bundled embedding models are not downloadable). */
    fun specFor(id: String): DownloadSpec? {
        AvailableLlmModels.list.firstOrNull { it.id == id }?.let { m ->
            return DownloadSpec(
                id = m.id, displayName = m.name, url = m.downloadUrl,
                destination = File(File(context.filesDir, "models/llm"), m.fileName),
                sizeBytes = m.sizeBytes, sha256 = m.sha256, format = ModelFileFormat.GGUF
            )
        }
        AvailableEmbeddingModels.list.firstOrNull { it.id == id && !it.isBundled }?.let { e ->
            return DownloadSpec(
                id = e.id, displayName = e.name, url = e.downloadUrl,
                destination = EmbeddingManager.getInstance(context).fileFor(e),
                sizeBytes = e.sizeBytes, sha256 = e.sha256,
                format = if (e.fileName.endsWith(".gguf")) ModelFileFormat.GGUF else ModelFileFormat.ONNX
            )
        }
        return null
    }

    fun isEmbedding(id: String) = AvailableEmbeddingModels.list.any { it.id == id }

    fun isActive(id: String): Boolean = when (_statuses.value[id]) {
        is DownloadStatus.Queued, is DownloadStatus.Downloading, DownloadStatus.Verifying -> true
        else -> false
    }

    fun enqueue(id: String): EnqueueResult {
        val spec = specFor(id) ?: return EnqueueResult.UNKNOWN_MODEL
        synchronized(this) {
            if (isActive(id)) return EnqueueResult.ALREADY_ACTIVE
            if (spec.destination.exists() && spec.destination.length() == spec.sizeBytes) return EnqueueResult.ALREADY_INSTALLED
            val alreadyOnDisk = if (spec.partFile.exists()) spec.partFile.length() else 0L
            val needed = spec.sizeBytes - alreadyOnDisk + SPACE_MARGIN_BYTES
            spec.destination.parentFile?.mkdirs()
            val free = runCatching { StatFs(spec.destination.parentFile!!.absolutePath).availableBytes }.getOrDefault(Long.MAX_VALUE)
            if (free < needed) {
                setStatus(id, DownloadStatus.Failed("Not enough free storage: ${needed / 1_000_000} MB needed, ${free / 1_000_000} MB available."))
                return EnqueueResult.NOT_ENOUGH_SPACE
            }
            userCancelled.remove(id)
            setStatus(id, DownloadStatus.Queued())
        }
        val request = OneTimeWorkRequestBuilder<ModelDownloadWorker>()
            .setInputData(workDataOf(ModelDownloadWorker.KEY_MODEL_ID to id))
            .setConstraints(Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build())
            .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 15, TimeUnit.SECONDS)
            .addTag(WORK_PREFIX)
            .build()
        workManager.enqueueUniqueWork(workName(id), ExistingWorkPolicy.KEEP, request)
        return EnqueueResult.STARTED
    }

    /** Stops a queued or running download and deletes its partial file. */
    fun cancel(id: String) {
        userCancelled.add(id)
        workManager.cancelUniqueWork(workName(id))
        specFor(id)?.partFile?.delete()
        _statuses.value = _statuses.value - id
    }

    /** Clears a finished/failed status from its card. */
    fun dismiss(id: String) {
        if (!isActive(id)) _statuses.value = _statuses.value - id
    }

    private fun setStatus(id: String, status: DownloadStatus) {
        _statuses.value = _statuses.value + (id to status)
    }

    /**
     * Runs one download (called by [ModelDownloadWorker]). Returns true on success, false on a permanent
     * failure, and throws [IOException] when the worker should retry later (network problems).
     */
    internal suspend fun runDownload(id: String, attempt: Int, onProgress: (Int) -> Unit): Boolean {
        val spec = specFor(id) ?: run {
            setStatus(id, DownloadStatus.Failed("Unknown model."))
            return false
        }
        if (!isActive(id)) setStatus(id, DownloadStatus.Queued())
        return transferMutex.withLock {
            withContext(Dispatchers.IO) {
                val job = coroutineContext
                try {
                    var lastPercent = -1
                    downloader.download(
                        spec,
                        onProgress = { bytes, total ->
                            setStatus(id, DownloadStatus.Downloading(bytes, total))
                            val pct = ((bytes * 100) / total).toInt()
                            if (pct != lastPercent) {
                                lastPercent = pct
                                onProgress(pct)
                            }
                        },
                        onVerifying = { setStatus(id, DownloadStatus.Verifying) },
                        checkCancelled = { job.ensureActive() }
                    )
                    setStatus(id, DownloadStatus.Completed)
                    _events.tryEmit(DownloadEvent.Completed(id, isEmbedding(id)))
                    Log.i(TAG, "Installed ${spec.destination.name}")
                    true
                } catch (e: CancellationException) {
                    if (userCancelled.remove(id)) {
                        spec.partFile.delete()
                        _statuses.value = _statuses.value - id
                    } else {
                        setStatus(id, DownloadStatus.Queued("Paused, will resume"))
                    }
                    throw e
                } catch (e: ModelVerificationException) {
                    fail(id, e.message ?: "Verification failed.")
                    false
                } catch (e: IOException) {
                    if (attempt + 1 >= MAX_NETWORK_ATTEMPTS) {
                        fail(id, "Download failed: ${e.message ?: "network error"}. Tap Retry to continue where it stopped.")
                        false
                    } else {
                        setStatus(id, DownloadStatus.Queued("Connection problem, retrying…"))
                        throw e
                    }
                } catch (e: Exception) {
                    fail(id, "Download failed: ${e.message ?: e.javaClass.simpleName}")
                    false
                }
            }
        }
    }

    private fun fail(id: String, message: String) {
        setStatus(id, DownloadStatus.Failed(message))
        _events.tryEmit(DownloadEvent.Failed(id, message))
    }

    /**
     * Deletes partial files that no pending download will resume (e.g. left by a cancelled download
     * or an interrupted import) and restores the status of downloads WorkManager is still running.
     */
    suspend fun cleanupAndRestore() = withContext(Dispatchers.IO) {
        val dirs = listOf(File(context.filesDir, "models/llm"), File(context.filesDir, "models/embeddings"))
        val catalogIds = AvailableLlmModels.list.map { it.id } + AvailableEmbeddingModels.list.filter { !it.isBundled }.map { it.id }
        val pendingIds = catalogIds.filter { id ->
            runCatching { workManager.getWorkInfosForUniqueWork(workName(id)).get() }.getOrDefault(emptyList())
                .any { it.state == WorkInfo.State.ENQUEUED || it.state == WorkInfo.State.RUNNING || it.state == WorkInfo.State.BLOCKED }
        }.toSet()
        pendingIds.forEach { if (!isActive(it)) setStatus(it, DownloadStatus.Queued("Waiting to resume")) }
        val keep = pendingIds.mapNotNull { specFor(it)?.partFile?.absolutePath }.toSet()
        for (dir in dirs) {
            dir.listFiles()?.filter { it.name.endsWith(".part") && it.absolutePath !in keep }?.forEach {
                Log.i(TAG, "Removing orphaned partial file ${it.name}")
                it.delete()
            }
        }
    }
}
