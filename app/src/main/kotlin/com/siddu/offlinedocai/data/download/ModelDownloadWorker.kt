package com.siddu.offlinedocai.data.download

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.pm.ServiceInfo
import android.os.Build
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.work.CoroutineWorker
import androidx.work.ForegroundInfo
import androidx.work.WorkerParameters
import java.io.IOException

/**
 * Runs one model download as a foreground (dataSync) job so it keeps going when the app is closed.
 * Network errors return [Result.retry]; the partial file is kept and the next attempt resumes it.
 */
class ModelDownloadWorker(appContext: Context, params: WorkerParameters) : CoroutineWorker(appContext, params) {

    companion object {
        const val KEY_MODEL_ID = "model_id"
        private const val CHANNEL_ID = "model_downloads"
        private const val TAG = "ModelDownloadWorker"
    }

    private val manager = ModelDownloadManager.getInstance(appContext)
    private val modelId: String = inputData.getString(KEY_MODEL_ID).orEmpty()
    private val displayName: String = manager.specFor(modelId)?.displayName ?: "model"
    private val notificationId: Int = 4100 + (modelId.hashCode() and 0x3FF)

    override suspend fun doWork(): Result {
        if (modelId.isEmpty()) return Result.failure()
        runCatching { setForeground(foregroundInfo(-1)) }
            .onFailure { Log.w(TAG, "Could not start foreground notification: ${it.message}") }
        return try {
            val ok = manager.runDownload(modelId, runAttemptCount) { percent ->
                runCatching { setForegroundAsync(foregroundInfo(percent)) }
            }
            if (ok) Result.success() else Result.failure()
        } catch (e: IOException) {
            Result.retry()
        }
    }

    override suspend fun getForegroundInfo(): ForegroundInfo = foregroundInfo(-1)

    private fun foregroundInfo(percent: Int): ForegroundInfo {
        val nm = applicationContext.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= 26 && nm.getNotificationChannel(CHANNEL_ID) == null) {
            nm.createNotificationChannel(NotificationChannel(CHANNEL_ID, "Model downloads", NotificationManager.IMPORTANCE_LOW))
        }
        val notification = NotificationCompat.Builder(applicationContext, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle("Downloading $displayName")
            .setContentText(if (percent >= 0) "$percent%" else "Starting…")
            .setProgress(100, percent.coerceAtLeast(0), percent < 0)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .build()
        return if (Build.VERSION.SDK_INT >= 29) {
            ForegroundInfo(notificationId, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            ForegroundInfo(notificationId, notification)
        }
    }
}
