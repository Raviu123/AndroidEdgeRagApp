package com.offlinedocai.ai.llm

import android.content.Context
import android.os.Build
import android.util.Log
import java.io.File

/** Receives the text generated so far (UTF-8) while a completion is running. */
fun interface NativeTextListener {
    fun onText(utf8: ByteArray)
}

/**
 * Process-wide entry point to the llama.cpp JNI library, shared by the language model and the
 * GGUF embedding model.
 *
 * On arm64 every Android CPU variant of the ggml CPU backend is shipped; the native side loads the
 * best one the CPU supports (a variant whose features the CPU lacks refuses to load). **Compatibility
 * mode** forces the baseline (ARMv8.0) variant. It is enabled automatically after a native crash
 * during model load, calibration or generation (crash sentinel), or manually from Settings. On
 * MediaTek Helio MT676x/677x/678x chips (crash history with another library, fix dc9cbf7) the
 * sentinel stays armed for every run. Changes take effect after the app restarts, because a CPU
 * backend cannot be swapped once loaded.
 */
object LlamaNative {
    private const val TAG = "LlamaNative"
    private const val PREFS = "llama_native"
    private const val KEY_COMPAT_MODE = "compatibility_mode"
    private const val KEY_COMPAT_REASON = "compatibility_reason"
    private const val SENTINEL_NAME = "llama_native_crash_sentinel"

    @Volatile private var initialized = false
    @Volatile var isAvailable = false
        private set
    @Volatile var loadError: String? = null
        private set
    /** CPU variant actually in use ("android_armv8.2_2", "static", ...); empty when unavailable. */
    @Volatile var cpuVariant: String = ""
        private set
    @Volatile var compatibilityModeActive = false
        private set
    /** Why compatibility mode was switched on automatically (shown once in Settings). */
    @Volatile var compatibilityReason: String? = null
        private set
    /** True on chips with a history of native crashes: every model run keeps the crash sentinel armed. */
    @Volatile var alwaysGuarded = false
        private set

    private var sentinelFile: File? = null

    @Synchronized
    fun ensureInitialized(context: Context): Boolean {
        if (initialized) return isAvailable
        initialized = true
        val app = context.applicationContext
        val prefs = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        sentinelFile = File(app.filesDir, SENTINEL_NAME)

        // A sentinel left behind means the process died inside native model code last time.
        if (sentinelFile?.exists() == true) {
            sentinelFile?.delete()
            if (!prefs.getBoolean(KEY_COMPAT_MODE, false)) {
                prefs.edit()
                    .putBoolean(KEY_COMPAT_MODE, true)
                    .putString(KEY_COMPAT_REASON, "The app closed unexpectedly while loading or running a model, so compatibility mode was turned on.")
                    .apply()
            }
        }
        // Helio chips are no longer forced onto the slow baseline kernels: the crash reported for them came
        // from a different library (llama.rn), and ggml only loads a CPU variant whose features the chip
        // reports. As a safety net every model run on these chips keeps the crash sentinel armed, so a
        // native crash switches compatibility mode on for the next launch.
        alwaysGuarded = isKnownUnstableChip()
        compatibilityModeActive = prefs.getBoolean(KEY_COMPAT_MODE, false)
        compatibilityReason = prefs.getString(KEY_COMPAT_REASON, null)

        return try {
            System.loadLibrary("llama_jni")
            cpuVariant = nativeInitBackends(app.applicationInfo.nativeLibraryDir, compatibilityModeActive)
            isAvailable = cpuVariant.isNotEmpty()
            if (!isAvailable) loadError = "No compatible CPU backend for the local model was found on this device."
            Log.i(TAG, "llama.cpp ready: cpu=$cpuVariant compatibilityMode=$compatibilityModeActive")
            isAvailable
        } catch (t: Throwable) {
            loadError = "Native inference library is unavailable (${t.message})."
            Log.e(TAG, "Failed to load llama_jni", t)
            false
        }
    }

    fun isCompatibilityModePreferred(context: Context): Boolean =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getBoolean(KEY_COMPAT_MODE, false)

    /** Persists the user's choice; applied on next app start. */
    fun setCompatibilityModePreferred(context: Context, enabled: Boolean) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
            .putBoolean(KEY_COMPAT_MODE, enabled)
            .remove(KEY_COMPAT_REASON)
            .apply()
    }

    /** Runs [block] with the crash sentinel armed, so a native abort switches on compatibility mode next launch. */
    inline fun <T> guarded(block: () -> T): T {
        armSentinel()
        try {
            return block()
        } finally {
            disarmSentinel()
        }
    }

    @PublishedApi internal fun armSentinel() {
        runCatching { sentinelFile?.writeText(System.currentTimeMillis().toString()) }
    }

    @PublishedApi internal fun disarmSentinel() {
        runCatching { sentinelFile?.delete() }
    }

    private fun isKnownUnstableChip(): Boolean {
        val ids = listOf(Build.HARDWARE, Build.BOARD, if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else "")
            .joinToString(" ") { it.orEmpty().lowercase() }
        val cpuInfo = runCatching { File("/proc/cpuinfo").readText().lowercase() }.getOrDefault("")
        val text = "$ids $cpuInfo"
        return listOf("mt676", "mt677", "mt678", "helio").any { it in text }
    }

    // --- JNI ---------------------------------------------------------------------------------------

    @JvmStatic external fun nativeInitBackends(libDir: String?, safeMode: Boolean): String
    @JvmStatic external fun nativeInit(modelPath: ByteArray, contextSize: Int, threadCount: Int): Long
    @JvmStatic external fun nativeFree(handle: Long)
    @JvmStatic external fun nativeGetContextSize(handle: Long): Int
    @JvmStatic external fun nativeCountTokens(handle: Long, text: ByteArray, isPrompt: Boolean): Int
    @JvmStatic external fun nativeGetMetadata(handle: Long, key: String): ByteArray?
    @JvmStatic external fun nativeApplyChatTemplate(handle: Long, roles: Array<String>, contents: Array<ByteArray>, addAssistant: Boolean): ByteArray?
    @JvmStatic external fun nativeResetCancel(handle: Long)
    @JvmStatic external fun nativeCancel(handle: Long)
    @JvmStatic external fun nativeInference(
        handle: Long, prompt: ByteArray, maxTokens: Int, temperature: Float, topP: Float,
        repeatPenalty: Float, seed: Int, listener: NativeTextListener?
    ): ByteArray
    @JvmStatic external fun nativeSetThreads(handle: Long, threadsDecode: Int, threadsBatch: Int)
    /** [promptTokens, reusedTokens, prefillMs, firstTokenMs, decodeMs, generatedTokens, threadsDecode, threadsBatch] of the last inference. */
    @JvmStatic external fun nativeLastStats(handle: Long): LongArray?
    /** [prefillMs, decodeMs] of a synthetic run, or null if it failed or was cancelled. */
    @JvmStatic external fun nativeBenchmark(handle: Long, promptTokens: Int, genTokens: Int, threadsDecode: Int, threadsBatch: Int): LongArray?

    @JvmStatic external fun nativeEmbedInit(modelPath: ByteArray, contextSize: Int, threadCount: Int): Long
    @JvmStatic external fun nativeEmbedFree(handle: Long)
    @JvmStatic external fun nativeEmbedDim(handle: Long): Int
    @JvmStatic external fun nativeEmbed(handle: Long, text: ByteArray): FloatArray?
}
