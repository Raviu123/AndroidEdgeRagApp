package com.siddu.offlinedocai.ai.llm

import android.content.Context
import android.os.Build
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/** How long the user is willing to wait for the model to read the sources (prompt processing). */
enum class SpeedMode(val label: String, val targetPrefillSeconds: Int) {
    FASTEST("Fastest", 20),
    BALANCED("Balanced", 35),
    THOROUGH("Thorough", 75);

    companion object {
        fun fromName(name: String?): SpeedMode = entries.firstOrNull { it.name == name } ?: BALANCED
    }
}

/** Device heat / power state, from Android's own signals. */
enum class ThermalState { NORMAL, WARM, HOT }

/** Threads used for one request. */
data class ThreadPlan(val decode: Int, val batch: Int)

/**
 * Per-device, per-model performance tuning. Nothing is hard-coded for a phone:
 *  1. [CpuTopology] proposes candidate thread counts from the CPU layout.
 *  2. [calibrate] measures them once (short synthetic runs) and keeps the fastest; results are stored
 *     per phone model + CPU kernels + model file + app version.
 *  3. [record] learns real prompt-processing / generation speeds from every answer (moving average).
 *  4. [evidenceTokenCap] sizes the evidence so prompt processing fits the [SpeedMode] target on this
 *     device; [threadPlan] and the cap back off when Android reports heat or battery saver.
 */
class PerformanceTuner private constructor(context: Context) {

    private val TAG = "PerformanceTuner"
    private val app = context.applicationContext
    private val prefs = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
    private val powerManager = app.getSystemService(Context.POWER_SERVICE) as? PowerManager

    val topology: CpuTopology = CpuTopology.device

    data class Profile(
        val decodeThreads: Int,
        val batchThreads: Int,
        /** Prompt tokens/s measured by calibration (short prompt), 0 = unknown. */
        val calibratedPrefillTps: Double = 0.0,
        val calibratedDecodeTps: Double = 0.0,
        /** Moving averages over real answers (long prompts), 0 = no observation yet. */
        val observedPrefillTps: Double = 0.0,
        val observedDecodeTps: Double = 0.0,
        val calibrated: Boolean = false
    )

    @Volatile private var key: String = ""
    @Volatile var profile: Profile = defaultProfile()
        private set

    private val stopCalibration = AtomicBoolean(false)
    @Volatile var isCalibrating = false
        private set

    var speedMode: SpeedMode
        get() = SpeedMode.fromName(prefs.getString(KEY_SPEED_MODE, null))
        set(value) { prefs.edit().putString(KEY_SPEED_MODE, value.name).apply() }

    private fun defaultProfile() = Profile(topology.defaultDecodeThreads, topology.defaultBatchThreads)

    /** Selects (and loads) the stored profile for [modelFile] on this device. */
    @Synchronized
    fun bind(modelFile: File, cpuVariant: String) {
        val newKey = listOf(
            Build.MANUFACTURER, Build.MODEL, cpuVariant, topology.label,
            modelFile.name, modelFile.length(), appVersion(), PROFILE_VERSION
        ).joinToString("|")
        if (newKey == key) return
        key = newKey
        profile = load(newKey)?.let(::sanitize) ?: defaultProfile()
        Log.i(TAG, "Profile for ${modelFile.name}: cpu=${topology.label} (${topology.source}) $profile")
    }

    val needsCalibration: Boolean get() = key.isNotEmpty() && !profile.calibrated

    /** Forgets the measured profile of the bound model so the next warm-up calibrates again. */
    @Synchronized
    fun resetCalibration() {
        if (key.isEmpty()) return
        prefs.edit().remove("profile:$key").apply()
        profile = defaultProfile()
    }

    // ---------------------------------------------------------------------------------------------
    // Heat / power
    // ---------------------------------------------------------------------------------------------

    fun thermalState(): ThermalState {
        val pm = powerManager ?: return ThermalState.NORMAL
        val status = if (Build.VERSION.SDK_INT >= 29) runCatching { pm.currentThermalStatus }.getOrDefault(0) else 0
        if (Build.VERSION.SDK_INT >= 29 && status >= PowerManager.THERMAL_STATUS_SEVERE) return ThermalState.HOT
        // Forecast 10 s ahead; 1.0 means the device would start throttling. NaN when unsupported.
        val headroom = if (Build.VERSION.SDK_INT >= 30) runCatching { pm.getThermalHeadroom(10) }.getOrDefault(Float.NaN) else Float.NaN
        val moderate = Build.VERSION.SDK_INT >= 29 && status >= PowerManager.THERMAL_STATUS_MODERATE
        return if (moderate || pm.isPowerSaveMode || (!headroom.isNaN() && headroom >= 0.95f)) ThermalState.WARM else ThermalState.NORMAL
    }

    /** Threads for the next request, reduced when the phone is hot or in battery saver. */
    fun threadPlan(thermal: ThermalState = thermalState()): ThreadPlan {
        val p = profile
        return when (thermal) {
            ThermalState.NORMAL -> ThreadPlan(p.decodeThreads, p.batchThreads)
            ThermalState.WARM -> ThreadPlan(p.decodeThreads, maxOf(p.decodeThreads, (p.batchThreads * 3 + 3) / 4))
            ThermalState.HOT -> {
                val fast = topology.fastCores.coerceIn(1, topology.maxSafeThreads)
                ThreadPlan(minOf(p.decodeThreads, fast), minOf(p.batchThreads, fast))
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Calibration
    // ---------------------------------------------------------------------------------------------

    /** Asks a running [calibrate] to stop as soon as possible (a user request is waiting). */
    fun interruptCalibration() {
        if (isCalibrating) stopCalibration.set(true)
    }

    /**
     * Measures the candidate thread counts with [bench] (promptTokens, genTokens, decodeThreads,
     * batchThreads) -> [prefillMs, decodeMs] or null. Returns true when a profile was stored. Any
     * failure or interruption keeps the untuned defaults and calibration is retried on the next warm-up.
     */
    fun calibrate(bench: (Int, Int, Int, Int) -> LongArray?): Boolean {
        if (!needsCalibration) return false
        stopCalibration.set(false)
        isCalibrating = true
        val started = SystemClock.elapsedRealtime()
        try {
            val d0 = topology.defaultDecodeThreads
            // Warm-up: the first run pages the model weights in from storage and is not representative.
            bench(WARMUP_TOKENS, 0, d0, topology.defaultBatchThreads) ?: return false

            var bestBatch = topology.defaultBatchThreads
            var bestPrefillTps = 0.0
            for (threads in topology.batchCandidates) {
                if (stopCalibration.get()) return false
                val r = bench(CALIBRATION_PROMPT_TOKENS, 0, d0, threads) ?: return false
                val tps = CALIBRATION_PROMPT_TOKENS * 1000.0 / r[0].coerceAtLeast(1)
                Log.i(TAG, "calibrate prefill threads=$threads -> %.1f tok/s".format(tps))
                if (isClearlyBetter(tps, bestPrefillTps)) { bestPrefillTps = tps; bestBatch = threads }
                if (SystemClock.elapsedRealtime() - started > MAX_CALIBRATION_MS) break
            }

            var bestDecode = d0
            var bestDecodeTps = 0.0
            for (threads in topology.decodeCandidates) {
                if (stopCalibration.get()) return false
                val r = bench(DECODE_PROMPT_TOKENS, CALIBRATION_GEN_TOKENS, threads, bestBatch) ?: return false
                val tps = CALIBRATION_GEN_TOKENS * 1000.0 / r[1].coerceAtLeast(1)
                Log.i(TAG, "calibrate decode threads=$threads -> %.1f tok/s".format(tps))
                if (isClearlyBetter(tps, bestDecodeTps)) { bestDecodeTps = tps; bestDecode = threads }
                if (SystemClock.elapsedRealtime() - started > MAX_CALIBRATION_MS) break
            }

            val tuned = sanitize(
                profile.copy(
                    decodeThreads = bestDecode,
                    batchThreads = bestBatch,
                    calibratedPrefillTps = bestPrefillTps,
                    calibratedDecodeTps = bestDecodeTps,
                    calibrated = true
                )
            )
            synchronized(this) { profile = tuned; save(key, tuned) }
            Log.i(TAG, "Calibrated in ${SystemClock.elapsedRealtime() - started} ms: $tuned")
            return true
        } finally {
            isCalibrating = false
            stopCalibration.set(false)
        }
    }

    /** Fewer threads win ties: a candidate must beat the current best by [MIN_GAIN] to be chosen. */
    private fun isClearlyBetter(candidate: Double, best: Double) = best <= 0.0 || candidate > best * (1 + MIN_GAIN)

    // ---------------------------------------------------------------------------------------------
    // Learning from real answers
    // ---------------------------------------------------------------------------------------------

    /** Updates the speed averages from one real inference (ignored when too short to be meaningful). */
    @Synchronized
    fun record(newPromptTokens: Long, prefillMs: Long, generatedTokens: Long, decodeMs: Long) {
        if (key.isEmpty()) return
        var p = profile
        if (newPromptTokens >= MIN_PREFILL_SAMPLE && prefillMs > 0) {
            p = p.copy(observedPrefillTps = ema(p.observedPrefillTps, newPromptTokens * 1000.0 / prefillMs))
        }
        if (generatedTokens >= MIN_DECODE_SAMPLE && decodeMs > 0) {
            p = p.copy(observedDecodeTps = ema(p.observedDecodeTps, generatedTokens * 1000.0 / decodeMs))
        }
        if (p != profile) {
            profile = p
            save(key, p)
        }
    }

    private fun ema(previous: Double, sample: Double) = if (previous <= 0.0) sample else previous * (1 - EMA_ALPHA) + sample * EMA_ALPHA

    /** Best estimate of prompt-processing speed for long, real prompts (tokens/s), null when unknown. */
    fun prefillTokensPerSecond(): Double? {
        val p = profile
        return when {
            p.observedPrefillTps > 0 -> p.observedPrefillTps
            // Calibration uses a short prompt; real prompts are longer and somewhat slower per token.
            p.calibratedPrefillTps > 0 -> p.calibratedPrefillTps * SHORT_PROMPT_DISCOUNT
            else -> null
        }
    }

    fun decodeTokensPerSecond(): Double? = profile.let { p ->
        when {
            p.observedDecodeTps > 0 -> p.observedDecodeTps
            p.calibratedDecodeTps > 0 -> p.calibratedDecodeTps
            else -> null
        }
    }

    /** Seconds of prompt processing allowed for the next request (speed mode, reduced when hot). */
    fun targetPrefillSeconds(thermal: ThermalState = thermalState()): Double =
        speedMode.targetPrefillSeconds * when (thermal) {
            ThermalState.NORMAL -> 1.0
            ThermalState.WARM -> 0.75
            ThermalState.HOT -> 0.5
        }

    /**
     * Largest evidence size (tokens) whose processing fits the time target, given [otherNewTokens] prompt
     * tokens that must be processed besides the evidence. Null when the speed is not known yet.
     */
    fun evidenceTokenCap(otherNewTokens: Int, thermal: ThermalState = thermalState()): Int? {
        val tps = prefillTokensPerSecond() ?: return null
        return evidenceCapFor(tps, targetPrefillSeconds(thermal), otherNewTokens)
    }

    companion object {
        private const val PREFS = "performance_tuning"
        private const val KEY_SPEED_MODE = "speed_mode"
        private const val PROFILE_VERSION = 1

        private const val WARMUP_TOKENS = 32
        private const val CALIBRATION_PROMPT_TOKENS = 96
        private const val DECODE_PROMPT_TOKENS = 16
        private const val CALIBRATION_GEN_TOKENS = 12
        private const val MAX_CALIBRATION_MS = 45_000L
        private const val MIN_GAIN = 0.05
        private const val EMA_ALPHA = 0.3
        private const val MIN_PREFILL_SAMPLE = 64L
        private const val MIN_DECODE_SAMPLE = 8L
        private const val SHORT_PROMPT_DISCOUNT = 0.8

        /** Evidence never shrinks below this, so even slow phones get a few passages. */
        const val MIN_EVIDENCE_TOKENS = 600

        @Volatile private var instance: PerformanceTuner? = null

        fun getInstance(context: Context): PerformanceTuner =
            instance ?: synchronized(this) { instance ?: PerformanceTuner(context).also { instance = it } }

        /** Pure budget rule (unit-tested): tokens readable in [seconds] minus the other new tokens, floored. */
        fun evidenceCapFor(prefillTps: Double, seconds: Double, otherNewTokens: Int): Int =
            ((prefillTps * seconds).toInt() - otherNewTokens.coerceAtLeast(0)).coerceAtLeast(MIN_EVIDENCE_TOKENS)
    }

    // ---------------------------------------------------------------------------------------------
    // Persistence
    // ---------------------------------------------------------------------------------------------

    /** Keeps stored thread counts inside today's safe limits (the topology or limits may have changed). */
    private fun sanitize(p: Profile): Profile = p.copy(
        decodeThreads = p.decodeThreads.coerceIn(1, topology.maxSafeThreads),
        batchThreads = p.batchThreads.coerceIn(1, topology.maxSafeThreads)
    )

    private fun load(k: String): Profile? {
        val raw = prefs.getString("profile:$k", null) ?: return null
        return runCatching {
            val v = raw.split(",")
            Profile(v[0].toInt(), v[1].toInt(), v[2].toDouble(), v[3].toDouble(), v[4].toDouble(), v[5].toDouble(), v[6].toBoolean())
        }.getOrNull()
    }

    private fun save(k: String, p: Profile) {
        if (k.isEmpty()) return
        val raw = listOf(p.decodeThreads, p.batchThreads, p.calibratedPrefillTps, p.calibratedDecodeTps, p.observedPrefillTps, p.observedDecodeTps, p.calibrated)
            .joinToString(",")
        prefs.edit().putString("profile:$k", raw).apply()
    }

    private fun appVersion(): Long = runCatching {
        val info = app.packageManager.getPackageInfo(app.packageName, 0)
        if (Build.VERSION.SDK_INT >= 28) info.longVersionCode else @Suppress("DEPRECATION") info.versionCode.toLong()
    }.getOrDefault(0L)
}
