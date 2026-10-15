package com.siddu.offlinedocai.ai.llm

import java.io.File

/**
 * CPU layout of the device, read at runtime (never hard-coded per phone).
 *
 * Cores are grouped into clusters by the kernel's per-core performance rating (`cpu_capacity`), or by
 * maximum frequency when that is unavailable. The slowest cluster counts as "little" cores; every other
 * cluster (big, prime) counts as "fast". When neither source can be read, half of the cores are
 * assumed fast (the common big.LITTLE split); calibration then measures what really works.
 */
data class CpuTopology(
    val totalCores: Int,
    /** Core counts per cluster, fastest cluster first, e.g. [2, 6] or [1, 3, 4]. */
    val clusters: List<Int>,
    /** Where [clusters] came from: "capacity", "frequency", "uniform" or "assumed". */
    val source: String
) {
    val littleCores: Int get() = if (clusters.size > 1) clusters.last() else 0
    val fastCores: Int get() = totalCores - littleCores

    /** Short label for logs and metrics, e.g. "2+6" or "1+3+4". */
    val label: String get() = clusters.joinToString("+")

    /** Cores kept free for the UI and the system: 2 on 8+ core CPUs, 1 on 4-7 cores, none below. */
    val maxSafeThreads: Int
        get() {
            val reserve = when {
                totalCores >= 8 -> 2
                totalCores >= 4 -> 1
                else -> 0
            }
            return (totalCores - reserve).coerceIn(1, MAX_THREADS)
        }

    /**
     * Generation is memory-bound and every step waits for its slowest thread, so it runs on the fast
     * cores only (at most 4: more threads rarely help single-token decoding).
     */
    val defaultDecodeThreads: Int get() = fastCores.coerceIn(1, minOf(4, maxSafeThreads))

    /** Prompt processing is compute-bound: fast cores plus half of the little cores, within the safe limit. */
    val defaultBatchThreads: Int get() = (fastCores + littleCores / 2).coerceIn(1, maxSafeThreads)

    /** Thread counts worth measuring for prompt processing (deduplicated, ascending). */
    val batchCandidates: List<Int>
        get() = listOf(fastCores, fastCores + littleCores / 2, maxSafeThreads)
            .map { it.coerceIn(1, maxSafeThreads) }.distinct().sorted()

    /** Thread counts worth measuring for generation (deduplicated, ascending). */
    val decodeCandidates: List<Int>
        get() = listOf(defaultDecodeThreads, minOf(fastCores + 2, maxSafeThreads), if (fastCores > 4) fastCores else 0)
            .filter { it > 0 }.map { it.coerceIn(1, maxSafeThreads) }.distinct().sorted()

    companion object {
        const val MAX_THREADS = 8

        /** Topology of this device, read once. */
        val device: CpuTopology by lazy { read() }

        private fun read(): CpuTopology {
            val total = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
            fun perCore(name: String) = (0 until total).map { i ->
                runCatching { File("/sys/devices/system/cpu/cpu$i/$name").readText().trim().toLong() }.getOrNull()
            }
            return fromValues(total, perCore("cpu_capacity"), perCore("cpufreq/cpuinfo_max_freq"))
        }

        /**
         * Builds the topology from per-core values (null = unreadable). Capacity is preferred because it
         * also separates cores of equal clock but different design; frequency is the fallback.
         */
        fun fromValues(totalCores: Int, capacities: List<Long?>, frequencies: List<Long?>): CpuTopology {
            val total = totalCores.coerceAtLeast(1)
            clustersOf(capacities, total)?.let { return CpuTopology(total, it, if (it.size > 1) "capacity" else "uniform") }
            clustersOf(frequencies, total)?.let { return CpuTopology(total, it, if (it.size > 1) "frequency" else "uniform") }
            val fast = (total + 1) / 2
            return CpuTopology(total, if (total > 1) listOf(fast, total - fast) else listOf(1), "assumed")
        }

        /** Groups equal values, fastest first; null unless every core has a positive value. */
        private fun clustersOf(values: List<Long?>, total: Int): List<Int>? {
            if (values.size != total || values.any { it == null || it <= 0 }) return null
            return values.filterNotNull().groupingBy { it }.eachCount().toSortedMap(compareByDescending { it }).values.toList()
        }
    }
}
