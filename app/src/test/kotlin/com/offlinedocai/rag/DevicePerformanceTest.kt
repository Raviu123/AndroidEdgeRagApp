package com.offlinedocai.rag

import com.offlinedocai.ai.llm.CpuTopology
import com.offlinedocai.ai.llm.PerformanceTuner
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** CPU layouts of real phone classes, expressed as /sys values; nothing here is tuned for one phone. */
class DevicePerformanceTest {

    private fun topo(capacities: List<Long?>, freqs: List<Long?> = capacities.map { null }) =
        CpuTopology.fromValues(capacities.size, capacities, freqs)

    @Test
    fun budgetPhone2Big6Little() {
        // Helio G85 class: 2x A75 + 6x A55.
        val t = topo(List(2) { 1024L } + List(6) { 403L })
        assertEquals(listOf(2, 6), t.clusters)
        assertEquals("capacity", t.source)
        assertEquals(2, t.fastCores)
        assertEquals(6, t.maxSafeThreads)          // 2 cores stay free for the UI
        assertEquals(2, t.defaultDecodeThreads)     // generation on the fast cores
        assertEquals(5, t.defaultBatchThreads)      // prompt processing adds half the little cores
        assertEquals(listOf(2, 5, 6), t.batchCandidates)
        assertEquals(listOf(2, 4), t.decodeCandidates)
    }

    @Test
    fun flagshipPrimeBigLittle() {
        // 1 prime + 3 big + 4 little.
        val t = topo(List(1) { 1024L } + List(3) { 870L } + List(4) { 325L })
        assertEquals(listOf(1, 3, 4), t.clusters)
        assertEquals(4, t.fastCores)
        assertEquals(4, t.defaultDecodeThreads)
        assertEquals(6, t.defaultBatchThreads)
        assertTrue(t.batchCandidates.all { it <= t.maxSafeThreads })
    }

    @Test
    fun frequencyIsTheFallbackWhenCapacityIsUnreadable() {
        val t = CpuTopology.fromValues(8, List(8) { null }, List(4) { 2_300_000L } + List(4) { 1_800_000L })
        assertEquals(listOf(4, 4), t.clusters)
        assertEquals("frequency", t.source)
    }

    @Test
    fun uniformCoresAreAllFast() {
        val t = topo(List(8) { 1024L })
        assertEquals(listOf(8), t.clusters)
        assertEquals(0, t.littleCores)
        assertEquals(4, t.defaultDecodeThreads)     // decoding rarely gains beyond 4 threads
        assertEquals(6, t.defaultBatchThreads)
    }

    @Test
    fun unreadableSysfsAssumesHalfFast() {
        val t = CpuTopology.fromValues(8, List(8) { null }, List(8) { null })
        assertEquals("assumed", t.source)
        assertEquals(4, t.fastCores)
    }

    @Test
    fun smallCpusNeverGetZeroThreadsOrExceedTheirCores() {
        for (cores in 1..4) {
            val t = CpuTopology.fromValues(cores, List(cores) { null }, List(cores) { null })
            val all = t.batchCandidates + t.decodeCandidates + t.defaultBatchThreads + t.defaultDecodeThreads
            assertTrue("cores=$cores $all", all.all { it in 1..cores })
        }
    }

    @Test
    fun evidenceBudgetFollowsMeasuredSpeed() {
        // Slow phone: 15 tok/s for 35 s = 525 tokens, minus 100 other tokens -> floored at the minimum.
        assertEquals(PerformanceTuner.MIN_EVIDENCE_TOKENS, PerformanceTuner.evidenceCapFor(15.0, 35.0, 100))
        // Mid-range: 60 tok/s for 35 s = 2100 - 100.
        assertEquals(2000, PerformanceTuner.evidenceCapFor(60.0, 35.0, 100))
        // Flagship: far above any intent cap, so the intent cap decides.
        assertTrue(PerformanceTuner.evidenceCapFor(400.0, 35.0, 300) > 10_000)
    }
}
