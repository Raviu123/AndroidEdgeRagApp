package com.offlinedocai.ai.retrieval

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.offlinedocai.BuildConfig
import com.offlinedocai.ai.embeddings.AvailableEmbeddingModels
import com.offlinedocai.ai.embeddings.ONNXEmbeddingEngine
import com.offlinedocai.ai.embeddings.WordPieceTokenizer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.LongBuffer

/** Scores how well each passage answers a question; larger is better. */
interface PassageReranker {
    /** One score per passage, or null when the reranker is unavailable (callers keep their own order). */
    suspend fun score(question: String, passages: List<String>): FloatArray?
}

/**
 * Cross-encoder reranker (ms-marco-MiniLM-L-6-v2, ONNX INT8, ~23 MB, bundled in the APK).
 *
 * Unlike the bi-encoder used for search, it reads the question and the passage together, so it can
 * tell "data transmitted over a network" from "protecting stored data" even when both share most
 * words. Its output is a relevance logit: roughly > 0 for passages that answer the question and
 * strongly negative for unrelated ones, which also makes it usable as a "not in the document" signal.
 * It shares the uncased BERT vocabulary of the bundled embedder.
 */
class CrossEncoderReranker private constructor(private val appContext: Context) : PassageReranker {

    companion object {
        private const val TAG = "CrossEncoderReranker"
        const val ASSET = "models/reranker/ms-marco-MiniLM-L-6-v2.onnx"
        /** Question + passage tokens; chunks are at most ~180 tokens, questions rarely exceed 40. */
        private const val MAX_SEQUENCE_LENGTH = 256
        private const val MAX_QUESTION_TOKENS = 64
        private const val BATCH_SIZE = 8

        @Volatile private var instance: CrossEncoderReranker? = null

        fun getInstance(context: Context): CrossEncoderReranker =
            instance ?: synchronized(this) {
                instance ?: CrossEncoderReranker(context.applicationContext).also { instance = it }
            }
    }

    private var session: OrtSession? = null
    private var tokenizer: WordPieceTokenizer? = null
    private var hasTokenTypeIds = true
    @Volatile private var failed = false

    @Synchronized
    private fun initialize(): Boolean {
        if (session != null) return true
        if (failed) return false
        val startedAt = SystemClock.elapsedRealtime()
        return try {
            val dir = File(appContext.filesDir, "models/reranker").apply { mkdirs() }
            val onnx = ONNXEmbeddingEngine.extractAssetAtomically(appContext, ASSET, File(dir, File(ASSET).name), BuildConfig.RERANKER_ASSET_SHA256)
            val vocab = ONNXEmbeddingEngine.extractAssetAtomically(appContext, AvailableEmbeddingModels.VOCAB_ASSET, File(dir, "vocab.txt"), BuildConfig.RERANKER_ASSET_SHA256)
            val threads = com.offlinedocai.ai.llm.CpuTopology.device.defaultBatchThreads.coerceIn(1, 4)
            val options = OrtSession.SessionOptions().apply { setIntraOpNumThreads(threads) }
            val created = OrtEnvironment.getEnvironment().createSession(onnx.absolutePath, options)
            hasTokenTypeIds = created.inputNames.contains("token_type_ids")
            tokenizer = WordPieceTokenizer(vocab)
            session = created
            Log.i(TAG, "Reranker ready in ${SystemClock.elapsedRealtime() - startedAt} ms")
            true
        } catch (t: Throwable) {
            // Retrieval keeps working without reranking; do not retry on every question.
            failed = true
            Log.e(TAG, "Reranker unavailable: ${t.message}", t)
            false
        }
    }

    override suspend fun score(question: String, passages: List<String>): FloatArray? = withContext(Dispatchers.Default) {
        if (passages.isEmpty()) return@withContext FloatArray(0)
        if (!initialize()) return@withContext null
        val tok = tokenizer ?: return@withContext null
        try {
            val startedAt = SystemClock.elapsedRealtime()
            val q = tok.tokenizeToIds(question.trim()).take(MAX_QUESTION_TOKENS)
            val pairs = passages.map { encodePair(tok, q, tok.tokenizeToIds(it.trim())) }
            val scores = FloatArray(passages.size)
            // Similar lengths share a batch so little time is spent on padding.
            val order = pairs.indices.sortedBy { pairs[it].first.size }
            for (group in order.chunked(BATCH_SIZE)) {
                val logits = runBatch(group.map { pairs[it] })
                group.forEachIndexed { k, idx -> scores[idx] = logits[k] }
            }
            Log.i(TAG, "Reranked ${passages.size} passages in ${SystemClock.elapsedRealtime() - startedAt} ms")
            scores
        } catch (t: Throwable) {
            Log.e(TAG, "Reranking failed: ${t.message}", t)
            null
        }
    }

    /** [CLS] question [SEP] passage [SEP] with segment ids 0 / 1; the passage is truncated to fit. */
    private fun encodePair(tok: WordPieceTokenizer, question: List<Int>, passage: List<Int>): Pair<IntArray, IntArray> {
        val room = MAX_SEQUENCE_LENGTH - question.size - 3
        val body = passage.take(room.coerceAtLeast(0))
        val ids = IntArray(question.size + body.size + 3)
        val types = IntArray(ids.size)
        var i = 0
        ids[i++] = tok.clsTokenId
        for (t in question) ids[i++] = t
        ids[i++] = tok.sepTokenId
        val passageStart = i
        for (t in body) ids[i++] = t
        ids[i++] = tok.sepTokenId
        for (j in passageStart until ids.size) types[j] = 1
        return ids to types
    }

    private fun runBatch(batch: List<Pair<IntArray, IntArray>>): FloatArray {
        val s = session ?: throw IllegalStateException("Reranker session is null")
        val env = OrtEnvironment.getEnvironment()
        val seqLen = batch.maxOf { it.first.size }
        val n = batch.size
        val ids = LongArray(n * seqLen)
        val mask = LongArray(n * seqLen)
        val types = LongArray(n * seqLen)
        for ((row, pair) in batch.withIndex()) {
            for (j in pair.first.indices) {
                ids[row * seqLen + j] = pair.first[j].toLong()
                mask[row * seqLen + j] = 1L
                types[row * seqLen + j] = pair.second[j].toLong()
            }
        }
        val shape = longArrayOf(n.toLong(), seqLen.toLong())
        val inputs = HashMap<String, OnnxTensor>()
        try {
            inputs["input_ids"] = OnnxTensor.createTensor(env, LongBuffer.wrap(ids), shape)
            inputs["attention_mask"] = OnnxTensor.createTensor(env, LongBuffer.wrap(mask), shape)
            if (hasTokenTypeIds) inputs["token_type_ids"] = OnnxTensor.createTensor(env, LongBuffer.wrap(types), shape)
            s.run(inputs).use { results ->
                @Suppress("UNCHECKED_CAST")
                val logits = results[0].value as Array<FloatArray> // [batch, 1]
                return FloatArray(n) { logits[it][0] }
            }
        } finally {
            inputs.values.forEach { runCatching { it.close() } }
        }
    }
}
