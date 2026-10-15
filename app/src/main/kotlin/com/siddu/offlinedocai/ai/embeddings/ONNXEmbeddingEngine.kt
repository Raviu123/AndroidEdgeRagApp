package com.siddu.offlinedocai.ai.embeddings

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.siddu.offlinedocai.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.LongBuffer
import kotlin.math.sqrt

/**
 * On-device sentence embedding with ONNX Runtime for BERT-family models (all-MiniLM-L6-v2,
 * BGE-small): WordPiece tokenization, mean or CLS pooling (per model), L2 normalisation, and
 * padded batches with an attention mask for fast indexing.
 *
 * The bundled MiniLM engine is a process-wide instance ([getInstance]); downloaded ONNX models get
 * their own engine from [EmbeddingManager].
 */
class ONNXEmbeddingEngine private constructor(
    private val appContext: Context?,
    val spec: EmbeddingModelSpec,
    /** Null for the bundled model (extracted from assets on first use). */
    private val modelFile: File?,
    private val allowMockInJvmTests: Boolean
) : EmbeddingModel {

    private val TAG = "ONNXEmbeddingEngine"

    companion object {
        /** Persisted with every chunk of the default model; change it whenever the model or pooling changes. */
        const val MODEL_IDENTIFIER = AvailableEmbeddingModels.MINILM_ONNX_ID
        const val EMBEDDING_DIM = 384
        /** The model's position limit is 512, but it was trained on 128–256 token inputs. */
        const val MAX_SEQUENCE_LENGTH = 256
        /** Usable content tokens per input (excludes [CLS] and [SEP]). Chunks are sized to fit this. */
        const val MAX_CONTENT_TOKENS = MAX_SEQUENCE_LENGTH - 2

        private const val BATCH_SIZE = 8

        @Volatile
        private var instance: ONNXEmbeddingEngine? = null

        /** The bundled default model (all-MiniLM-L6-v2). */
        fun getInstance(context: Context): ONNXEmbeddingEngine =
            instance ?: synchronized(this) {
                instance ?: ONNXEmbeddingEngine(context.applicationContext, AvailableEmbeddingModels.default, null, false).also { instance = it }
            }

        /** Engine for a downloaded ONNX model file. */
        fun forFile(context: Context, spec: EmbeddingModelSpec, file: File): ONNXEmbeddingEngine =
            ONNXEmbeddingEngine(context.applicationContext, spec, file, false)

        /** Test-only engine that falls back to a deterministic hash embedding when ONNX Runtime is unavailable. */
        fun createForJvmTests(context: Context? = null): ONNXEmbeddingEngine =
            ONNXEmbeddingEngine(context, AvailableEmbeddingModels.default, null, true)

        /**
         * Copies an asset to [target] via a temp file + rename, so an interrupted copy can never leave a
         * truncated model behind. The marker is the asset's build-time SHA-256, so a changed asset is
         * always re-extracted.
         */
        internal fun extractAssetAtomically(context: Context, assetPath: String, target: File, version: String): File {
            val marker = File(target.parentFile, target.name + ".version")
            if (target.exists() && marker.exists() && marker.readText() == version) return target
            target.parentFile?.mkdirs()
            val tmp = File(target.parentFile, target.name + ".tmp")
            context.assets.open(assetPath).use { input -> tmp.outputStream().use { input.copyTo(it) } }
            if (target.exists()) target.delete()
            if (!tmp.renameTo(target)) {
                tmp.delete()
                throw IllegalStateException("Could not move extracted $assetPath into place")
            }
            marker.writeText(version)
            return target
        }
    }

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    @Volatile private var tokenizer: WordPieceTokenizer? = null
    private var hasTokenTypeIds = false
    @Volatile var lastError: String? = null
        private set

    override val modelId: String get() = spec.id
    val dimension: Int get() = spec.dimension

    @Synchronized
    fun initialize(): Boolean {
        if (ortSession != null && tokenizer != null) return true
        val context = appContext ?: return false
        val startedAt = SystemClock.elapsedRealtime()
        return try {
            val modelDir = File(context.filesDir, "models/embedding").apply { mkdirs() }
            val onnxFile = modelFile ?: extractAssetAtomically(
                context, spec.bundledAsset!!, File(modelDir, spec.fileName), BuildConfig.EMBEDDING_ASSET_SHA256
            )
            val vocabFile = extractAssetAtomically(
                context, AvailableEmbeddingModels.VOCAB_ASSET, File(modelDir, "vocab.txt"), BuildConfig.EMBEDDING_ASSET_SHA256
            )

            val env = OrtEnvironment.getEnvironment()
            // Embedding is compute-bound like prompt processing; cap at 4 because the 22M-parameter model gains little beyond that.
            val threads = com.siddu.offlinedocai.ai.llm.CpuTopology.device.defaultBatchThreads.coerceIn(1, 4)
            val options = OrtSession.SessionOptions().apply { setIntraOpNumThreads(threads) }
            val session = env.createSession(onnxFile.absolutePath, options)
            hasTokenTypeIds = session.inputNames.contains("token_type_ids")
            tokenizer = WordPieceTokenizer(vocabFile)
            ortEnv = env
            ortSession = session
            lastError = null
            Log.i(TAG, "Embedding model ${spec.id} ready in ${SystemClock.elapsedRealtime() - startedAt} ms")
            true
        } catch (e: Throwable) {
            lastError = e.message ?: e.javaClass.simpleName
            Log.e(TAG, "Failed to initialize ONNX embedding session for ${spec.id}: ${e.message}", e)
            close()
            false
        }
    }

    fun isLoaded(): Boolean = ortSession != null && tokenizer != null

    /** Tokenizer used for chunk sizing; guarantees chunk sizes are measured exactly like the model sees them. */
    fun requireTokenizer(): WordPieceTokenizer {
        tokenizer?.let { return it }
        check(initialize()) { "The on-device embedding model failed to load: ${lastError ?: "unknown error"}" }
        return tokenizer!!
    }

    fun countTokens(text: String): Int = requireTokenizer().countTokens(text)

    override suspend fun getEmbedding(text: String): FloatArray = embedBatch(listOf(text)).single()

    override suspend fun embedQuery(text: String): FloatArray = embedBatch(listOf(spec.queryPrefix + text)).single()

    override suspend fun embedDocuments(texts: List<String>): List<FloatArray> = embedBatch(texts)

    private suspend fun embedBatch(texts: List<String>): List<FloatArray> = withContext(Dispatchers.Default) {
        if (texts.isEmpty()) return@withContext emptyList()
        if (!isLoaded() && !initialize()) {
            if (allowMockInJvmTests) return@withContext texts.map { deterministicMockEmbedding(it.trim()) }
            throw IllegalStateException("The on-device embedding model failed to load: ${lastError ?: "unknown error"}")
        }
        val tok = tokenizer ?: throw IllegalStateException("Tokenizer is null")
        val encoded = texts.map { tok.encode(it.trim(), spec.maxSequenceLength) }
        val out = arrayOfNulls<FloatArray>(texts.size)
        // Batch inputs of similar length together to keep padding small.
        val order = encoded.indices.sortedBy { encoded[it].size }
        for (group in order.chunked(BATCH_SIZE)) {
            val vectors = try {
                runBatch(group.map { encoded[it] })
            } catch (e: Throwable) {
                Log.e(TAG, "Embedding inference failed: ${e.message}", e)
                if (allowMockInJvmTests) group.map { deterministicMockEmbedding(texts[it].trim()) }
                else throw IllegalStateException("Embedding inference failed: ${e.message}", e)
            }
            group.forEachIndexed { k, idx -> out[idx] = vectors[k] }
        }
        out.map { it!! }
    }

    private fun runBatch(batch: List<IntArray>): List<FloatArray> {
        val session = ortSession ?: throw IllegalStateException("ONNX session is null")
        val env = ortEnv ?: OrtEnvironment.getEnvironment()
        val seqLen = batch.maxOf { it.size }
        val n = batch.size
        val ids = LongArray(n * seqLen)
        val mask = LongArray(n * seqLen)
        for ((row, tokens) in batch.withIndex()) {
            for (j in tokens.indices) {
                ids[row * seqLen + j] = tokens[j].toLong()
                mask[row * seqLen + j] = 1L
            }
        }
        val shape = longArrayOf(n.toLong(), seqLen.toLong())
        val inputs = HashMap<String, OnnxTensor>()
        try {
            inputs["input_ids"] = OnnxTensor.createTensor(env, LongBuffer.wrap(ids), shape)
            inputs["attention_mask"] = OnnxTensor.createTensor(env, LongBuffer.wrap(mask), shape)
            if (hasTokenTypeIds) inputs["token_type_ids"] = OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(n * seqLen)), shape)
            session.run(inputs).use { results ->
                val output = results[0].value as? Array<*> ?: throw IllegalStateException("Unexpected output tensor type")
                return List(n) { row ->
                    val pooled = when (val first = output[row]) {
                        is FloatArray -> first.copyOf() // already pooled [batch, dim]
                        is Array<*> -> {
                            @Suppress("UNCHECKED_CAST")
                            val tokens = first as Array<FloatArray>
                            if (spec.pooling == EmbeddingPooling.CLS) tokens[0].copyOf() else meanPool(tokens, batch[row].size)
                        }
                        else -> throw IllegalStateException("Unexpected tensor element type: ${first?.javaClass}")
                    }
                    val normalized = normalize(pooled)
                    check(normalized.none { it.isNaN() } && normalized.any { it != 0f }) { "Embedding inference produced an invalid vector" }
                    normalized
                }
            }
        } finally {
            inputs.values.forEach { runCatching { it.close() } }
        }
    }

    /** Mean pooling over the first [length] (unpadded) positions. */
    private fun meanPool(tokenEmbeddings: Array<FloatArray>, length: Int): FloatArray {
        val hidden = tokenEmbeddings[0].size
        val mean = FloatArray(hidden)
        for (t in 0 until length) {
            val vec = tokenEmbeddings[t]
            for (j in 0 until hidden) mean[j] += vec[j]
        }
        for (j in 0 until hidden) mean[j] /= length.toFloat()
        return mean
    }

    private fun normalize(vector: FloatArray): FloatArray {
        var sumSq = 0.0
        for (v in vector) sumSq += (v * v).toDouble()
        val norm = sqrt(sumSq).toFloat()
        if (norm > 1e-12f) for (i in vector.indices) vector[i] /= norm
        return vector
    }

    /** JVM-test fallback: bag-of-words hashing. Captures lexical overlap only — never used on device. */
    private fun deterministicMockEmbedding(text: String): FloatArray {
        val vector = FloatArray(EMBEDDING_DIM)
        val words = text.lowercase().split(Regex("[^\\p{L}\\p{N}]+")).filter { it.isNotBlank() }
        if (words.isEmpty()) {
            vector[0] = 1f
            return vector
        }
        for (w in words) {
            vector[Math.floorMod(w.hashCode(), EMBEDDING_DIM)] += 1f
            vector[Math.floorMod((w + "_salt").hashCode(), EMBEDDING_DIM)] += 0.5f
        }
        return normalize(vector)
    }

    @Synchronized
    fun close() {
        runCatching { ortSession?.close() }
        ortSession = null
        ortEnv = null // OrtEnvironment is a process singleton; never close it.
        tokenizer = null
    }
}
