package com.siddu.offlinedocai

import android.content.Context
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import androidx.room.Room
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.siddu.offlinedocai.ai.llm.AnswerLength
import com.siddu.offlinedocai.ai.retrieval.CrossEncoderReranker
import com.siddu.offlinedocai.ai.retrieval.VectorStore
import com.siddu.offlinedocai.data.AppDatabase
import com.siddu.offlinedocai.data.entities.DocumentEntity
import com.siddu.offlinedocai.domain.repository.ChatRepository
import com.siddu.offlinedocai.domain.repository.DocumentRepository
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * End-to-end answer-quality benchmark on OWASP_MASVS.pdf with the model installed on the device.
 * It asks the fixed question set through the real [ChatRepository] (one conversation, so follow-up
 * handling is exercised as in the app) and writes answers, the evidence shown to the model and
 * timings to files/bench/<label>.json for scoring on the PC (benchmarks/masvs_eval/score.py).
 *
 * Not part of the normal test run: it needs the PDF and a GGUF model on the device and takes minutes.
 *   adb shell am instrument -w -e class com.siddu.offlinedocai.MasvsBenchmarkTest -e label baseline \
 *     com.siddu.offlinedocai.test/androidx.test.runner.AndroidJUnitRunner
 * Optional arguments:
 *   -e only 2,5,12       question numbers (2..19 as in the evaluation sheets) to rerun a subset
 *   -e rerank false      answer without the cross-encoder reranker (default: on, as in the app)
 *   -e generate false    retrieval only: no answers, just what each question would retrieve (seconds)
 * Every run also records retrieval with and without the reranker ("retrieval" in the JSON), so the
 * reranker's effect and its score thresholds can be checked from any run.
 */
@RunWith(AndroidJUnit4::class)
class MasvsBenchmarkTest {

    private val questions = listOf(
        "What is MASVS?",
        "What are the main security areas covered by MASVS?",
        "What does MASVS say about secure data storage?",
        "What security measures are recommended for network communication?",
        "What is the purpose of cryptography in mobile applications?",
        "What security measures should be used for user authentication?small paragraph anwer",
        "What does MASVS recommend for protecting sensitive data? explain in a short paragraph",
        "Why should mobile applications validate data received from external sources?",
        "What is the purpose of secure session management?",
        "What security risks can arise from improper data storage?",
        "What does MASVS recommend for protecting data transmitted over a network?",
        "How does MASVS address privacy in mobile applications?",
        "What is the purpose of resilience against reverse engineering?",
        "What security considerations apply to cryptographic keys?",
        "How do secure data storage and cryptography work together to protect sensitive information?",
        "How do authentication and session management help protect a mobile application?",
        "How do secure network communication and certificate validation help prevent attacks?",
        "How do data storage security, privacy, and cryptography work together to protect user information?"
    )

    @Test
    fun answerMasvsQuestions() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val args = InstrumentationRegistry.getArguments()
        val rerank = args.getString("rerank")?.toBooleanStrictOrNull() ?: true
        val generate = args.getString("generate")?.toBooleanStrictOrNull() ?: true
        val stamp = java.text.SimpleDateFormat("MMdd-HHmm", java.util.Locale.US).format(java.util.Date())
        val label = args.getString("label") ?: "${if (rerank) "rerank" else "norerank"}${if (generate) "" else "-retrieval"}-$stamp"
        val only = args.getString("only")?.split(',')?.mapNotNull { it.trim().toIntOrNull() }?.toSet()

        val pdf = findPdf(context)
        assumeTrue("OWASP_MASVS.pdf not found on the device", pdf != null)
        assumeTrue("No GGUF model installed", File(context.filesDir, "models/llm").listFiles()?.any { it.extension == "gguf" } == true)

        val db = Room.inMemoryDatabaseBuilder(context, AppDatabase::class.java).build()
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
        var importedCopy: File? = null
        try {
            val indexStart = SystemClock.elapsedRealtime()
            DocumentRepository(context, db).importDocument(Uri.fromFile(pdf))
            val doc = waitForIndex(db)
            val indexMs = SystemClock.elapsedRealtime() - indexStart
            importedCopy = doc.localPath?.let(::File)?.takeIf { it.absolutePath != pdf!!.absolutePath }
            Log.i(TAG, "Indexed ${doc.name} in $indexMs ms (${db.chunkDao().countForDocument(doc.id)} chunks)")

            val reranker = CrossEncoderReranker.getInstance(context)
            val plain = ChatRepository(context, db, VectorStore(db.chunkDao()), scope)
            val reranked = ChatRepository(context, db, VectorStore(db.chunkDao(), reranker), scope)
            val chat = if (rerank) reranked else plain

            // Retrieval alone, both ways (fast): what reaches the model and the reranker's scores.
            val retrieval = JSONArray()
            questions.forEachIndexed { i, question ->
                val number = i + 2
                if (only != null && number !in only) return@forEachIndexed
                val entry = JSONObject().put("n", number).put("question", question)
                for ((key, repo) in listOf("plain" to plain, "rerank" to reranked)) {
                    val started = SystemClock.elapsedRealtime()
                    val r = repo.retrieve(question, listOf(doc.id), previousQuestion = null)
                    val ms = SystemClock.elapsedRealtime() - started
                    val focused = ChatRepository.focusOnRelevant(r.results).map { it.chunk.id }.toSet()
                    entry.put(key, JSONObject()
                        .put("ms", ms)
                        .put("intent", r.classification.intent.name)
                        .put("passesGate", r.passesGate)
                        .put("results", JSONArray().apply {
                            r.results.take(12).forEach { x ->
                                put(JSONObject()
                                    .put("chunk", x.chunk.id)
                                    .put("pages", x.chunk.pageSpan)
                                    .put("source", x.source.name)
                                    .put("dense", x.denseScore.toDouble())
                                    .put("bm25", x.bm25Score.toDouble())
                                    .put("rerank", x.rerankScore?.toDouble() ?: JSONObject.NULL)
                                    .put("kept", x.chunk.id in focused)
                                    .put("text", x.chunk.text.take(160)))
                            }
                        }))
                }
                retrieval.put(entry)
            }

            val conversationId = chat.startConversation("bench", doc.id)
            val out = JSONArray()
            questions.forEachIndexed { i, question ->
                val number = i + 2 // numbered like the evaluation sheets, where #1 is the welcome message
                if (!generate || (only != null && number !in only)) return@forEachIndexed
                val started = SystemClock.elapsedRealtime()
                chat.ask(conversationId, question, listOf(doc.id), AnswerLength.DEFAULT)
                val elapsed = SystemClock.elapsedRealtime() - started
                val reply = db.messageDao().getMessagesForConversation(conversationId).first().last()
                Log.i(TAG, "Q$number done in $elapsed ms")
                out.put(
                    JSONObject()
                        .put("n", number)
                        .put("question", question)
                        .put("answer", reply.content)
                        .put("sources", reply.sources?.let(::JSONArray) ?: JSONArray())
                        .put("metrics", reply.metrics?.let { runCatching { JSONObject(it) }.getOrNull() } ?: JSONObject.NULL)
                        .put("elapsedMs", elapsed)
                )
            }
            val result = JSONObject()
                .put("label", label)
                .put("device", "${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}")
                .put("model", File(context.filesDir, "models/llm").listFiles()?.firstOrNull { it.extension == "gguf" }?.name)
                .put("indexMs", indexMs)
                .put("chunks", db.chunkDao().countForDocument(doc.id))
                .put("rerank", rerank)
                .put("retrieval", retrieval)
                .put("results", out)
            val dir = File(context.filesDir, "bench").apply { mkdirs() }
            File(dir, "$label.json").writeText(result.toString(2))
            Log.i(TAG, "Wrote ${File(dir, "$label.json")}")
        } finally {
            scope.cancel()
            db.close()
            importedCopy?.delete() // the import copied the PDF into the app's documents folder
        }
    }

    private fun findPdf(context: Context): File? =
        File(context.filesDir, "documents").listFiles()?.firstOrNull { it.name.endsWith("OWASP_MASVS.pdf") }
            ?: File(context.filesDir, "bench/OWASP_MASVS.pdf").takeIf { it.exists() }

    private suspend fun waitForIndex(db: AppDatabase): DocumentEntity = withTimeout(900_000) {
        while (true) {
            val doc = db.documentDao().getAllDocumentsOnce().firstOrNull()
            if (doc?.status == DocumentRepository.STATUS_COMPLETED) return@withTimeout doc
            if (doc?.status == DocumentRepository.STATUS_FAILED) throw AssertionError("Import failed: ${doc.errorMessage}")
            delay(500)
        }
        @Suppress("UNREACHABLE_CODE") throw IllegalStateException()
    }

    private companion object {
        const val TAG = "MasvsBenchmark"
    }
}
