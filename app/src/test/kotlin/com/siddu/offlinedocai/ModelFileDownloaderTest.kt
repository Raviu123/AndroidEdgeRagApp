package com.siddu.offlinedocai

import com.siddu.offlinedocai.data.download.DownloadSpec
import com.siddu.offlinedocai.data.download.ModelFileDownloader
import com.siddu.offlinedocai.data.download.ModelFileFormat
import com.siddu.offlinedocai.data.download.ModelVerificationException
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.io.IOException
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicInteger
import kotlin.random.Random

class ModelFileDownloaderTest {

    @get:Rule
    val tmp = TemporaryFolder()

    private val payload: ByteArray = "GGUF".toByteArray() + Random(7).nextBytes(300_000)
    private val sha: String = MessageDigest.getInstance("SHA-256").digest(payload).joinToString("") { "%02x".format(it) }
    private val rangeRequests = AtomicInteger()
    /** When > 0, the server closes the connection after sending this many bytes (once). */
    @Volatile private var cutAfter = 0

    /** Minimal HTTP/1.1 server (the Android unit-test classpath has no JDK HttpServer). */
    private lateinit var server: java.net.ServerSocket
    private var serverThread: Thread? = null

    @Before
    fun startServer() {
        server = java.net.ServerSocket(0, 50, java.net.InetAddress.getByName("127.0.0.1"))
        serverThread = Thread {
            while (!server.isClosed) {
                val socket = try { server.accept() } catch (e: IOException) { break }
                socket.use { handle(it) }
            }
        }.apply { isDaemon = true; start() }
    }

    private fun handle(socket: java.net.Socket) {
        val reader = socket.getInputStream().bufferedReader(Charsets.ISO_8859_1)
        val requestLine = reader.readLine() ?: return
        val path = requestLine.split(" ").getOrNull(1) ?: "/"
        var range: String? = null
        while (true) {
            val line = reader.readLine() ?: break
            if (line.isEmpty()) break
            if (line.startsWith("Range:", ignoreCase = true)) range = line.substringAfter(":").trim()
        }
        val out = socket.getOutputStream()
        fun head(status: String, headers: List<String>) =
            out.write(("HTTP/1.1 $status\r\n" + headers.joinToString("") { "$it\r\n" } + "Connection: close\r\n\r\n").toByteArray(Charsets.ISO_8859_1))
        if (path == "/redirect") {
            head("302 Found", listOf("Location: /model.gguf", "Content-Length: 0"))
            return
        }
        val start = range?.let { Regex("""bytes=(\d+)-""").find(it)?.groupValues?.get(1)?.toInt() } ?: 0
        if (range != null) rangeRequests.incrementAndGet()
        val body = payload.copyOfRange(start, payload.size)
        if (start > 0) head("206 Partial Content", listOf("Content-Range: bytes $start-${payload.size - 1}/${payload.size}", "Content-Length: ${body.size}"))
        else head("200 OK", listOf("Content-Length: ${body.size}"))
        val limit = cutAfter
        if (limit in 1 until body.size) {
            cutAfter = 0
            out.write(body, 0, limit)
            out.flush()
            return // connection closes early
        }
        out.write(body)
        out.flush()
    }

    @After
    fun stopServer() = server.close()

    private fun url(path: String) = "http://127.0.0.1:${server.localPort}$path"

    private fun spec(path: String = "/model.gguf", size: Long = payload.size.toLong(), hash: String = sha) = DownloadSpec(
        id = "test", displayName = "Test model", url = url(path),
        destination = File(tmp.root, "models/model.gguf"), sizeBytes = size, sha256 = hash, format = ModelFileFormat.GGUF
    )

    @Test
    fun downloadsVerifiesAndInstallsFollowingRedirects() {
        val s = spec("/redirect")
        val file = ModelFileDownloader().download(s)
        assertArrayEquals(payload, file.readBytes())
        assertFalse("part file must be gone after install", s.partFile.exists())
    }

    @Test
    fun resumesPartialDownloadWithRangeRequest() {
        val s = spec()
        s.destination.parentFile!!.mkdirs()
        s.partFile.writeBytes(payload.copyOfRange(0, 100_000))
        val file = ModelFileDownloader().download(s)
        assertEquals(1, rangeRequests.get())
        assertArrayEquals(payload, file.readBytes())
    }

    @Test
    fun connectionDropKeepsPartForNextAttempt() {
        cutAfter = 120_000
        val s = spec()
        try {
            ModelFileDownloader().download(s)
            fail("expected a network error")
        } catch (e: IOException) {
            // expected
        }
        assertFalse(s.destination.exists())
        assertTrue("partial data is kept for resuming", s.partFile.exists() && s.partFile.length() > 0)
        val file = ModelFileDownloader().download(s) // retry resumes
        assertArrayEquals(payload, file.readBytes())
    }

    @Test
    fun wrongChecksumIsRejectedAndNothingInstalled() {
        val s = spec(hash = "0".repeat(64))
        try {
            ModelFileDownloader().download(s)
            fail("expected verification failure")
        } catch (e: ModelVerificationException) {
            assertTrue(e.message!!.contains("Checksum"))
        }
        assertFalse(s.destination.exists())
        assertFalse(s.partFile.exists())
    }

    @Test
    fun wrongSizeIsRejected() {
        val s = spec(size = payload.size + 10L)
        try {
            ModelFileDownloader().download(s)
            fail("expected failure")
        } catch (e: Exception) {
            // IOException (closed early) or verification failure: either way nothing is installed
        }
        assertFalse(s.destination.exists())
    }

    @Test
    fun verifyRejectsNonGgufContent() {
        val f = tmp.newFile("x.gguf").apply { writeBytes(ByteArray(payload.size)) }
        val reason = ModelFileDownloader().verify(spec(), f)
        assertTrue(reason!!.contains("not a valid GGUF"))
    }
}
