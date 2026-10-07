package com.siddu.offlinedocai.data.download

import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

enum class ModelFileFormat { GGUF, ONNX }

/** Everything needed to fetch and verify one model file. */
data class DownloadSpec(
    val id: String,
    val displayName: String,
    val url: String,
    val destination: File,
    val sizeBytes: Long,
    val sha256: String,
    val format: ModelFileFormat
) {
    val partFile: File get() = File(destination.parentFile, destination.name + ".part")
}

/** The file failed verification; retrying the same bytes will not help. */
class ModelVerificationException(message: String) : Exception(message)

/**
 * Plain-JVM model file downloader (no Android dependencies, unit-testable):
 *  - downloads into `<name>.part` and **resumes** an existing part with an HTTP Range request,
 *  - follows redirects manually (Hugging Face → CDN), sending the Range header on every hop,
 *  - verifies the on-disk size, SHA-256 and file format before atomically moving the file into place.
 * A file is never visible under its final name unless it passed every check.
 */
class ModelFileDownloader(
    private val connectTimeoutMs: Int = 30_000,
    private val readTimeoutMs: Int = 60_000
) {
    /**
     * @param onProgress called with (bytesOnDisk, totalBytes) at most every ~250 ms.
     * @param checkCancelled throws (e.g. CancellationException) to abort; the part file is kept for resuming.
     */
    fun download(
        spec: DownloadSpec,
        onProgress: (Long, Long) -> Unit = { _, _ -> },
        onVerifying: () -> Unit = {},
        checkCancelled: () -> Unit = {}
    ): File {
        spec.destination.parentFile?.mkdirs()
        val part = spec.partFile
        if (part.exists() && part.length() > spec.sizeBytes) part.delete()

        if (!part.exists() || part.length() < spec.sizeBytes) {
            fetch(spec, part, onProgress, checkCancelled)
        }

        onVerifying()
        val failure = verify(spec, part)
        if (failure != null) {
            part.delete()
            throw ModelVerificationException(failure)
        }
        if (spec.destination.exists()) spec.destination.delete()
        if (!part.renameTo(spec.destination)) {
            part.copyTo(spec.destination, overwrite = true)
            part.delete()
        }
        return spec.destination
    }

    private fun fetch(spec: DownloadSpec, part: File, onProgress: (Long, Long) -> Unit, checkCancelled: () -> Unit) {
        val existing = if (part.exists()) part.length() else 0L
        val conn = open(spec.url, existing)
        try {
            val code = conn.responseCode
            val append = when {
                existing > 0 && code == 206 -> {
                    val start = conn.getHeaderField("Content-Range")?.let { Regex("""bytes\s+(\d+)-""").find(it)?.groupValues?.get(1)?.toLongOrNull() }
                    if (start != existing) throw IOException("Server resumed at an unexpected offset ($start, expected $existing)")
                    true
                }
                existing > 0 && code == 416 -> return // part is already complete; verification decides
                code == HttpURLConnection.HTTP_OK -> false
                else -> throw IOException("Server returned HTTP $code for ${spec.displayName}")
            }
            var onDisk = if (append) existing else 0L
            var lastReport = 0L
            conn.inputStream.buffered(64 * 1024).use { input ->
                FileOutputStream(part, append).buffered(64 * 1024).use { output ->
                    val buffer = ByteArray(64 * 1024)
                    while (true) {
                        checkCancelled()
                        val read = input.read(buffer)
                        if (read < 0) break
                        output.write(buffer, 0, read)
                        onDisk += read
                        if (onDisk > spec.sizeBytes) throw ModelVerificationException("The server sent more data than expected for ${spec.displayName}.")
                        val now = System.currentTimeMillis()
                        if (now - lastReport > 250) {
                            lastReport = now
                            onProgress(onDisk, spec.sizeBytes)
                        }
                    }
                }
            }
            onProgress(onDisk, spec.sizeBytes)
            if (onDisk < spec.sizeBytes) throw IOException("Connection closed early (${onDisk} of ${spec.sizeBytes} bytes)")
        } finally {
            conn.disconnect()
        }
    }

    private fun open(startUrl: String, rangeStart: Long): HttpURLConnection {
        var url = startUrl
        repeat(8) {
            val conn = (URL(url).openConnection() as HttpURLConnection).apply {
                instanceFollowRedirects = false
                connectTimeout = connectTimeoutMs
                readTimeout = readTimeoutMs
                setRequestProperty("User-Agent", "OfflineDocAI/1.0 (Android)")
                setRequestProperty("Accept-Encoding", "identity")
                if (rangeStart > 0) setRequestProperty("Range", "bytes=$rangeStart-")
            }
            val code = conn.responseCode
            if (code in 300..399 && code != 304) {
                val location = conn.getHeaderField("Location") ?: throw IOException("Redirect without a location")
                conn.disconnect()
                url = URL(URL(url), location).toString()
                return@repeat
            }
            return conn
        }
        throw IOException("Too many redirects")
    }

    /** Returns a user-facing reason when [file] is not exactly the expected model, null when it is. */
    fun verify(spec: DownloadSpec, file: File): String? {
        if (!file.exists()) return "The downloaded file is missing."
        if (file.length() != spec.sizeBytes) return "Download incomplete: ${file.length()} of ${spec.sizeBytes} bytes."
        val head = ByteArray(4)
        FileInputStream(file).use { it.read(head) }
        val formatOk = when (spec.format) {
            ModelFileFormat.GGUF -> String(head, Charsets.US_ASCII) == "GGUF"
            ModelFileFormat.ONNX -> head[0] == 0x08.toByte() // ModelProto starts with field 1 (ir_version, varint)
        }
        if (!formatOk) return "The downloaded file is not a valid ${spec.format.name} model."
        val actual = sha256(file)
        if (!actual.equals(spec.sha256, ignoreCase = true)) return "Checksum mismatch: the downloaded file is damaged. Please try again."
        return null
    }

    companion object {
        fun sha256(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            FileInputStream(file).use { input ->
                val buffer = ByteArray(256 * 1024)
                while (true) {
                    val read = input.read(buffer)
                    if (read < 0) break
                    digest.update(buffer, 0, read)
                }
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }
    }
}
