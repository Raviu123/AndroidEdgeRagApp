package com.siddu.offlinedocai.ai.llm

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Metadata extracted from a GGUF model binary.
 */
data class GGUFMetadata(
    val filename: String,
    val architecture: String = "unknown",
    val modelName: String = "",
    val parameterCount: String = "",
    val quantization: String = "",
    val contextLength: Int = 2048,
    val sizeBytes: Long = 0L,
    val isValid: Boolean = true,
    val errorMessage: String? = null
)

/**
 * Fast, pure-Kotlin GGUF binary file parser.
 * Reads the GGUF header and metadata key-value pairs without loading model weights into memory.
 */
object GGUFMetadataParser {

    private const val GGUF_MAGIC = 0x46554747 // "GGUF" in little endian ('G', 'G', 'U', 'F')

    fun parse(file: File): GGUFMetadata {
        if (!file.exists()) {
            return GGUFMetadata(file.name, isValid = false, errorMessage = "File does not exist")
        }
        if (!file.canRead()) {
            return GGUFMetadata(file.name, isValid = false, errorMessage = "File is not readable")
        }
        if (file.length() < 1024) {
            return GGUFMetadata(file.name, isValid = false, errorMessage = "File size is too small or empty")
        }

        try {
            LeReader(java.io.BufferedInputStream(java.io.FileInputStream(file), 64 * 1024)).use { r ->
                val magic = r.int32()
                if (magic != GGUF_MAGIC) {
                    return GGUFMetadata(file.name, isValid = false, errorMessage = "Invalid or unsupported GGUF model")
                }
                val version = r.int32()
                if (version < 1 || version > 4) {
                    return GGUFMetadata(file.name, isValid = false, errorMessage = "Unsupported GGUF version: $version")
                }
                r.int64() // tensor count
                val kvCount = r.int64()
                if (kvCount <= 0 || kvCount > 100_000) {
                    return GGUFMetadata(file.name, isValid = false, errorMessage = "Invalid or unsupported GGUF model")
                }

                var architecture = ""
                var modelName = ""
                var paramCount = ""
                var quantization = ""
                var contextLength = 0

                for (i in 0 until kvCount) {
                    val key = r.string() ?: break
                    // Tokenizer tables (100k+ entries) follow the general/architecture keys: stop reading there.
                    if (key.startsWith("tokenizer.") && architecture.isNotEmpty()) break
                    val valType = r.int32()
                    when {
                        key == "general.architecture" -> (readValue(r, valType) as? String)?.let { architecture = it }
                        key == "general.name" -> (readValue(r, valType) as? String)?.let { modelName = it }
                        key == "general.size_label" -> (readValue(r, valType) as? String)?.let { if (paramCount.isEmpty()) paramCount = it.uppercase() }
                        key == "general.parameter_count" -> (readValue(r, valType) as? Number)?.let { paramCount = formatParamCount(it.toLong()) }
                        key == "general.file_type" -> (readValue(r, valType) as? Number)?.let { fileTypeToQuant(it.toInt())?.let { q -> quantization = q } }
                        key.endsWith(".context_length") -> (readValue(r, valType) as? Number)?.let { contextLength = it.toInt() }
                        else -> skipValue(r, valType)
                    }
                }

                if (architecture.isEmpty()) {
                    val fn = file.name.lowercase()
                    architecture = when {
                        fn.contains("qwen") -> "qwen2"
                        fn.contains("llama") -> "llama"
                        fn.contains("mistral") -> "mistral"
                        fn.contains("phi") -> "phi3"
                        fn.contains("gemma") -> "gemma"
                        else -> "llm"
                    }
                }
                if (quantization.isEmpty()) quantization = extractQuantFromFilename(file.name)
                if (paramCount.isEmpty()) paramCount = extractParamFromFilename(file.name)
                if (modelName.isEmpty()) modelName = deriveDisplayName(file.name, architecture, paramCount)

                return GGUFMetadata(
                    filename = file.name,
                    architecture = architecture,
                    modelName = modelName,
                    parameterCount = paramCount,
                    quantization = quantization,
                    contextLength = if (contextLength > 0) contextLength else 2048,
                    sizeBytes = file.length(),
                    isValid = true
                )
            }
        } catch (e: Exception) {
            return GGUFMetadata(file.name, isValid = false, errorMessage = "Invalid or unsupported GGUF model: ${e.message}")
        }
    }

    /** Little-endian reader over a buffered stream (no per-value system calls). */
    private class LeReader(private val input: java.io.InputStream) : java.io.Closeable {
        private val scratch = ByteArray(8)

        private fun fill(n: Int) {
            var off = 0
            while (off < n) {
                val read = input.read(scratch, off, n - off)
                if (read < 0) throw java.io.EOFException()
                off += read
            }
        }

        fun int32(): Int { fill(4); return ByteBuffer.wrap(scratch, 0, 4).order(ByteOrder.LITTLE_ENDIAN).int }
        fun int64(): Long { fill(8); return ByteBuffer.wrap(scratch, 0, 8).order(ByteOrder.LITTLE_ENDIAN).long }
        fun int16(): Short { fill(2); return ByteBuffer.wrap(scratch, 0, 2).order(ByteOrder.LITTLE_ENDIAN).short }
        fun float32(): Float { fill(4); return ByteBuffer.wrap(scratch, 0, 4).order(ByteOrder.LITTLE_ENDIAN).float }
        fun float64(): Double { fill(8); return ByteBuffer.wrap(scratch, 0, 8).order(ByteOrder.LITTLE_ENDIAN).double }
        fun byte(): Int { val b = input.read(); if (b < 0) throw java.io.EOFException(); return b }

        fun skip(n: Long) {
            var left = n
            while (left > 0) {
                val skipped = input.skip(left)
                if (skipped <= 0) { byte(); left-- } else left -= skipped
            }
        }

        /** Strings longer than 1 MB are skipped and returned as null. */
        fun string(): String? {
            val len = int64()
            if (len < 0) throw java.io.IOException("Negative string length")
            if (len > 1_000_000) { skip(len); return null }
            val bytes = ByteArray(len.toInt())
            var off = 0
            while (off < bytes.size) {
                val read = input.read(bytes, off, bytes.size - off)
                if (read < 0) throw java.io.EOFException()
                off += read
            }
            return String(bytes, Charsets.UTF_8)
        }

        override fun close() = input.close()
    }

    private fun readValue(r: LeReader, valType: Int): Any? = when (valType) {
        0 -> r.byte()
        1 -> r.byte().toByte()
        2 -> r.int16().toInt() and 0xFFFF
        3 -> r.int16()
        4 -> r.int32().toLong() and 0xFFFFFFFFL
        5 -> r.int32()
        6 -> r.float32()
        7 -> r.byte() != 0
        8 -> r.string()
        10, 11 -> r.int64()
        12 -> r.float64()
        else -> { skipValue(r, valType); null }
    }

    private fun skipValue(r: LeReader, valType: Int) {
        when (valType) {
            0, 1, 7 -> r.skip(1)
            2, 3 -> r.skip(2)
            4, 5, 6 -> r.skip(4)
            10, 11, 12 -> r.skip(8)
            8 -> r.skip(r.int64())
            9 -> {
                val elemType = r.int32()
                val count = r.int64()
                val elemSize = when (elemType) {
                    0, 1, 7 -> 1L
                    2, 3 -> 2L
                    4, 5, 6 -> 4L
                    10, 11, 12 -> 8L
                    else -> 0L
                }
                if (elemSize > 0) r.skip(count * elemSize) else for (i in 0 until count) skipValue(r, elemType)
            }
            else -> throw java.io.IOException("Unknown GGUF value type $valType")
        }
    }

    private fun formatParamCount(p: Long): String {
        return when {
            p >= 1_000_000_000L -> String.format(java.util.Locale.US, "%.2fB", p / 1_000_000_000.0)
            p >= 1_000_000L -> String.format(java.util.Locale.US, "%.0fM", p / 1_000_000.0)
            else -> p.toString()
        }
    }

    private fun fileTypeToQuant(ft: Int): String? {
        return when (ft) {
            0 -> "F32"
            1 -> "F16"
            2 -> "Q4_0"
            3 -> "Q4_1"
            7 -> "Q8_0"
            8 -> "Q5_0"
            9 -> "Q5_1"
            10 -> "Q2_K"
            11 -> "Q3_K_S"
            12 -> "Q3_K_M"
            13 -> "Q3_K_L"
            14 -> "Q4_K_S"
            15 -> "Q4_K_M"
            16 -> "Q5_K_S"
            17 -> "Q5_K_M"
            18 -> "Q6_K"
            else -> null
        }
    }

    fun extractQuantFromFilename(filename: String): String {
        val regex = Regex("""(?i)\b(q\d+_[a-z0-9_]+|f16|f32|bf16)\b""")
        val match = regex.find(filename)
        return match?.value?.uppercase() ?: ""
    }

    fun extractParamFromFilename(filename: String): String {
        val regex = Regex("""(?i)\b(\d+(?:\.\d+)?[bm])\b""")
        val match = regex.find(filename)
        return match?.value?.uppercase() ?: ""
    }

    fun deriveDisplayName(filename: String, architecture: String, paramCount: String): String {
        val baseName = filename.removeSuffix(".gguf")
        return when {
            baseName.contains("qwen2.5-1.5b", ignoreCase = true) -> "Qwen 2.5 1.5B Instruct"
            baseName.contains("qwen2.5-0.5b", ignoreCase = true) -> "Qwen 2.5 0.5B Instruct"
            baseName.contains("llama-3.2-1b", ignoreCase = true) -> "Llama 3.2 1B Instruct"
            baseName.contains("llama-3.2-3b", ignoreCase = true) -> "Llama 3.2 3B Instruct"
            baseName.contains("qwen", ignoreCase = true) -> "Qwen $paramCount ($baseName)"
            baseName.contains("llama", ignoreCase = true) -> "Llama $paramCount ($baseName)"
            else -> baseName
        }
    }
}
