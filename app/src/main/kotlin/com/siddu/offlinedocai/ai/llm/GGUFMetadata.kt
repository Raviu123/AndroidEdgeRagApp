package com.siddu.offlinedocai.ai.llm

import java.io.File
import java.io.RandomAccessFile
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
            RandomAccessFile(file, "r").use { raf ->
                val channel = raf.channel
                val headerBuf = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN)
                channel.read(headerBuf)
                headerBuf.flip()

                if (headerBuf.remaining() < 24) {
                    return GGUFMetadata(file.name, isValid = false, errorMessage = "Invalid or unsupported GGUF model")
                }

                val magic = headerBuf.int
                if (magic != GGUF_MAGIC) {
                    return GGUFMetadata(file.name, isValid = false, errorMessage = "Invalid or unsupported GGUF model")
                }

                val version = headerBuf.int
                if (version < 1 || version > 4) {
                    return GGUFMetadata(file.name, isValid = false, errorMessage = "Unsupported GGUF version: $version")
                }

                val tensorCount = headerBuf.long
                val kvCount = headerBuf.long

                if (kvCount <= 0 || kvCount > 100_000) {
                    return GGUFMetadata(file.name, isValid = false, errorMessage = "Invalid or unsupported GGUF model")
                }

                var architecture = ""
                var modelName = ""
                var paramCount = ""
                var quantization = ""
                var contextLength = 2048

                // Parse metadata key-values
                for (i in 0 until kvCount) {
                    val key = readString(raf) ?: break
                    val valType = readInt32(raf) ?: break

                    when (key) {
                        "general.architecture" -> {
                            val v = readValue(raf, valType)
                            if (v is String) architecture = v
                        }
                        "general.name" -> {
                            val v = readValue(raf, valType)
                            if (v is String) modelName = v
                        }
                        "general.file_type" -> {
                            val v = readValue(raf, valType)
                            val ftInt = when (v) {
                                is Number -> v.toInt()
                                else -> -1
                            }
                            if (quantization.isEmpty() && ftInt >= 0) {
                                quantization = fileTypeToQuant(ftInt)
                            }
                        }
                        "$architecture.context_length", "qwen2.context_length", "llama.context_length" -> {
                            val v = readValue(raf, valType)
                            if (v is Number) contextLength = v.toInt()
                        }
                        "general.parameter_count" -> {
                            val v = readValue(raf, valType)
                            if (v is Number) {
                                val p = v.toLong()
                                paramCount = formatParamCount(p)
                            }
                        }
                        else -> {
                            skipValue(raf, valType)
                        }
                    }

                    // If we have collected the core fields and position is past the first 1MB of metadata,
                    // we can stop early if all key fields are populated
                    if (architecture.isNotEmpty() && modelName.isNotEmpty() && contextLength > 2048 && paramCount.isNotEmpty()) {
                        break
                    }
                }

                // Fallback detection from filename if metadata didn't specify explicitly
                val fn = file.name.lowercase()
                if (architecture.isEmpty()) {
                    architecture = when {
                        fn.contains("qwen") -> "qwen2"
                        fn.contains("llama") -> "llama"
                        fn.contains("mistral") -> "mistral"
                        fn.contains("phi") -> "phi3"
                        fn.contains("gemma") -> "gemma"
                        else -> "llm"
                    }
                }
                if (quantization.isEmpty()) {
                    quantization = extractQuantFromFilename(file.name)
                }
                if (paramCount.isEmpty()) {
                    paramCount = extractParamFromFilename(file.name)
                }
                if (modelName.isEmpty()) {
                    modelName = deriveDisplayName(file.name, architecture, paramCount)
                }

                return GGUFMetadata(
                    filename = file.name,
                    architecture = architecture,
                    modelName = modelName,
                    parameterCount = paramCount,
                    quantization = quantization,
                    contextLength = contextLength,
                    sizeBytes = file.length(),
                    isValid = true
                )
            }
        } catch (e: Exception) {
            return GGUFMetadata(file.name, isValid = false, errorMessage = "Invalid or unsupported GGUF model: ${e.message}")
        }
    }

    private fun readString(raf: RandomAccessFile): String? {
        val lenBuf = ByteArray(8)
        if (raf.read(lenBuf) < 8) return null
        val len = ByteBuffer.wrap(lenBuf).order(ByteOrder.LITTLE_ENDIAN).long
        if (len < 0 || len > 100_000) return null
        val strBytes = ByteArray(len.toInt())
        if (raf.read(strBytes) < len.toInt()) return null
        return String(strBytes, Charsets.UTF_8)
    }

    private fun readInt32(raf: RandomAccessFile): Int? {
        val buf = ByteArray(4)
        if (raf.read(buf) < 4) return null
        return ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN).int
    }

    private fun readValue(raf: RandomAccessFile, valType: Int): Any? {
        return when (valType) {
            0 -> raf.read() // UINT8
            1 -> raf.readByte() // INT8
            2 -> {
                val b = ByteArray(2)
                raf.readFully(b)
                ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).short.toInt() and 0xFFFF
            }
            3 -> {
                val b = ByteArray(2)
                raf.readFully(b)
                ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).short
            }
            4 -> {
                val b = ByteArray(4)
                raf.readFully(b)
                ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).int.toLong() and 0xFFFFFFFFL
            }
            5 -> {
                val b = ByteArray(4)
                raf.readFully(b)
                ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).int
            }
            6 -> {
                val b = ByteArray(4)
                raf.readFully(b)
                ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).float
            }
            7 -> raf.readByte() != 0.toByte() // BOOL
            8 -> readString(raf) // STRING
            9 -> { // ARRAY
                val elemType = readInt32(raf) ?: return null
                val lenBuf = ByteArray(8)
                raf.readFully(lenBuf)
                val elemCount = ByteBuffer.wrap(lenBuf).order(ByteOrder.LITTLE_ENDIAN).long
                // Skip reading massive arrays like vocab tokens, just skip bytes
                for (i in 0 until elemCount) {
                    skipValue(raf, elemType)
                    if (i > 100_000) break
                }
                null
            }
            10 -> {
                val b = ByteArray(8)
                raf.readFully(b)
                ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).long
            }
            11 -> {
                val b = ByteArray(8)
                raf.readFully(b)
                ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).long
            }
            12 -> {
                val b = ByteArray(8)
                raf.readFully(b)
                ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).double
            }
            else -> null
        }
    }

    private fun skipValue(raf: RandomAccessFile, valType: Int) {
        when (valType) {
            0, 1, 7 -> raf.skipBytes(1)
            2, 3 -> raf.skipBytes(2)
            4, 5, 6 -> raf.skipBytes(4)
            10, 11, 12 -> raf.skipBytes(8)
            8 -> {
                val lenBuf = ByteArray(8)
                if (raf.read(lenBuf) == 8) {
                    val len = ByteBuffer.wrap(lenBuf).order(ByteOrder.LITTLE_ENDIAN).long
                    if (len in 0..10_000_000) {
                        raf.seek(raf.filePointer + len)
                    }
                }
            }
            9 -> {
                val elemType = readInt32(raf) ?: return
                val lenBuf = ByteArray(8)
                if (raf.read(lenBuf) == 8) {
                    val count = ByteBuffer.wrap(lenBuf).order(ByteOrder.LITTLE_ENDIAN).long
                    if (elemType in 0..7) {
                        val elemSize = when (elemType) {
                            0, 1, 7 -> 1
                            2, 3 -> 2
                            4, 5, 6 -> 4
                            else -> 1
                        }
                        raf.seek(raf.filePointer + (count * elemSize))
                    } else if (elemType in 10..12) {
                        raf.seek(raf.filePointer + (count * 8))
                    } else {
                        for (i in 0 until count) {
                            skipValue(raf, elemType)
                            if (i > 500_000) break
                        }
                    }
                }
            }
        }
    }

    private fun formatParamCount(p: Long): String {
        return when {
            p >= 1_000_000_000L -> String.format(java.util.Locale.US, "%.2fB", p / 1_000_000_000.0)
            p >= 1_000_000L -> String.format(java.util.Locale.US, "%.0fM", p / 1_000_000.0)
            else -> p.toString()
        }
    }

    private fun fileTypeToQuant(ft: Int): String {
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
            else -> "Q4_K_M"
        }
    }

    fun extractQuantFromFilename(filename: String): String {
        val regex = Regex("""(?i)\b(q\d+_[a-z0-9_]+|f16|f32|bf16)\b""")
        val match = regex.find(filename)
        return match?.value?.uppercase() ?: "Q4_K_M"
    }

    fun extractParamFromFilename(filename: String): String {
        val regex = Regex("""(?i)\b(\d+(?:\.\d+)?[bm])\b""")
        val match = regex.find(filename)
        return match?.value?.uppercase() ?: "1.5B"
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
