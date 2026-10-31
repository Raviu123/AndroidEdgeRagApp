package com.offlinedocai

import com.offlinedocai.ai.llm.GGUFMetadataParser
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class GGUFMetadataParserTest {

    @get:Rule
    val tempFolder = TemporaryFolder()

    @Test
    fun testRejectsNonExistentFile() {
        val file = File(tempFolder.root, "does_not_exist.gguf")
        val meta = GGUFMetadataParser.parse(file)
        assertFalse("Non-existent file should be invalid", meta.isValid)
        assertEquals("File does not exist", meta.errorMessage)
    }

    @Test
    fun testRejectsSmallOrEmptyFile() {
        val file = tempFolder.newFile("empty.gguf")
        val meta = GGUFMetadataParser.parse(file)
        assertFalse("Empty file should be invalid", meta.isValid)
        assertTrue("Error should mention size", meta.errorMessage?.contains("too small") == true)
    }

    @Test
    fun testRejectsInvalidMagic() {
        val file = tempFolder.newFile("fake.gguf")
        val bytes = ByteArray(2048)
        bytes[0] = 'N'.code.toByte()
        bytes[1] = 'O'.code.toByte()
        bytes[2] = 'P'.code.toByte()
        bytes[3] = 'E'.code.toByte()
        FileOutputStream(file).use { it.write(bytes) }

        val meta = GGUFMetadataParser.parse(file)
        assertFalse("File with invalid magic should be rejected", meta.isValid)
        assertEquals("Invalid or unsupported GGUF model", meta.errorMessage)
    }

    @Test
    fun testRejectsUnsupportedVersion() {
        val file = tempFolder.newFile("bad_version.gguf")
        val buf = ByteBuffer.allocate(2048).order(ByteOrder.LITTLE_ENDIAN)
        buf.put('G'.code.toByte())
        buf.put('G'.code.toByte())
        buf.put('U'.code.toByte())
        buf.put('F'.code.toByte())
        buf.putInt(99) // Invalid version 99
        buf.putLong(10L)
        buf.putLong(1L)
        FileOutputStream(file).use { it.write(buf.array()) }

        val meta = GGUFMetadataParser.parse(file)
        assertFalse("Unsupported version must be rejected", meta.isValid)
        assertTrue("Error should mention version", meta.errorMessage?.contains("version") == true)
    }

    @Test
    fun testParsesValidSyntheticGGUFHeader() {
        val file = tempFolder.newFile("qwen2.5-1.5b-instruct-q4_k_m.gguf")
        val buf = ByteBuffer.allocate(2048).order(ByteOrder.LITTLE_ENDIAN)

        // Magic: "GGUF"
        buf.put('G'.code.toByte())
        buf.put('G'.code.toByte())
        buf.put('U'.code.toByte())
        buf.put('F'.code.toByte())

        // Version: 3
        buf.putInt(3)

        // Tensor count: 120
        buf.putLong(120L)

        // KV count: 3
        buf.putLong(3L)

        // KV 1: "general.architecture" -> "qwen2"
        val keyArch = "general.architecture"
        buf.putLong(keyArch.length.toLong())
        buf.put(keyArch.toByteArray(Charsets.UTF_8))
        buf.putInt(8) // Type STRING = 8
        val valArch = "qwen2"
        buf.putLong(valArch.length.toLong())
        buf.put(valArch.toByteArray(Charsets.UTF_8))

        // KV 2: "general.name" -> "Qwen2.5 1.5B Instruct"
        val keyName = "general.name"
        buf.putLong(keyName.length.toLong())
        buf.put(keyName.toByteArray(Charsets.UTF_8))
        buf.putInt(8) // Type STRING = 8
        val valName = "Qwen2.5 1.5B Instruct"
        buf.putLong(valName.length.toLong())
        buf.put(valName.toByteArray(Charsets.UTF_8))

        // KV 3: "qwen2.context_length" -> 32768 (UINT32 = 4)
        val keyCtx = "qwen2.context_length"
        buf.putLong(keyCtx.length.toLong())
        buf.put(keyCtx.toByteArray(Charsets.UTF_8))
        buf.putInt(4) // Type UINT32 = 4
        buf.putInt(32768)

        FileOutputStream(file).use { it.write(buf.array()) }

        val meta = GGUFMetadataParser.parse(file)
        assertTrue("Valid GGUF header should be accepted", meta.isValid)
        assertEquals("Architecture should match", "qwen2", meta.architecture)
        assertEquals("Model name should match", "Qwen2.5 1.5B Instruct", meta.modelName)
        assertEquals("Context length should match", 32768, meta.contextLength)
        assertEquals("Quantization should be derived from filename", "Q4_K_M", meta.quantization)
        assertEquals("Param count should be derived from filename", "1.5B", meta.parameterCount)
    }
}
