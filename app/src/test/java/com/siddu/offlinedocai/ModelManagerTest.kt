package com.siddu.offlinedocai

import android.content.Context
import android.content.ContextWrapper
import android.content.SharedPreferences
import com.siddu.offlinedocai.data.ai.ModelItem
import com.siddu.offlinedocai.data.ai.ModelManager
import com.siddu.offlinedocai.data.ai.ModelStatus
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class ModelManagerTest {

    @get:Rule
    val tempFolder = TemporaryFolder()

    private class InMemorySharedPreferences : SharedPreferences {
        private val data = mutableMapOf<String, Any?>()

        override fun getAll(): MutableMap<String, *> = HashMap(data)
        override fun getString(key: String?, defValue: String?): String? = data[key] as? String ?: defValue
        override fun getStringSet(key: String?, defValues: MutableSet<String>?): MutableSet<String>? = data[key] as? MutableSet<String> ?: defValues
        override fun getInt(key: String?, defValue: Int): Int = data[key] as? Int ?: defValue
        override fun getLong(key: String?, defValue: Long): Long = data[key] as? Long ?: defValue
        override fun getFloat(key: String?, defValue: Float): Float = data[key] as? Float ?: defValue
        override fun getBoolean(key: String?, defValue: Boolean): Boolean = data[key] as? Boolean ?: defValue
        override fun contains(key: String?): Boolean = data.containsKey(key)
        override fun edit(): SharedPreferences.Editor = EditorImpl(this)
        override fun registerOnSharedPreferenceChangeListener(listener: SharedPreferences.OnSharedPreferenceChangeListener?) {}
        override fun unregisterOnSharedPreferenceChangeListener(listener: SharedPreferences.OnSharedPreferenceChangeListener?) {}

        private class EditorImpl(private val prefs: InMemorySharedPreferences) : SharedPreferences.Editor {
            private val changes = mutableMapOf<String, Any?>()
            private var clear = false

            override fun putString(key: String?, value: String?): SharedPreferences.Editor { changes[key!!] = value; return this }
            override fun putStringSet(key: String?, values: MutableSet<String>?): SharedPreferences.Editor { changes[key!!] = values; return this }
            override fun putInt(key: String?, value: Int): SharedPreferences.Editor { changes[key!!] = value; return this }
            override fun putLong(key: String?, value: Long): SharedPreferences.Editor { changes[key!!] = value; return this }
            override fun putFloat(key: String?, value: Float): SharedPreferences.Editor { changes[key!!] = value; return this }
            override fun putBoolean(key: String?, value: Boolean): SharedPreferences.Editor { changes[key!!] = value; return this }
            override fun remove(key: String?): SharedPreferences.Editor { changes[key!!] = null; return this }
            override fun clear(): SharedPreferences.Editor { clear = true; return this }
            override fun commit(): Boolean { apply(); return true }
            override fun apply() {
                if (clear) prefs.data.clear()
                changes.forEach { (k, v) ->
                    if (v == null) prefs.data.remove(k) else prefs.data[k] = v
                }
            }
        }
    }

    private fun createTestContext(rootDir: File, prefs: SharedPreferences): Context {
        return object : ContextWrapper(null) {
            override fun getFilesDir(): File = rootDir
            override fun getApplicationContext(): Context = this
            override fun getSharedPreferences(name: String?, mode: Int): SharedPreferences = prefs
        }
    }

    private fun writeSyntheticGGUF(file: File) {
        val buf = ByteBuffer.allocate(2048).order(ByteOrder.LITTLE_ENDIAN)
        buf.put('G'.code.toByte())
        buf.put('G'.code.toByte())
        buf.put('U'.code.toByte())
        buf.put('F'.code.toByte())
        buf.putInt(3) // version
        buf.putLong(50L) // tensor count
        buf.putLong(1L) // kv count
        val key = "general.architecture"
        buf.putLong(key.length.toLong())
        buf.put(key.toByteArray(Charsets.UTF_8))
        buf.putInt(8) // STRING
        val value = "qwen2"
        buf.putLong(value.length.toLong())
        buf.put(value.toByteArray(Charsets.UTF_8))

        FileOutputStream(file).use { it.write(buf.array()) }
    }

    @Test
    fun testModelDiscovery_EmptyStorage() {
        val filesDir = tempFolder.newFolder("files_empty")
        val prefs = InMemorySharedPreferences()
        val context = createTestContext(filesDir, prefs)

        val modelManager = ModelManager(context)
        val models = modelManager.listInstalledModels()
        assertTrue("Initially no models should be installed", models.isEmpty())
        assertNull("No active model should exist initially", modelManager.getActiveModel())
    }

    @Test
    fun testModelDiscovery_FindsModelInAppDirectory() {
        val filesDir = tempFolder.newFolder("files_with_model")
        val modelDir = File(filesDir, "models/llm").apply { mkdirs() }
        val modelFile = File(modelDir, "qwen2.5-1.5b-instruct-q4_k_m.gguf")
        writeSyntheticGGUF(modelFile)

        val prefs = InMemorySharedPreferences()
        val context = createTestContext(filesDir, prefs)

        val modelManager = ModelManager(context)
        val models = modelManager.listInstalledModels()

        assertEquals("Should discover 1 model", 1, models.size)
        val discovered = models[0]
        assertEquals("Filename should match", "qwen2.5-1.5b-instruct-q4_k_m.gguf", discovered.filename)
        assertEquals("Status should be INSTALLED", ModelStatus.INSTALLED, discovered.status)
        assertTrue("Discovered target model should be marked active", discovered.isActive)
        assertEquals("Active model should be set in manager", discovered.id, modelManager.getActiveModel()?.id)
    }

    @Test
    fun testActiveModelSwitchingAndPersistence() {
        val filesDir = tempFolder.newFolder("files_multi_model")
        val modelDir = File(filesDir, "models/llm").apply { mkdirs() }
        val model1 = File(modelDir, "qwen2.5-1.5b-instruct-q4_k_m.gguf")
        val model2 = File(modelDir, "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
        writeSyntheticGGUF(model1)
        writeSyntheticGGUF(model2)

        val prefs = InMemorySharedPreferences()
        val context = createTestContext(filesDir, prefs)

        val manager1 = ModelManager(context)
        val list1 = manager1.listInstalledModels()
        assertEquals(2, list1.size)

        // Select model2 as active
        val switched = manager1.setActiveModel("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
        assertTrue("Switching model should succeed", switched)
        assertEquals("Llama-3.2-1B-Instruct-Q4_K_M.gguf", manager1.getActiveModel()?.filename)

        // Simulate app restart by creating a new ModelManager instance with same prefs and directory
        val manager2 = ModelManager(context)
        assertEquals("Active model selection must persist across restart",
            "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            manager2.getActiveModel()?.filename
        )
    }

    @Test
    fun testSafeModelDeletion_BlockedDuringActiveGeneration() {
        val filesDir = tempFolder.newFolder("files_deletion")
        val modelDir = File(filesDir, "models/llm").apply { mkdirs() }
        val modelFile = File(modelDir, "qwen2.5-1.5b-instruct-q4_k_m.gguf")
        writeSyntheticGGUF(modelFile)

        val prefs = InMemorySharedPreferences()
        val context = createTestContext(filesDir, prefs)

        val modelManager = ModelManager(context)
        assertEquals(1, modelManager.listInstalledModels().size)

        // Simulate active inference
        modelManager.setGenerating(true)
        assertTrue("Generation is active", modelManager.isGenerationActive())

        val deleteResult = modelManager.removeModel("qwen2.5-1.5b-instruct-q4_k_m.gguf")
        assertFalse("Cannot delete model while generation is active", deleteResult)
        assertTrue("Model file must still exist", modelFile.exists())

        // End inference and delete
        modelManager.setGenerating(false)
        val deleteSuccess = modelManager.removeModel("qwen2.5-1.5b-instruct-q4_k_m.gguf")
        assertTrue("Deletion should succeed when idle", deleteSuccess)
        assertFalse("Model file should be deleted", modelFile.exists())
        assertTrue("Models list should be empty", modelManager.listInstalledModels().isEmpty())
        assertNull("Active model should be cleared", modelManager.getActiveModel())
    }
}
