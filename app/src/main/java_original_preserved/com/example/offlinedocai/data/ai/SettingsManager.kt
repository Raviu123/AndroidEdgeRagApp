package com.example.offlinedocai.data.ai

import android.content.Context
import android.content.SharedPreferences

class SettingsManager(context: Context) {
    private val prefs: SharedPreferences = context.getSharedPreferences("offline_doc_ai_prefs", Context.MODE_PRIVATE)

    companion object {
        private const val KEY_THEME = "theme_mode" // SYSTEM, LIGHT, DARK
        private const val KEY_GGUF_PATH = "gguf_model_path"
        private const val KEY_ONNX_PATH = "onnx_model_path"
        private const val KEY_CHUNK_SIZE = "rag_chunk_size"
        private const val KEY_CHUNK_OVERLAP = "rag_chunk_overlap"
        private const val KEY_TOP_K = "rag_top_k"
        private const val KEY_TEMP = "gen_temp"
        private const val KEY_MAX_TOKENS = "gen_max_tokens"
    }

    var themeMode: String
        get() = prefs.getString(KEY_THEME, "SYSTEM") ?: "SYSTEM"
        set(value) = prefs.edit().putString(KEY_THEME, value).apply()

    var ggufModelPath: String
        get() = prefs.getString(KEY_GGUF_PATH, "") ?: ""
        set(value) = prefs.edit().putString(KEY_GGUF_PATH, value).apply()

    var onnxModelPath: String
        get() = prefs.getString(KEY_ONNX_PATH, "") ?: ""
        set(value) = prefs.edit().putString(KEY_ONNX_PATH, value).apply()

    var chunkSize: Int
        get() = prefs.getInt(KEY_CHUNK_SIZE, 300)
        set(value) = prefs.edit().putInt(KEY_CHUNK_SIZE, value).apply()

    var chunkOverlap: Int
        get() = prefs.getInt(KEY_CHUNK_OVERLAP, 50)
        set(value) = prefs.edit().putInt(KEY_CHUNK_OVERLAP, value).apply()

    var topK: Int
        get() = prefs.getInt(KEY_TOP_K, 3)
        set(value) = prefs.edit().putInt(KEY_TOP_K, value).apply()

    var temperature: Float
        get() = prefs.getFloat(KEY_TEMP, 0.7f)
        set(value) = prefs.edit().putFloat(KEY_TEMP, value).apply()

    var maxTokens: Int
        get() = prefs.getInt(KEY_MAX_TOKENS, 512)
        set(value) = prefs.edit().putInt(KEY_MAX_TOKENS, value).apply()
}
