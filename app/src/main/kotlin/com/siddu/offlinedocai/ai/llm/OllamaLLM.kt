package com.siddu.offlinedocai.ai.llm

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL

/**
 * Development‑only implementation that talks to a local Ollama server.
 * It sends a JSON payload to the `/api/generate` endpoint and returns the generated text.
 * The request is performed on the IO dispatcher.
 */
class OllamaLLM : LocalLLM {
    private val TAG = "OllamaLLM"
    private val endpoint = "http://127.0.0.1:11434/api/generate"

    override suspend fun generate(prompt: String): String = withContext(Dispatchers.IO) {
        try {
            val url = URL(endpoint)
            val conn = url.openConnection() as HttpURLConnection
            conn.requestMethod = "POST"
            conn.doOutput = true
            conn.setRequestProperty("Content-Type", "application/json")

            val json = JSONObject().apply {
                put("model", "llama2") // model name must exist on the Ollama instance
                put("prompt", prompt)
                put("stream", false)
            }

            OutputStreamWriter(conn.outputStream).use { it.write(json.toString()) }

            val responseCode = conn.responseCode
            if (responseCode != HttpURLConnection.HTTP_OK) {
                throw IllegalStateException("Ollama returned HTTP $responseCode")
            }

            BufferedReader(InputStreamReader(conn.inputStream)).use { reader ->
                val resp = StringBuilder()
                var line: String?
                while (reader.readLine().also { line = it } != null) {
                    resp.append(line)
                }
                val respJson = JSONObject(resp.toString())
                // Ollama returns the generated text under the "response" key
                respJson.optString("response", "")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Ollama generation failed", e)
            throw e
        }
    }
}
