package com.example.offlinedocai.data.ai

data class ModelConfig(
    val embeddingAssetPath: String =
        "models/embedding/all-MiniLM-L6-v2.onnx",

    val embeddingVocabAssetPath: String =
        "models/embedding/vocab.txt",

    val llmAssetPath: String =
        "models/llm/qwen2.5-0.5b-instruct-q4_k_m.gguf"
)