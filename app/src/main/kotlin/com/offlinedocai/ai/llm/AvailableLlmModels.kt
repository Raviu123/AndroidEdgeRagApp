package com.offlinedocai.ai.llm

data class AvailableLlmModel(
    val id: String,
    val name: String,
    val parameterSize: String,
    val badge: String,
    val description: String,
    val ramRequirement: String,
    val fileName: String,
    val downloadUrl: String,
    /** Exact file size and SHA-256 published by Hugging Face; downloads are rejected if they differ. */
    val sizeBytes: Long,
    val sha256: String
) {
    /** Download size as Hugging Face shows it (decimal units), derived from the exact byte count. */
    val downloadSize: String get() = formatDownloadSize(sizeBytes)
}

fun formatDownloadSize(bytes: Long): String =
    if (bytes >= 1_000_000_000L) String.format(java.util.Locale.US, "%.1f GB", bytes / 1_000_000_000.0)
    else String.format(java.util.Locale.US, "%d MB", Math.round(bytes / 1_000_000.0))

object AvailableLlmModels {
    val list: List<AvailableLlmModel> = listOf(
        AvailableLlmModel(
            id = "qwen-0.5b",
            name = "Qwen 2.5 0.5B Instruct",
            parameterSize = "0.5B",
            badge = "Fastest / Low RAM",
            description = "Ultra-fast response with minimal memory usage. Ideal for entry-level devices.",
            ramRequirement = "Min 2 GB RAM",
            fileName = "qwen2.5-0.5b-instruct-q4_k_m.gguf",
            downloadUrl = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
            sizeBytes = 491400032L,
            sha256 = "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"
        ),
        AvailableLlmModel(
            id = "llama-1b",
            name = "Llama 3.2 1B Instruct",
            parameterSize = "1B",
            badge = "Standard Mobile",
            description = "Meta's lightweight model for general conversation and simple Q&A.",
            ramRequirement = "Min 2.5 GB RAM",
            fileName = "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            downloadUrl = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            sizeBytes = 807694464L,
            sha256 = "6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83"
        ),
        AvailableLlmModel(
            id = "qwen-1.5b",
            name = "Qwen 2.5 1.5B Instruct",
            parameterSize = "1.5B",
            badge = "High Quality",
            description = "Top-tier sub-2B model. Exceptional document comprehension, table extraction & reasoning.",
            ramRequirement = "Min 3 GB RAM",
            fileName = "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            downloadUrl = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
            sizeBytes = 1117320736L,
            sha256 = "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"
        ),
        AvailableLlmModel(
            id = "deepseek-r1-1.5b",
            name = "DeepSeek-R1 Distill 1.5B (Q6_K)",
            parameterSize = "1.5B",
            badge = "Advanced Reasoning",
            description = "Thinks step by step before answering (slower; uses a separate thinking budget). Best for reasoning questions.",
            ramRequirement = "Min 4 GB RAM",
            fileName = "DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf",
            downloadUrl = "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf",
            sizeBytes = 1464178784L,
            sha256 = "c9c8aff5a84e2e9c1fd6ea57af5ab52cf5d7272d34460403d82880075e611039"
        )
    )
}
