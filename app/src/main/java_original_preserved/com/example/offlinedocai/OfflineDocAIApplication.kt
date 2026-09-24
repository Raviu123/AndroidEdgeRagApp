package com.example.offlinedocai

import android.app.Application
import com.example.offlinedocai.ai.embeddings.EmbeddingModel
import com.example.offlinedocai.ai.llm.LlamaRunner
import com.example.offlinedocai.data.database.AppDatabase
import com.example.offlinedocai.data.repository.ChatRepositoryImpl
import com.example.offlinedocai.data.repository.DocumentRepositoryImpl
import com.example.offlinedocai.domain.rag.DocumentProcessor
import com.example.offlinedocai.domain.repository.ChatRepository
import com.example.offlinedocai.domain.repository.DocumentRepository

class OfflineDocAIApplication : Application() {
    lateinit var database: AppDatabase
        private set

    lateinit var documentRepository: DocumentRepository
        private set

    lateinit var chatRepository: ChatRepository
        private set

    lateinit var embeddingModel: EmbeddingModel
        private set

    lateinit var llamaRunner: LlamaRunner
        private set

    lateinit var bundledModelManager: com.example.offlinedocai.data.ai.BundledModelManager
        private set

    lateinit var documentProcessor: DocumentProcessor
        private set

    override fun onCreate() {
        super.onCreate()

        database = AppDatabase.getDatabase(this)
        documentRepository = DocumentRepositoryImpl(database.documentDao())
        chatRepository = ChatRepositoryImpl(database.chatDao())
        embeddingModel = EmbeddingModel(this)
        
        // Try to load user ONNX model from settings path if saved
        val onnxPath = com.example.offlinedocai.data.ai.SettingsManager(this).onnxModelPath
        if (onnxPath.isNotEmpty()) {
            embeddingModel.loadModel(onnxPath)
        }

        llamaRunner = LlamaRunner(this)
        documentProcessor = DocumentProcessor(this, documentRepository, embeddingModel)
        bundledModelManager = com.example.offlinedocai.data.ai.BundledModelManager(this)
    }
}
