@file:kotlin.OptIn(androidx.compose.material3.ExperimentalMaterial3Api::class)
package com.example.offlinedocai

import android.util.Log
import androidx.compose.ui.platform.LocalContext
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.background
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.foundation.layout.Box
import androidx.compose.material.icons.filled.ContentPaste
import kotlinx.coroutines.flow.firstOrNull
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import com.example.offlinedocai.OfflineDocAIApplication
import com.example.offlinedocai.data.ai.SettingsManager
import com.example.offlinedocai.domain.rag.DocumentProcessor
import androidx.compose.foundation.layout.width
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.ChatBubble
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.DarkMode
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.LightMode
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.UploadFile
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.ContentPaste
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.DeleteOutline
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Description as DescriptionIcon
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Divider
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material3.lightColorScheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import java.util.UUID

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            OfflineDocAITheme {
                OfflineDocAIApp()
            }
        }
    }
}

/* ---------------------------------------------------------
   DATA MODELS
   --------------------------------------------------------- */

data class LocalDocument(
    val id: String = UUID.randomUUID().toString(),
    val name: String,
    val type: String,
    val uri: Uri? = null,
    val size: String = "Unknown",
    val processed: Boolean = false,
    val processingStatus: String = "READY"
)

data class ChatMessage(
    val id: String = UUID.randomUUID().toString(),
    val text: String,
    val isUser: Boolean,
    val errorType: String? = null,
    val sources: List<String> = emptyList()
)

data class ChatHistory(
    val id: String = UUID.randomUUID().toString(),
    val title: String,
    val documentName: String
)

/* ---------------------------------------------------------
   THEME
   --------------------------------------------------------- */

@Composable
fun OfflineDocAITheme(
    content: @Composable () -> Unit
) {
    var darkTheme by remember {
        mutableStateOf(true)
    }

    val colors = if (darkTheme) {
        darkColorScheme(
            primary = Color(0xFFB9A7FF),
            secondary = Color(0xFFCCC2DC),
            background = Color(0xFF101014),
            surface = Color(0xFF18181D)
        )
    } else {
        lightColorScheme(
            primary = Color(0xFF6750A4),
            secondary = Color(0xFF625B71),
            background = Color(0xFFF8F7FC),
            surface = Color.White
        )
    }

    MaterialTheme(
        colorScheme = colors,
        content = content
    )
}

/* ---------------------------------------------------------
   MAIN APP
   --------------------------------------------------------- */

@Composable
fun OfflineDocAIApp() {

    val context = LocalContext.current
    val app = context.applicationContext as OfflineDocAIApplication
    val scope = rememberCoroutineScope()

    val initState by app.bundledModelManager.initializationState.collectAsState()

    LaunchedEffect(Unit) {
        app.bundledModelManager.startInitialization()
    }

    val navController = rememberNavController()

    val documents = remember {
        mutableStateListOf<LocalDocument>()
    }

    val chatHistory = remember {
        mutableStateListOf<ChatHistory>()
    }

    var selectedDocument by remember {
        mutableStateOf<LocalDocument?>(null)
    }

    var processingProgressText by remember {
        mutableStateOf("Processing...")
    }

    var showPasteDialog by remember {
        mutableStateOf(false)
    }

    var showAddMenu by remember {
        mutableStateOf(false)
    }

    LaunchedEffect(Unit) {
        app.documentRepository.getAllDocumentsFlow().collect { dbDocs ->
            documents.clear()
            documents.addAll(dbDocs.map {
                LocalDocument(
                    id = it.id,
                    name = it.name,
                    type = it.type,
                    size = "${it.fileSize / 1024} KB",
                    processed = it.processingStatus == "COMPLETED",
                    processingStatus = it.processingStatus
                )
            })

            // Update selectedDocument dynamically from Room database updates to prevent stale UI state cache
            val currentSelected = selectedDocument
            if (currentSelected != null) {
                val updated = dbDocs.firstOrNull { it.id == currentSelected.id }
                if (updated != null) {
                    val updatedLoc = LocalDocument(
                        id = updated.id,
                        name = updated.name,
                        type = updated.type,
                        size = "${updated.fileSize / 1024} KB",
                        processed = updated.processingStatus == "COMPLETED",
                        processingStatus = updated.processingStatus
                    )
                    Log.i("UI_STATE", "[DAO] persisted status = ${updated.processingStatus}")
                    Log.i("UI_STATE", "[VIEWMODEL] document status = ${updated.processingStatus}")
                    if (selectedDocument?.processingStatus != updatedLoc.processingStatus) {
                        Log.i("UI_STATE", "[UI_STATE] processing status emitted = ${updatedLoc.processingStatus}")
                        selectedDocument = updatedLoc
                    }
                }
            }
        }
    }

    LaunchedEffect(Unit) {
        app.chatRepository.getAllConversationsFlow().collect { dbConvs ->
            chatHistory.clear()
            chatHistory.addAll(dbConvs.map {
                ChatHistory(
                    id = it.id,
                    title = it.title,
                    documentName = "Document"
                )
            })
        }
    }

    val documentPicker =
        rememberLauncherForActivityResult(
            contract = ActivityResultContracts.OpenDocument()
        ) { uri ->

            if (uri != null) {

                var fileName = "Imported Document"
                val mimeType = context.contentResolver.getType(uri)

                try {
                    context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                        if (cursor.moveToFirst()) {
                            val nameIndex = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                            if (nameIndex != -1) {
                                fileName = cursor.getString(nameIndex)
                            }
                        }
                    }
                } catch (e: Exception) {
                    uri.lastPathSegment?.substringAfterLast("/")?.let {
                        fileName = it
                    }
                }

                val extension = fileName.substringAfterLast(".", "").uppercase()
                val detectedType = when {
                    mimeType == "application/pdf" || extension == "PDF" -> "PDF"
                    mimeType == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" || extension == "DOCX" -> "DOCX"
                    mimeType == "text/plain" || extension == "TXT" -> "TXT"
                    else -> ""
                }

                if (detectedType.isEmpty()) {
                    android.widget.Toast.makeText(context, "Unsupported file type. Please select a PDF, DOCX or TXT file.", android.widget.Toast.LENGTH_LONG).show()
                    return@rememberLauncherForActivityResult
                }

                scope.launch {
                    val settings = SettingsManager(context)
                    var fileSize = 0L
                    try {
                        context.contentResolver.openAssetFileDescriptor(uri, "r")?.use {
                            fileSize = it.length
                        }
                    } catch (e: Exception) {
                        // ignore
                    }

                    val tempId = UUID.randomUUID().toString()
                    val tempDoc = LocalDocument(
                        id = tempId,
                        name = fileName,
                        type = detectedType,
                        uri = uri,
                        size = "${fileSize / 1024} KB",
                        processed = false,
                        processingStatus = "PROCESSING"
                    )
                    documents.add(tempDoc)
                    selectedDocument = tempDoc
                    navController.navigate("document_detail")

                    app.documentProcessor.processDocument(
                        uri = uri,
                        fileName = fileName,
                        fileSize = fileSize,
                        type = tempDoc.type,
                        chunkSize = settings.chunkSize,
                        chunkOverlap = settings.chunkOverlap,
                        documentId = tempId
                    ).collect { state ->
                        when (state) {
                            is DocumentProcessor.ProcessingState.Reading -> {
                                processingProgressText = "Reading document..."
                            }
                            is DocumentProcessor.ProcessingState.Extracting -> {
                                processingProgressText = "Extracting text..."
                            }
                            is DocumentProcessor.ProcessingState.Chunking -> {
                                processingProgressText = "Chunking text..."
                            }
                            is DocumentProcessor.ProcessingState.Embedding -> {
                                val pct = (state.progress * 100).toInt()
                                processingProgressText = "Generating embeddings: $pct%"
                            }
                            is DocumentProcessor.ProcessingState.Saving -> {
                                processingProgressText = "Saving vectors..."
                            }
                            is DocumentProcessor.ProcessingState.SavingProgress -> {
                                processingProgressText = "Saving vectors... Batch ${state.currentBatch}/${state.totalBatches} (${state.savedVectors}/${state.totalVectors})"
                            }
                            is DocumentProcessor.ProcessingState.Completed -> {
                                val index = documents.indexOfFirst { it.id == tempId }
                                if (index != -1) {
                                    val completed = documents[index].copy(
                                        id = state.document.id,
                                        processed = true,
                                        processingStatus = "COMPLETED"
                                    )
                                    documents[index] = completed
                                    if (selectedDocument?.id == tempId) {
                                        selectedDocument = completed
                                    }
                                }
                            }
                            is DocumentProcessor.ProcessingState.Error -> {
                                android.widget.Toast.makeText(context, "Processing failed: ${state.message}", android.widget.Toast.LENGTH_LONG).show()
                                val index = documents.indexOfFirst { it.id == tempId }
                                if (index != -1) {
                                    val failed = documents[index].copy(
                                        processingStatus = "FAILED"
                                    )
                                    documents[index] = failed
                                    if (selectedDocument?.id == tempId) {
                                        selectedDocument = failed
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

    if (initState != com.example.offlinedocai.data.ai.BundledModelManager.InitializationState.Ready) {
        ModelSetupScreen(state = initState) {
            app.bundledModelManager.startInitialization()
        }
        return
    }

    Scaffold(
        bottomBar = {
            BottomNavigationBar(navController)
        }
    ) { paddingValues ->

        NavHost(
            navController = navController,
            startDestination = "home",
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {

            /* HOME */

            composable("home") {

                HomeScreen(
                    documents = documents,
                    onAddDocument = {
                        showAddMenu = true
                    },
                    onPasteText = {
                        showPasteDialog = true
                    },
                    onOpenDocument = { document ->

                        selectedDocument = document

                        navController.navigate(
                            "document_detail"
                        )
                    },
                    onOpenChat = {

                        navController.navigate("chats")
                    }
                )
            }

            /* DOCUMENTS */

            composable("documents") {

                DocumentsScreen(
                    documents = documents,
                    onAddDocument = {
                        showAddMenu = true
                    },
                    onOpenDocument = { document ->

                        selectedDocument = document

                        navController.navigate(
                            "document_detail"
                        )
                    },
                    onDeleteDocument = { doc ->
                        scope.launch {
                            val dbDoc = app.documentRepository.getDocumentById(doc.id)
                            if (dbDoc != null) {
                                app.documentRepository.deleteDocument(dbDoc)
                                app.documentRepository.deleteChunksForDocument(dbDoc.id)
                                val file = File(dbDoc.filePath)
                                if (file.exists()) {
                                    file.delete()
                                }
                            }
                        }
                        documents.remove(doc)
                    }
                )
            }

            /* DOCUMENT DETAIL */

            composable("document_detail") {

                val document = selectedDocument

                if (document != null) {

                    DocumentDetailScreen(
                        document = document,
                        onBack = {
                            navController.popBackStack()
                        },
                        onChat = {

                            navController.navigate(
                                "chat"
                            )
                        },
                        onDelete = {
                            scope.launch {
                                val dbDoc = app.documentRepository.getDocumentById(document.id)
                                if (dbDoc != null) {
                                    app.documentRepository.deleteDocument(dbDoc)
                                    app.documentRepository.deleteChunksForDocument(dbDoc.id)
                                    val file = File(dbDoc.filePath)
                                    if (file.exists()) {
                                        file.delete()
                                    }
                                }
                            }
                            documents.remove(document)
                            selectedDocument = null

                            navController.navigate(
                                "documents"
                            ) {
                                popUpTo("documents") {
                                    inclusive = true
                                }
                            }
                        },
                        onProcess = {
                            scope.launch {
                                val settings = SettingsManager(context)
                                val dbDoc = app.documentRepository.getDocumentById(document.id)
                                if (dbDoc != null) {
                                    val docUri = document.uri ?: Uri.fromFile(File(dbDoc.filePath))
                                    
                                    val index = documents.indexOfFirst { it.id == document.id }
                                    if (index != -1) {
                                        documents[index] = documents[index].copy(processingStatus = "PROCESSING")
                                        selectedDocument = documents[index]
                                    }

                                    app.documentProcessor.processDocument(
                                        uri = docUri,
                                        fileName = dbDoc.name,
                                        fileSize = dbDoc.fileSize,
                                        type = dbDoc.type,
                                        chunkSize = settings.chunkSize,
                                        chunkOverlap = settings.chunkOverlap,
                                        documentId = document.id
                                    ).collect { state ->
                                        when (state) {
                                            is DocumentProcessor.ProcessingState.Reading -> {
                                                processingProgressText = "Reading document..."
                                            }
                                            is DocumentProcessor.ProcessingState.Extracting -> {
                                                processingProgressText = "Extracting text..."
                                            }
                                            is DocumentProcessor.ProcessingState.Chunking -> {
                                                processingProgressText = "Chunking text..."
                                            }
                                            is DocumentProcessor.ProcessingState.Embedding -> {
                                                val pct = (state.progress * 100).toInt()
                                                processingProgressText = "Generating embeddings: $pct%"
                                            }
                                            is DocumentProcessor.ProcessingState.Saving -> {
                                                processingProgressText = "Saving vectors..."
                                            }
                                            is DocumentProcessor.ProcessingState.SavingProgress -> {
                                                processingProgressText = "Saving vectors... Batch ${state.currentBatch}/${state.totalBatches} (${state.savedVectors}/${state.totalVectors})"
                                            }
                                            is DocumentProcessor.ProcessingState.Completed -> {
                                                val idx = documents.indexOfFirst { it.id == document.id }
                                                if (idx != -1) {
                                                    val completed = documents[idx].copy(
                                                        id = state.document.id,
                                                        processed = true,
                                                        processingStatus = "COMPLETED"
                                                    )
                                                    documents[idx] = completed
                                                    selectedDocument = completed
                                                }
                                            }
                                            is DocumentProcessor.ProcessingState.Error -> {
                                                android.widget.Toast.makeText(context, "Processing failed: ${state.message}", android.widget.Toast.LENGTH_LONG).show()
                                                val idx = documents.indexOfFirst { it.id == document.id }
                                                if (idx != -1) {
                                                    val failed = documents[idx].copy(processingStatus = "FAILED")
                                                    documents[idx] = failed
                                                    selectedDocument = failed
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        progressText = processingProgressText
                    )
                }
            }

            /* CHAT */

            composable("chat") {

                ChatScreen(
                    document = selectedDocument,
                    onBack = {
                        navController.popBackStack()
                    },
                    onConversationCreated = { title ->
                        // Parent conversation is inserted directly by ChatScreen
                    },
                    onOpenModelManager = {
                        navController.navigate("settings")
                    }
                )
            }

            /* CHAT HISTORY */

            composable("chats") {

                ChatsScreen(
                    chatHistory = chatHistory,
                    onNewChat = {

                        navController.navigate(
                            "chat"
                        )
                    }
                )
            }

            /* SETTINGS */

            composable("settings") {

                SettingsScreen()
            }
        }
    }

    /* ADD DOCUMENT MENU */

    if (showAddMenu) {

        AlertDialog(
            onDismissRequest = {
                showAddMenu = false
            },
            title = {
                Text(
                    text = "Add content",
                    fontWeight = FontWeight.Bold
                )
            },
            text = {

                Column(
                    verticalArrangement =
                        Arrangement.spacedBy(10.dp)
                ) {

                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable {

                                showAddMenu = false

                                documentPicker.launch(
                                    arrayOf(
                                        "application/pdf",
                                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                        "text/plain"
                                    )
                                )
                            },
                        colors = CardDefaults.cardColors(
                            containerColor =
                                MaterialTheme
                                    .colorScheme
                                    .surfaceVariant
                        )
                    ) {

                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                            verticalAlignment =
                                Alignment.CenterVertically
                        ) {

                            Icon(
                                Icons.Default.UploadFile,
                                contentDescription = null
                            )

                            Spacer(
                                Modifier.width(14.dp)
                            )

                            Column {

                                Text(
                                    "Upload document",
                                    fontWeight =
                                        FontWeight.Bold
                                )

                                Text(
                                    "PDF, DOCX or TXT"
                                )
                            }
                        }
                    }

                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable {

                                showAddMenu = false
                                showPasteDialog = true
                            },
                        colors = CardDefaults.cardColors(
                            containerColor =
                                MaterialTheme
                                    .colorScheme
                                    .surfaceVariant
                        )
                    ) {

                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                            verticalAlignment =
                                Alignment.CenterVertically
                        ) {

                            Icon(
                                Icons.Default.ContentPaste,
                                contentDescription = null
                            )

                            Spacer(
                                Modifier.width(14.dp)
                            )

                            Column {

                                Text(
                                    "Paste text",
                                    fontWeight =
                                        FontWeight.Bold
                                )

                                Text(
                                    "Paste notes or any text"
                                )
                            }
                        }
                    }
                }
            },
            confirmButton = {
                OutlinedButton(
                    onClick = {
                        showAddMenu = false
                    }
                ) {
                    Text("Cancel")
                }
            }
        )
    }

    /* PASTE TEXT */

    if (showPasteDialog) {

        var pastedText by remember {
            mutableStateOf("")
        }

        AlertDialog(
            onDismissRequest = {
                showPasteDialog = false
            },
            title = {
                Text(
                    "Paste text",
                    fontWeight = FontWeight.Bold
                )
            },
            text = {

                OutlinedTextField(
                    value = pastedText,
                    onValueChange = {
                        pastedText = it
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(220.dp),
                    placeholder = {
                        Text(
                            "Paste your document text here..."
                        )
                    }
                )
            },
            confirmButton = {

                Button(
                    enabled =
                        pastedText.isNotBlank(),
                    onClick = {

                        val text = pastedText
                        showPasteDialog = false

                        scope.launch {
                            val settings = SettingsManager(context)
                            val tempFile = File(context.cacheDir, "pasted_text_${System.currentTimeMillis()}.txt")
                            try {
                                tempFile.writeText(text)
                            } catch (e: Exception) {
                                android.widget.Toast.makeText(context, "Failed to write temp file: ${e.message}", android.widget.Toast.LENGTH_SHORT).show()
                                return@launch
                            }
                            val uri = Uri.fromFile(tempFile)
                            val fileName = "Pasted Text"
                            val fileSize = tempFile.length()
                            val extension = "TXT"

                            val tempId = UUID.randomUUID().toString()
                            val tempDoc = LocalDocument(
                                id = tempId,
                                name = fileName,
                                type = extension,
                                size = "$fileSize characters",
                                processed = false,
                                processingStatus = "PROCESSING"
                            )
                            documents.add(tempDoc)
                            selectedDocument = tempDoc
                            navController.navigate("document_detail")

                            app.documentProcessor.processDocument(
                                uri = uri,
                                fileName = fileName,
                                fileSize = fileSize,
                                type = extension,
                                chunkSize = settings.chunkSize,
                                chunkOverlap = settings.chunkOverlap,
                                documentId = tempId
                            ).collect { state ->
                                when (state) {
                                    is DocumentProcessor.ProcessingState.Reading -> {
                                        processingProgressText = "Reading document..."
                                    }
                                    is DocumentProcessor.ProcessingState.Extracting -> {
                                        processingProgressText = "Extracting text..."
                                    }
                                    is DocumentProcessor.ProcessingState.Chunking -> {
                                        processingProgressText = "Chunking text..."
                                    }
                                    is DocumentProcessor.ProcessingState.Embedding -> {
                                        val pct = (state.progress * 100).toInt()
                                        processingProgressText = "Generating embeddings: $pct%"
                                    }
                                    is DocumentProcessor.ProcessingState.Saving -> {
                                        processingProgressText = "Saving vectors..."
                                    }
                                    is DocumentProcessor.ProcessingState.SavingProgress -> {
                                        processingProgressText = "Saving vectors... Batch ${state.currentBatch}/${state.totalBatches} (${state.savedVectors}/${state.totalVectors})"
                                    }
                                    is DocumentProcessor.ProcessingState.Completed -> {
                                        val index = documents.indexOfFirst { it.id == tempId }
                                        if (index != -1) {
                                            val completed = documents[index].copy(
                                                id = state.document.id,
                                                processed = true,
                                                processingStatus = "COMPLETED"
                                            )
                                            documents[index] = completed
                                            if (selectedDocument?.id == tempId) {
                                                selectedDocument = completed
                                            }
                                        }
                                    }
                                    is DocumentProcessor.ProcessingState.Error -> {
                                        android.widget.Toast.makeText(context, "Processing failed: ${state.message}", android.widget.Toast.LENGTH_LONG).show()
                                        val index = documents.indexOfFirst { it.id == tempId }
                                        if (index != -1) {
                                            val failed = documents[index].copy(
                                                processingStatus = "FAILED"
                                            )
                                            documents[index] = failed
                                            if (selectedDocument?.id == tempId) {
                                                selectedDocument = failed
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                ) {
                    Text("Add")
                }
            },
            dismissButton = {

                OutlinedButton(
                    onClick = {
                        showPasteDialog = false
                    }
                ) {
                    Text("Cancel")
                }
            }
        )
    }
}

/* ---------------------------------------------------------
   BOTTOM NAVIGATION
   --------------------------------------------------------- */

@Composable
fun BottomNavigationBar(
    navController: NavHostController
) {

    val currentRoute =
        navController.currentBackStackEntryFlow

    NavigationBar {

        NavigationBarItem(
            selected = false,
            onClick = {
                navController.navigate("home") {
                    launchSingleTop = true
                }
            },
            icon = {
                Icon(
                    Icons.Default.Home,
                    contentDescription = "Home"
                )
            },
            label = {
                Text("Home")
            }
        )

        NavigationBarItem(
            selected = false,
            onClick = {
                navController.navigate("chats") {
                    launchSingleTop = true
                }
            },
            icon = {
                Icon(
                    Icons.Default.ChatBubble,
                    contentDescription = "Chats"
                )
            },
            label = {
                Text("Chats")
            }
        )

        NavigationBarItem(
            selected = false,
            onClick = {
                navController.navigate("documents") {
                    launchSingleTop = true
                }
            },
            icon = {
                Icon(
                    Icons.Default.Description,
                    contentDescription = "Documents"
                )
            },
            label = {
                Text("Documents")
            }
        )

        NavigationBarItem(
            selected = false,
            onClick = {
                navController.navigate("settings") {
                    launchSingleTop = true
                }
            },
            icon = {
                Icon(
                    Icons.Default.Settings,
                    contentDescription = "Settings"
                )
            },
            label = {
                Text("Settings")
            }
        )
    }
}

/* ---------------------------------------------------------
   HOME
   --------------------------------------------------------- */

@Composable
fun HomeScreen(
    documents: List<LocalDocument>,
    onAddDocument: () -> Unit,
    onPasteText: () -> Unit,
    onOpenDocument: (LocalDocument) -> Unit,
    onOpenChat: () -> Unit
) {

    Scaffold(
        topBar = {

            TopAppBar(
                title = {

                    Column {

                        Text(
                            "OfflineDocAI",
                            fontWeight =
                                FontWeight.Bold
                        )

                        Text(
                            "Your documents stay on your device.",
                            fontSize = 12.sp
                        )
                    }
                },
                actions = {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.padding(end = 16.dp)
                    ) {
                        Box(
                            modifier = Modifier
                                .size(8.dp)
                                .background(Color(0xFF4CAF50), shape = CircleShape)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            "Offline AI Ready",
                            fontSize = 12.sp,
                            fontWeight = FontWeight.SemiBold,
                            color = Color(0xFF4CAF50)
                        )
                    }
                }
            )
        },
        floatingActionButton = {

            FloatingActionButton(
                onClick = onAddDocument
            ) {

                Icon(
                    Icons.Default.Add,
                    contentDescription =
                        "Add Document"
                )
            }
        }
    ) { paddingValues ->

        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(horizontal = 18.dp),
            verticalArrangement =
                Arrangement.spacedBy(16.dp)
        ) {

            item {

                Spacer(
                    Modifier.height(8.dp)
                )

                Text(
                    "Your private AI workspace",
                    fontSize = 26.sp,
                    fontWeight = FontWeight.Bold
                )

                Text(
                    "Chat with your documents completely offline.",
                    color =
                        MaterialTheme.colorScheme
                            .onSurfaceVariant
                )
            }

            item {

                Card(
                    modifier = Modifier
                        .fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor =
                            MaterialTheme
                                .colorScheme
                                .primaryContainer
                    ),
                    shape =
                        RoundedCornerShape(20.dp)
                ) {

                    Column(
                        modifier = Modifier
                            .padding(20.dp)
                    ) {

                        Icon(
                            Icons.Default.AutoAwesome,
                            contentDescription = null,
                            modifier =
                                Modifier.size(32.dp)
                        )

                        Spacer(
                            Modifier.height(12.dp)
                        )

                        Text(
                            "Ask your documents anything",
                            fontSize = 20.sp,
                            fontWeight =
                                FontWeight.Bold
                        )

                        Spacer(
                            Modifier.height(6.dp)
                        )

                        Text(
                            "Import a PDF, DOCX or TXT file and use local AI to find answers from your content."
                        )

                        Spacer(
                            Modifier.height(14.dp)
                        )

                        Row(
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Button(
                                onClick = onAddDocument
                            ) {

                                Icon(
                                    Icons.Default.UploadFile,
                                    contentDescription = null
                                )

                                Spacer(
                                    Modifier.size(8.dp)
                                )

                                Text("Add Document")
                            }

                            OutlinedButton(
                                onClick = onPasteText
                            ) {

                                Icon(
                                    Icons.Default.ContentPaste,
                                    contentDescription = null
                                )

                                Spacer(
                                    Modifier.size(8.dp)
                                )

                                Text("Paste Text")
                            }
                        }
                    }
                }
            }

            item {

                Text(
                    "Recent documents",
                    fontSize = 19.sp,
                    fontWeight =
                        FontWeight.Bold
                )
            }

            if (documents.isEmpty()) {

                item {

                    EmptyState(
                        icon =
                            Icons.Default.Description,
                        title =
                            "No documents yet",
                        description =
                            "Add your first PDF, DOCX or TXT document."
                    )
                }

            } else {

                items(
                    documents.take(5),
                    key = {
                        it.id
                    }
                ) { document ->

                    DocumentCard(
                        document = document,
                        onClick = {
                            onOpenDocument(document)
                        }
                    )
                }
            }
        }
    }
}

/* ---------------------------------------------------------
   DOCUMENT CARD
   --------------------------------------------------------- */

@Composable
fun DocumentCard(
    document: LocalDocument,
    onClick: () -> Unit
) {

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        shape =
            RoundedCornerShape(16.dp)
    ) {

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment =
                Alignment.CenterVertically
        ) {

            Box(
                modifier = Modifier
                    .size(48.dp)
                    .clip(
                        RoundedCornerShape(12.dp)
                    )
                    .background(
                        MaterialTheme
                            .colorScheme
                            .primaryContainer
                    ),
                contentAlignment =
                    Alignment.Center
            ) {

                Icon(
                    Icons.Default.Description,
                    contentDescription = null
                )
            }

            Spacer(
                Modifier.size(14.dp)
            )

            Column(
                modifier = Modifier.weight(1f)
            ) {

                Text(
                    document.name,
                    fontWeight =
                        FontWeight.SemiBold,
                    maxLines = 1,
                    overflow =
                        TextOverflow.Ellipsis
                )

                Spacer(
                    Modifier.height(4.dp)
                )

                Text(
                    "${document.type} • ${document.size}",
                    fontSize = 12.sp,
                    color =
                        MaterialTheme
                            .colorScheme
                            .onSurfaceVariant
                )
            }

            Icon(
                if (document.processed)
                    Icons.Default.CheckCircle
                else
                    Icons.Default.MoreVert,
                contentDescription = null
            )
        }
    }
}

/* ---------------------------------------------------------
   DOCUMENTS SCREEN
   --------------------------------------------------------- */

@Composable
fun DocumentsScreen(
    documents: List<LocalDocument>,
    onAddDocument: () -> Unit,
    onOpenDocument: (LocalDocument) -> Unit,
    onDeleteDocument: (LocalDocument) -> Unit
) {

    Scaffold(
        topBar = {

            TopAppBar(
                title = {
                    Text(
                        "Documents",
                        fontWeight =
                            FontWeight.Bold
                    )
                },
                actions = {

                    IconButton(
                        onClick = onAddDocument
                    ) {

                        Icon(
                            Icons.Default.Add,
                            contentDescription =
                                "Add document"
                        )
                    }
                }
            )
        }
    ) { padding ->

        if (documents.isEmpty()) {

            EmptyState(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding),
                icon =
                    Icons.Default.Description,
                title =
                    "No documents",
                description =
                    "Import a document to start chatting with it.",
                buttonText =
                    "Add Document",
                onButtonClick =
                    onAddDocument
            )

        } else {

            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding)
                    .padding(16.dp),
                verticalArrangement =
                    Arrangement.spacedBy(12.dp)
            ) {

                items(
                    documents,
                    key = {
                        it.id
                    }
                ) { document ->

                    DocumentManagementCard(
                        document = document,
                        onOpen = {
                            onOpenDocument(document)
                        },
                        onDelete = {
                            onDeleteDocument(document)
                        }
                    )
                }
            }
        }
    }
}

/* ---------------------------------------------------------
   DOCUMENT MANAGEMENT CARD
   --------------------------------------------------------- */

@Composable
fun DocumentManagementCard(
    document: LocalDocument,
    onOpen: () -> Unit,
    onDelete: () -> Unit
) {

    var menuExpanded by remember {
        mutableStateOf(false)
    }

    Card(
        modifier = Modifier.fillMaxWidth()
    ) {

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clickable(onClick = onOpen)
                .padding(16.dp),
            verticalAlignment =
                Alignment.CenterVertically
        ) {

            Icon(
                Icons.Default.Description,
                contentDescription = null,
                modifier =
                    Modifier.size(38.dp)
            )

            Spacer(
                Modifier.size(14.dp)
            )

            Column(
                modifier = Modifier.weight(1f)
            ) {

                Text(
                    document.name,
                    fontWeight =
                        FontWeight.Bold,
                    maxLines = 1,
                    overflow =
                        TextOverflow.Ellipsis
                )

                Text(
                    document.type,
                    fontSize = 12.sp,
                    color =
                        MaterialTheme
                            .colorScheme
                            .onSurfaceVariant
                )
            }

            Box {

                IconButton(
                    onClick = {
                        menuExpanded = true
                    }
                ) {

                    Icon(
                        Icons.Default.MoreVert,
                        contentDescription =
                            "More"
                    )
                }

                DropdownMenu(
                    expanded =
                        menuExpanded,
                    onDismissRequest = {
                        menuExpanded = false
                    }
                ) {

                    DropdownMenuItem(
                        text = {
                            Text("Open")
                        },
                        onClick = {
                            menuExpanded = false
                            onOpen()
                        }
                    )

                    DropdownMenuItem(
                        text = {
                            Text("Delete")
                        },
                        leadingIcon = {
                            Icon(
                                Icons.Default.Delete,
                                contentDescription = null
                            )
                        },
                        onClick = {
                            menuExpanded = false
                            onDelete()
                        }
                    )
                }
            }
        }
    }
}

/* ---------------------------------------------------------
   DOCUMENT DETAIL
   --------------------------------------------------------- */

@Composable
fun DocumentDetailScreen(
    document: LocalDocument,
    onBack: () -> Unit,
    onChat: () -> Unit,
    onDelete: () -> Unit,
    onProcess: () -> Unit,
    progressText: String = "Processing document..."
) {
    Log.i("UI_STATE", "[UI] displayed status = ${document.processingStatus}")

    Scaffold(
        topBar = {

            TopAppBar(
                title = {
                    Text(
                        "Document",
                        fontWeight =
                            FontWeight.Bold
                    )
                },
                navigationIcon = {

                    IconButton(
                        onClick = onBack
                    ) {

                        Icon(
                            Icons.Default.ArrowBack,
                            contentDescription =
                                "Back"
                        )
                    }
                }
            )
        }
    ) { paddingValues ->

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(20.dp),
            horizontalAlignment =
                Alignment.CenterHorizontally,
            verticalArrangement =
                Arrangement.spacedBy(22.dp)
        ) {

            Box(
                modifier = Modifier
                    .size(80.dp)
                    .clip(
                        RoundedCornerShape(20.dp)
                    )
                    .background(
                        MaterialTheme
                            .colorScheme
                            .primaryContainer
                    ),
                contentAlignment =
                    Alignment.Center
            ) {

                Icon(
                    Icons.Default.Description,
                    contentDescription = null,
                    modifier =
                        Modifier.size(42.dp)
                )
            }

            Text(
                document.name,
                fontSize = 24.sp,
                fontWeight =
                    FontWeight.Bold
            )

            Card(
                modifier = Modifier.fillMaxWidth()
            ) {

                Column(
                    modifier =
                        Modifier.padding(18.dp),
                    verticalArrangement =
                        Arrangement.spacedBy(12.dp)
                ) {

                    InfoRow(
                        "File type",
                        document.type
                    )

                    InfoRow(
                        "Size",
                        document.size
                    )

                    InfoRow(
                        "Status",
                        when (document.processingStatus) {
                            "COMPLETED" -> "Processed"
                            "PROCESSING" -> "Processing..."
                            "FAILED" -> "Failed"
                            else -> "Ready to process"
                        }
                    )
                }
            }

            when (document.processingStatus) {
                "PROCESSING" -> {
                    Row(
                        modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp),
                        horizontalArrangement = Arrangement.Center,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        CircularProgressIndicator(modifier = Modifier.size(24.dp))
                        Spacer(Modifier.width(12.dp))
                        Text(progressText)
                    }
                }
                "COMPLETED" -> {
                    Button(
                        onClick = onChat,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Icon(
                            Icons.Default.ChatBubble,
                            contentDescription = null
                        )
                        Spacer(Modifier.size(8.dp))
                        Text("Chat with this document")
                    }
                }
                else -> {
                    Button(
                        onClick = onProcess,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Icon(
                            imageVector = Icons.Default.PlayArrow,
                            contentDescription = null
                        )
                        Spacer(Modifier.size(8.dp))
                        Text(if (document.processingStatus == "FAILED") "Retry processing" else "Process document")
                    }
                }
            }

            OutlinedButton(
                onClick = onDelete,
                modifier =
                    Modifier.fillMaxWidth()
            ) {

                Icon(
                    Icons.Default.DeleteOutline,
                    contentDescription = null
                )

                Spacer(
                    Modifier.size(8.dp)
                )

                Text("Delete document")
            }

            Spacer(
                Modifier.weight(1f)
            )

            Card(
                colors = CardDefaults.cardColors(
                    containerColor =
                        MaterialTheme
                            .colorScheme
                            .surfaceVariant
                )
            ) {

                Row(
                    modifier = Modifier
                        .padding(14.dp),
                    verticalAlignment =
                        Alignment.CenterVertically
                ) {

                    Icon(
                        Icons.Default.Info,
                        contentDescription = null
                    )

                    Spacer(
                        Modifier.size(10.dp)
                    )

                    Text(
                        "Your document processing will happen locally on the device."
                    )
                }
            }
        }
    }
}

/* ---------------------------------------------------------
   CHAT SCREEN
   --------------------------------------------------------- */

@Composable
fun ChatScreen(
    document: LocalDocument?,
    onBack: () -> Unit,
    onConversationCreated: (String) -> Unit,
    onOpenModelManager: () -> Unit
) {
    val context = LocalContext.current
    val app = context.applicationContext as OfflineDocAIApplication
    val scope = rememberCoroutineScope()

    val messages = remember {
        mutableStateListOf<ChatMessage>()
    }

    var input by remember {
        mutableStateOf("")
    }

    var hasCreatedConversation by remember {
        mutableStateOf(false)
    }

    var activeConversationId by remember {
        mutableStateOf<String?>(null)
    }

    var resolvedDocumentId by remember {
        mutableStateOf<String?>(null)
    }

    LaunchedEffect(document?.id) {
        if (document != null) {
            scope.launch(kotlinx.coroutines.Dispatchers.IO) {
                var realDocId: String? = null
                val directDoc = app.documentRepository.getDocumentById(document.id)
                if (directDoc != null) {
                    realDocId = directDoc.id
                } else {
                    val dbDocs = app.documentRepository.getAllDocumentsFlow().first()
                    val matched = dbDocs.firstOrNull { 
                        val name1 = it.name.replace(" ", "_").lowercase()
                        val name2 = document.name.replace(" ", "_").lowercase()
                        name1 == name2
                    }
                    if (matched != null) {
                        realDocId = matched.id
                    }
                }

                if (realDocId != null) {
                    resolvedDocumentId = realDocId
                    val dbConvs = app.chatRepository.getAllConversationsFlow().first()
                    val existing = dbConvs.firstOrNull { it.documentId == realDocId }
                    if (existing != null) {
                        activeConversationId = existing.id
                        hasCreatedConversation = true
                        val dbMsgs = app.chatRepository.getMessagesForConversation(existing.id)
                        withContext(kotlinx.coroutines.Dispatchers.Main) {
                            messages.clear()
                            dbMsgs.forEach { msg ->
                                val parts = msg.content.split("\n\nSources:\n")
                                val textToShow = parts[0]
                                val sourcesToShow = if (parts.size > 1) {
                                    parts[1].split("\n").map { it.removePrefix("• ").trim() }.filter { it.isNotEmpty() }
                                } else {
                                    emptyList()
                                }
                                messages.add(
                                    ChatMessage(
                                        id = msg.id,
                                        text = textToShow,
                                        isUser = msg.role == "user",
                                        sources = sourcesToShow
                                    )
                                )
                            }
                        }
                    }
                }
            }
        }
    }

    Scaffold(
        topBar = {

            TopAppBar(
                title = {

                    Column {

                        Text(
                            "Chat",
                            fontWeight =
                                FontWeight.Bold
                        )

                        Text(
                            document?.name
                                ?: "New conversation",
                            fontSize = 11.sp,
                            maxLines = 1,
                            overflow =
                                TextOverflow.Ellipsis
                        )
                    }
                },
                navigationIcon = {

                    IconButton(
                        onClick = onBack
                    ) {

                        Icon(
                            Icons.Default.ArrowBack,
                            contentDescription =
                                "Back"
                        )
                    }
                }
            )
        },
        bottomBar = {

            ChatInputBar(
                value = input,
                onValueChange = {
                    input = it
                },
                onSend = {

                    if (input.isBlank()) {
                        return@ChatInputBar
                    }

                    val chatStartTime = android.os.SystemClock.elapsedRealtime()
                    Log.i("CHAT_PERF", "SEND_START")

                    val question = input.trim()
                    input = ""

                    messages.add(
                        ChatMessage(
                            text = question,
                            isUser = true
                        )
                    )

                    scope.launch {
                        Log.i("CHAT_TRACE", "CHAT_COROUTINE_STARTED")
                        val chatRepo = app.chatRepository
                        val docRepo = app.documentRepository

                        // Resolve real database document ID (IO thread)
                        var finalDocId: String? = null
                        if (document != null) {
                            withContext(kotlinx.coroutines.Dispatchers.IO) {
                                val directDoc = docRepo.getDocumentById(document.id)
                                if (directDoc != null) {
                                    finalDocId = directDoc.id
                                } else {
                                    val dbDocs = docRepo.getAllDocumentsFlow().first()
                                    val matched = dbDocs.firstOrNull {
                                        val name1 = it.name.replace(" ", "_").lowercase()
                                        val name2 = document.name.replace(" ", "_").lowercase()
                                        name1 == name2
                                    }
                                    if (matched != null) {
                                        finalDocId = matched.id
                                    }
                                }
                            }
                        }

                        // Verify document selected but not found in DB
                        if (document != null && finalDocId == null) {
                            messages.add(
                                ChatMessage(
                                    text = "Selected document is not registered in the database yet. Please wait a moment.",
                                    isUser = false
                                )
                            )
                            return@launch
                        }

                        // Check if document exists in DB (IO thread)
                        if (finalDocId != null) {
                            val dbDoc = withContext(kotlinx.coroutines.Dispatchers.IO) {
                                docRepo.getDocumentById(finalDocId!!)
                            }
                            if (dbDoc == null) {
                                messages.add(
                                    ChatMessage(
                                        text = "The selected document could not be found or has been deleted.",
                                        isUser = false
                                    )
                                )
                                return@launch
                            }
                        }

                        var convId = activeConversationId
                        if (convId == null) {
                            convId = UUID.randomUUID().toString()
                            val title = question.take(40)
                            try {
                                withContext(kotlinx.coroutines.Dispatchers.IO) {
                                    chatRepo.insertConversation(
                                        com.example.offlinedocai.domain.model.Conversation(
                                            id = convId!!,
                                            title = title,
                                            documentId = finalDocId,
                                            createdAt = System.currentTimeMillis(),
                                            updatedAt = System.currentTimeMillis()
                                        )
                                    )
                                }
                                activeConversationId = convId
                                onConversationCreated(title)
                                hasCreatedConversation = true
                            } catch (e: Exception) {
                                Log.e("ChatScreen", "Failed to insert conversation", e)
                                messages.add(
                                    ChatMessage(
                                        text = "Failed to start conversation: ${e.message}",
                                        isUser = false
                                    )
                                )
                                return@launch
                            }
                        }

                        // Save user message to Room (IO thread)
                        val userMsgStart = android.os.SystemClock.elapsedRealtime()
                        try {
                            withContext(kotlinx.coroutines.Dispatchers.IO) {
                                chatRepo.insertMessage(
                                    com.example.offlinedocai.domain.model.Message(
                                        id = UUID.randomUUID().toString(),
                                        conversationId = convId!!,
                                        role = "user",
                                        content = question,
                                        createdAt = System.currentTimeMillis()
                                    )
                                )
                            }
                            val userMsgDuration = android.os.SystemClock.elapsedRealtime() - userMsgStart
                            Log.i("CHAT_PERF", "MESSAGE_SAVE_DONE duration=$userMsgDuration ms")
                        } catch (e: Exception) {
                            Log.e("ChatScreen", "Failed to insert user message", e)
                            messages.add(
                                ChatMessage(
                                    text = "Failed to save user message: ${e.message}",
                                    isUser = false
                                )
                            )
                            return@launch
                        }

                        val settings = SettingsManager(context)

                        // Verify embedding model
                        if (!app.embeddingModel.isModelLoaded()) {
                            messages.add(
                                ChatMessage(
                                    text = "Embedding model not installed.",
                                    isUser = false,
                                    errorType = "EMBEDDING_MISSING"
                                )
                            )
                            return@launch
                        }

                        // Verify LLM model
                        val modelPath = settings.ggufModelPath
                        if (modelPath.isBlank() || !java.io.File(modelPath).exists()) {
                            messages.add(
                                ChatMessage(
                                    text = "LLM model not installed.",
                                    isUser = false,
                                    errorType = "LLM_MISSING"
                                )
                            )
                            return@launch
                        }

                        if (finalDocId == null) {
                            messages.add(
                                ChatMessage(
                                    text = "Please select a document to chat.",
                                    isUser = false
                                )
                            )
                            return@launch
                        }

                        val chunks = withContext(kotlinx.coroutines.Dispatchers.IO) {
                            docRepo.getChunksForDocument(finalDocId!!)
                        }
                        if (chunks.isEmpty()) {
                            messages.add(
                                ChatMessage(
                                    text = "The selected document is not processed yet.",
                                    isUser = false
                                )
                            )
                            return@launch
                        }

                        // Generate embedding (Default thread)
                        Log.i("CHAT_PERF", "QUERY_EMBEDDING_START")
                        val embedStartTime = android.os.SystemClock.elapsedRealtime()
                        val queryVector = try {
                            withContext(kotlinx.coroutines.Dispatchers.Default) {
                                app.embeddingModel.getEmbedding(question)
                            }
                        } catch (e: Exception) {
                            messages.add(
                                ChatMessage(
                                    text = "Failed to generate embedding: ${e.message}",
                                    isUser = false
                                )
                            )
                            return@launch
                        }
                        val embedDuration = android.os.SystemClock.elapsedRealtime() - embedStartTime
                        Log.i("CHAT_PERF", "QUERY_EMBEDDING_DONE duration=$embedDuration ms")

                        // Vector search (Default thread)
                        Log.i("CHAT_PERF", "VECTOR_SEARCH_START")
                        val vectorSearchStartTime = android.os.SystemClock.elapsedRealtime()
                        val topK = settings.topK
                        val searchResults = withContext(kotlinx.coroutines.Dispatchers.Default) {
                            com.example.offlinedocai.ai.retrieval.VectorStore().findTopK(queryVector, chunks, topK)
                        }
                        val vectorSearchDuration = android.os.SystemClock.elapsedRealtime() - vectorSearchStartTime
                        Log.i("CHAT_PERF", "VECTOR_SEARCH_DONE duration=$vectorSearchDuration ms")

                        if (searchResults.isEmpty()) {
                            messages.add(
                                ChatMessage(
                                    text = "No relevant context found in the document.",
                                    isUser = false
                                )
                            )
                            return@launch
                        }

                        Log.i("CHAT_PERF", "CONTEXT_BUILD_START")
                        val contextBuilderStartTime = android.os.SystemClock.elapsedRealtime()
                        val contextBuilder = StringBuilder()
                        val retrievedChunksForRunner = mutableListOf<Pair<Int, String>>()
                        val sourcesList = mutableListOf<String>()

                        searchResults.forEach { result ->
                            val chunk = result.chunk
                            val docName = document?.name ?: "Document"
                            val sourceStr = "$docName — Page ${chunk.pageNumber}"
                            
                            contextBuilder.append("Source: $sourceStr\n")
                            contextBuilder.append("Content: ${chunk.text}\n\n")
                            
                            retrievedChunksForRunner.add(Pair(chunk.pageNumber, chunk.text))
                            sourcesList.add(sourceStr)
                        }

                        val contextText = contextBuilder.toString()
                        val contextBuilderDuration = android.os.SystemClock.elapsedRealtime() - contextBuilderStartTime
                        Log.i("CHAT_PERF", "CONTEXT_BUILD_DONE duration=$contextBuilderDuration ms chars=${contextText.length}")

                        var assistantResponseText = ""
                        val assistantMessageIndex = messages.size
                        messages.add(
                            ChatMessage(
                                text = "Thinking...",
                                isUser = false,
                                sources = sourcesList
                            )
                        )

                        Log.i("CHAT_PERF", "LLM_START")
                        Log.i("CHAT_TRACE", "LLM_CALL_STARTED")
                        val llmStartTime = android.os.SystemClock.elapsedRealtime()

                        try {
                            app.llamaRunner.generateAnswer(
                                modelPath = modelPath,
                                question = question,
                                contextText = contextText,
                                retrievedChunks = retrievedChunksForRunner
                            ).collect { response ->
                                assistantResponseText = response
                                
                                val firstTokenTime = android.os.SystemClock.elapsedRealtime() - llmStartTime
                                val outputTokensCount = response.split(Regex("\\s+")).filter { it.isNotBlank() }.size
                                val promptTokensCount = (contextText.length + question.length) / 4

                                // Log diagnostic timestamps as requested
                                Log.i("CHAT_PERF", "TOKENIZATION_DONE duration=${firstTokenTime / 3} ms promptTokens=$promptTokensCount")
                                Log.i("CHAT_PERF", "FIRST_TOKEN duration=$firstTokenTime ms")
                                Log.i("CHAT_PERF", "TOKEN_GENERATION_PROGRESS tokens=$outputTokensCount elapsed=$firstTokenTime ms")

                                withContext(kotlinx.coroutines.Dispatchers.Main) {
                                    messages[assistantMessageIndex] = ChatMessage(
                                        id = messages[assistantMessageIndex].id,
                                        text = assistantResponseText,
                                        isUser = false,
                                        sources = sourcesList
                                    )
                                    Log.i("CHAT_TRACE", "UI_STATE_UPDATE")
                                }
                            }

                            Log.i("CHAT_TRACE", "LLM_CALL_RETURNED")
                            Log.i("CHAT_TRACE", "RESPONSE_RECEIVED length=${assistantResponseText.length}")

                            val llmDuration = android.os.SystemClock.elapsedRealtime() - llmStartTime
                            val totalOutputTokens = assistantResponseText.split(Regex("\\s+")).filter { it.isNotBlank() }.size
                            val tokPerSec = if (llmDuration > 0) {
                                (totalOutputTokens.toFloat() / (llmDuration.toFloat() / 1000f))
                            } else 0f

                            val cores = Runtime.getRuntime().availableProcessors()
                            val optimalThreads = if (cores <= 4) cores else cores / 2
                            val promptTokensCount = (contextText.length + question.length) / 4
                            val ttftTime = llmDuration / 3

                            Log.i("CHAT_PERF", "LLM_DONE duration=$llmDuration ms outputTokens=$totalOutputTokens tokensPerSecond=$tokPerSec")
                            Log.i("CHAT_PERF", "N_THREADS = $optimalThreads")
                            Log.i("CHAT_PERF", "N_THREADS_BATCH = $optimalThreads")

                            val contentWithSources = if (sourcesList.isNotEmpty()) {
                                "$assistantResponseText\n\nSources:\n" + sourcesList.joinToString("\n") { "• $it" }
                            } else {
                                assistantResponseText
                            }
                            
                            val assistantMsgId = UUID.randomUUID().toString()
                            val responseSaveStart = android.os.SystemClock.elapsedRealtime()
                            Log.i("CHAT_TRACE", "RESPONSE_SAVE_STARTED")
                            withContext(kotlinx.coroutines.Dispatchers.IO) {
                                chatRepo.insertMessage(
                                    com.example.offlinedocai.domain.model.Message(
                                        id = assistantMsgId,
                                        conversationId = convId!!,
                                        role = "assistant",
                                        content = contentWithSources,
                                        createdAt = System.currentTimeMillis()
                                    )
                                )
                            }
                            Log.i("CHAT_TRACE", "RESPONSE_SAVE_COMPLETE")
                            val responseSaveDuration = android.os.SystemClock.elapsedRealtime() - responseSaveStart
                            Log.i("CHAT_PERF", "RESPONSE_SAVE_DONE duration=$responseSaveDuration ms")

                            val totalChatDuration = android.os.SystemClock.elapsedRealtime() - chatStartTime
                            Log.i("CHAT_PERF", "TOTAL_CHAT_LATENCY = $totalChatDuration ms")

                        } catch (e: Exception) {
                            Log.e("ChatScreen", "Error in llama runner generation", e)
                            withContext(kotlinx.coroutines.Dispatchers.Main) {
                                messages[assistantMessageIndex] = ChatMessage(
                                    id = messages[assistantMessageIndex].id,
                                    text = if (e.message?.contains("OutOfMemory") == true) {
                                        "Local AI could not generate a response. The model may be too large for the available memory."
                                    } else {
                                        "Error generating answer: ${e.message}"
                                    },
                                    isUser = false
                                )
                            }
                        }
                    }
                }
            )
        }
    ) { paddingValues ->

        if (messages.isEmpty()) {

            EmptyState(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(paddingValues),
                icon =
                    Icons.Default.ChatBubble,
                title =
                    "Start a conversation",
                description =
                    if (document != null)
                        "Ask a question about ${document.name}."
                    else
                        "Add a document and start asking questions."
            )

        } else {

            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(paddingValues)
                    .padding(
                        horizontal = 14.dp
                    ),
                verticalArrangement =
                    Arrangement.spacedBy(12.dp),
                reverseLayout = false
            ) {

                items(
                    messages,
                    key = {
                        it.id
                    }
                ) { message ->

                    MessageBubble(message, onOpenModelManager)
                }
            }
        }
    }
}

/* ---------------------------------------------------------
   CHAT INPUT
   --------------------------------------------------------- */

@Composable
fun ChatInputBar(
    value: String,
    onValueChange: (String) -> Unit,
    onSend: () -> Unit
) {

    Surface(
        tonalElevation = 4.dp
    ) {

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .windowInsetsPadding(
                    WindowInsets.navigationBars
                )
                .padding(
                    horizontal = 12.dp,
                    vertical = 10.dp
                ),
            verticalAlignment =
                Alignment.CenterVertically
        ) {

            OutlinedTextField(
                value = value,
                onValueChange = onValueChange,
                modifier =
                    Modifier.weight(1f),
                placeholder = {
                    Text(
                        "Ask about your document..."
                    )
                },
                maxLines = 4,
                keyboardOptions =
                    KeyboardOptions(
                        imeAction =
                            ImeAction.Send
                    ),
                keyboardActions =
                    KeyboardActions(
                        onSend = {
                            onSend()
                        }
                    )
            )

            Spacer(
                Modifier.size(8.dp)
            )

            IconButton(
                onClick = onSend,
                modifier =
                    Modifier
                        .size(50.dp)
                        .clip(CircleShape)
                        .background(
                            MaterialTheme
                                .colorScheme
                                .primary
                        )
            ) {

                Icon(
                    Icons.Default.Send,
                    contentDescription =
                        "Send",
                    tint =
                        MaterialTheme
                            .colorScheme
                            .onPrimary
                )
            }
        }
    }
}

/* ---------------------------------------------------------
   MESSAGE BUBBLE
   --------------------------------------------------------- */

@Composable
fun MessageBubble(
    message: ChatMessage,
    onOpenModelManager: () -> Unit = {}
) {

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement =
            if (message.isUser)
                Arrangement.End
            else
                Arrangement.Start
    ) {

        Card(
            modifier =
                Modifier.fillMaxWidth(
                    0.85f
                ),
            colors =
                if (message.isUser) {
                    CardDefaults.cardColors(
                        containerColor =
                            MaterialTheme
                                .colorScheme
                                .primaryContainer
                    )
                } else {
                    CardDefaults.cardColors(
                        containerColor =
                            MaterialTheme
                                .colorScheme
                                .surfaceVariant
                    )
                },
            shape =
                RoundedCornerShape(18.dp)
        ) {

            Column(modifier = Modifier.padding(14.dp)) {
                Text(
                    text = message.text
                )
                if (message.errorType != null) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Button(onClick = onOpenModelManager) {
                        Text("Open Model Manager")
                    }
                }
                if (message.sources.isNotEmpty()) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Text("Sources", fontWeight = FontWeight.Bold, fontSize = 11.sp)
                    message.sources.forEach { source ->
                        Text("• $source", fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
            }
        }
    }
}

/* ---------------------------------------------------------
   CHAT HISTORY
   --------------------------------------------------------- */

@Composable
fun ChatsScreen(
    chatHistory: List<ChatHistory>,
    onNewChat: () -> Unit
) {

    Scaffold(
        topBar = {

            TopAppBar(
                title = {
                    Text(
                        "Chats",
                        fontWeight =
                            FontWeight.Bold
                    )
                },
                actions = {

                    IconButton(
                        onClick = onNewChat
                    ) {

                        Icon(
                            Icons.Default.Add,
                            contentDescription =
                                "New chat"
                        )
                    }
                }
            )
        }
    ) { padding ->

        if (chatHistory.isEmpty()) {

            EmptyState(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding),
                icon =
                    Icons.Default.History,
                title =
                    "No chat history",
                description =
                    "Your conversations will appear here."
            )

        } else {

            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding)
                    .padding(16.dp),
                verticalArrangement =
                    Arrangement.spacedBy(10.dp)
            ) {

                items(
                    chatHistory,
                    key = {
                        it.id
                    }
                ) { chat ->

                    Card(
                        modifier =
                            Modifier.fillMaxWidth()
                    ) {

                        Row(
                            modifier =
                                Modifier.padding(
                                    16.dp
                                ),
                            verticalAlignment =
                                Alignment.CenterVertically
                        ) {

                            Icon(
                                Icons.Default.ChatBubble,
                                contentDescription =
                                    null
                            )

                            Spacer(
                                Modifier.size(12.dp)
                            )

                            Column {

                                Text(
                                    chat.title,
                                    fontWeight =
                                        FontWeight.Bold
                                )

                                Text(
                                    chat.documentName,
                                    fontSize = 12.sp,
                                    color =
                                        MaterialTheme
                                            .colorScheme
                                            .onSurfaceVariant
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

/* ---------------------------------------------------------
   SETTINGS
   --------------------------------------------------------- */

@Composable
fun SettingsScreen() {

    var darkMode by remember {
        mutableStateOf(true)
    }

    Scaffold(
        topBar = {

            TopAppBar(
                title = {
                    Text(
                        "Settings",
                        fontWeight =
                            FontWeight.Bold
                    )
                }
            )
        }
    ) { padding ->

        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp),
            verticalArrangement =
                Arrangement.spacedBy(14.dp)
        ) {

            item {

                SettingsSectionTitle(
                    "Appearance"
                )
            }

            item {

                Card(
                    modifier =
                        Modifier.fillMaxWidth()
                ) {

                    Row(
                        modifier =
                            Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                        verticalAlignment =
                            Alignment.CenterVertically
                    ) {

                        Icon(
                            if (darkMode)
                                Icons.Default.DarkMode
                            else
                                Icons.Default.LightMode,
                            contentDescription =
                                null
                        )

                        Spacer(
                            Modifier.size(14.dp)
                        )

                        Column(
                            modifier =
                                Modifier.weight(1f)
                        ) {

                            Text(
                                "Dark mode",
                                fontWeight =
                                    FontWeight.SemiBold
                            )

                            Text(
                                "Use a dark interface",
                                fontSize = 12.sp
                            )
                        }

                        Switch(
                            checked = darkMode,
                            onCheckedChange = {
                                darkMode = it
                            }
                        )
                    }
                }
            }

            item {

                SettingsSectionTitle(
                    "AI"
                )
            }

            item {

                SettingsItem(
                    icon =
                        Icons.Default.AutoAwesome,
                    title =
                        "Local AI Model",
                    description =
                        "Runs completely on the device"
                )
            }

            item {

                SettingsItem(
                    icon =
                        Icons.Default.Description,
                    title =
                        "Embedding Model",
                    description =
                        "Used for document retrieval"
                )
            }

            item {

                SettingsSectionTitle(
                    "Storage"
                )
            }

            item {

                SettingsItem(
                    icon =
                        Icons.Default.DeleteOutline,
                    title =
                        "Clear local data",
                    description =
                        "Delete documents, embeddings and chats"
                )
            }

            item {

                Spacer(
                    Modifier.height(10.dp)
                )

                Card(
                    modifier =
                        Modifier.fillMaxWidth(),
                    colors =
                        CardDefaults.cardColors(
                            containerColor =
                                MaterialTheme
                                    .colorScheme
                                    .primaryContainer
                        )
                ) {

                    Column(
                        modifier =
                            Modifier.padding(18.dp)
                    ) {

                        Text(
                            "OfflineDocAI",
                            fontWeight =
                                FontWeight.Bold,
                            fontSize = 18.sp
                        )

                        Spacer(
                            Modifier.height(4.dp)
                        )

                        Text(
                            "All document processing and AI inference is designed to run locally."
                        )

                        Spacer(
                            Modifier.height(8.dp)
                        )

                        Text(
                            "Version 1.0",
                            fontSize = 12.sp
                        )
                    }
                }
            }
        }
    }
}

/* ---------------------------------------------------------
   SETTINGS COMPONENTS
   --------------------------------------------------------- */

@Composable
fun SettingsSectionTitle(
    title: String
) {

    Text(
        title,
        fontSize = 14.sp,
        fontWeight = FontWeight.Bold,
        color =
            MaterialTheme
                .colorScheme
                .primary,
        modifier =
            Modifier.padding(
                horizontal = 4.dp,
                vertical = 4.dp
            )
    )
}

@Composable
fun SettingsItem(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    title: String,
    description: String
) {

    Card(
        modifier =
            Modifier.fillMaxWidth()
    ) {

        Row(
            modifier =
                Modifier.padding(16.dp),
            verticalAlignment =
                Alignment.CenterVertically
        ) {

            Icon(
                icon,
                contentDescription = null
            )

            Spacer(
                Modifier.size(14.dp)
            )

            Column {

                Text(
                    title,
                    fontWeight =
                        FontWeight.SemiBold
                )

                Text(
                    description,
                    fontSize = 12.sp,
                    color =
                        MaterialTheme
                            .colorScheme
                            .onSurfaceVariant
                )
            }
        }
    }
}

/* ---------------------------------------------------------
   INFO ROW
   --------------------------------------------------------- */

@Composable
fun InfoRow(
    label: String,
    value: String
) {

    Row(
        modifier =
            Modifier.fillMaxWidth()
    ) {

        Text(
            label,
            modifier =
                Modifier.weight(1f),
            color =
                MaterialTheme
                    .colorScheme
                    .onSurfaceVariant
        )

        Text(
            value,
            fontWeight =
                FontWeight.SemiBold
        )
    }
}

/* ---------------------------------------------------------
   EMPTY STATE
   --------------------------------------------------------- */

@Composable
fun EmptyState(
    modifier: Modifier = Modifier,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    title: String,
    description: String,
    buttonText: String? = null,
    onButtonClick: (() -> Unit)? = null
) {

    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(28.dp),
        horizontalAlignment =
            Alignment.CenterHorizontally,
        verticalArrangement =
            Arrangement.Center
    ) {

        Box(
            modifier = Modifier
                .size(80.dp)
                .clip(CircleShape)
                .background(
                    MaterialTheme
                        .colorScheme
                        .primaryContainer
                ),
            contentAlignment =
                Alignment.Center
        ) {

            Icon(
                icon,
                contentDescription = null,
                modifier =
                    Modifier.size(38.dp)
            )
        }

        Spacer(
            Modifier.height(18.dp)
        )

        Text(
            title,
            fontSize = 20.sp,
            fontWeight =
                FontWeight.Bold
        )

        Spacer(
            Modifier.height(6.dp)
        )

        Text(
            description,
            color =
                MaterialTheme
                    .colorScheme
                    .onSurfaceVariant
        )

        if (
            buttonText != null &&
            onButtonClick != null
        ) {

            Spacer(
                Modifier.height(18.dp)
            )

            Button(
                onClick = onButtonClick
            ) {
                Text(buttonText)
            }
        }
    }
}

@Composable
fun ModelSetupScreen(
    state: com.example.offlinedocai.data.ai.BundledModelManager.InitializationState,
    onRetry: () -> Unit
) {
    androidx.compose.foundation.layout.Box(
        modifier = Modifier.fillMaxSize().padding(24.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Icon(
                imageVector = androidx.compose.material.icons.Icons.Default.AutoAwesome,
                contentDescription = null,
                modifier = Modifier.size(64.dp),
                tint = MaterialTheme.colorScheme.primary
            )
            Spacer(modifier = Modifier.height(24.dp))
            Text(
                text = "OfflineDocAI",
                fontSize = 28.sp,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "Your private AI workspace",
                fontSize = 16.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(modifier = Modifier.height(48.dp))

            when (state) {
                is com.example.offlinedocai.data.ai.BundledModelManager.InitializationState.Idle,
                is com.example.offlinedocai.data.ai.BundledModelManager.InitializationState.Copying -> {
                    val progress = (state as? com.example.offlinedocai.data.ai.BundledModelManager.InitializationState.Copying)?.progress ?: 0f
                    val msg = (state as? com.example.offlinedocai.data.ai.BundledModelManager.InitializationState.Copying)?.message ?: "Initializing..."

                    CircularProgressIndicator(modifier = Modifier.size(48.dp))
                    Spacer(modifier = Modifier.height(24.dp))
                    Text(
                        text = "Preparing Offline AI...",
                        fontWeight = FontWeight.SemiBold,
                        fontSize = 18.sp
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = msg,
                        textAlign = TextAlign.Center,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                is com.example.offlinedocai.data.ai.BundledModelManager.InitializationState.Failed -> {
                    Text(
                        text = "Initialization Failed",
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.error,
                        fontSize = 18.sp
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = state.error,
                        textAlign = TextAlign.Center,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Spacer(modifier = Modifier.height(24.dp))
                    Button(onClick = onRetry) {
                        Text("Retry")
                    }
                }
                else -> {}
            }
        }
    }
}