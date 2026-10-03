package com.siddu.offlinedocai

import android.os.Bundle
import android.net.Uri
import android.view.MotionEvent
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Chat
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Settings
import com.siddu.offlinedocai.ai.llm.AnswerLength
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.NavHostController
import androidx.navigation.compose.*
import com.siddu.offlinedocai.data.entities.ConversationEntity
import com.siddu.offlinedocai.data.entities.DocumentEntity
import com.siddu.offlinedocai.data.entities.MessageEntity
import com.siddu.offlinedocai.domain.repository.DocumentRepository
import com.siddu.offlinedocai.domain.repository.ChatRepository
import com.siddu.offlinedocai.domain.repository.DownloadState
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        try {
            com.tom_roush.pdfbox.android.PDFBoxResourceLoader.init(applicationContext)
        } catch (_: Exception) {}
        setContent { OfflineDocAIApp() }
    }

    override fun dispatchGenericMotionEvent(event: MotionEvent?): Boolean {
        if (event?.actionMasked == MotionEvent.ACTION_HOVER_EXIT) {
            return true // Safely consume hover exit to prevent AndroidComposeView 1.6.x race condition crash
        }
        return super.dispatchGenericMotionEvent(event)
    }

    override fun dispatchTouchEvent(ev: MotionEvent?): Boolean {
        if (ev?.actionMasked == MotionEvent.ACTION_HOVER_EXIT) {
            return true
        }
        return super.dispatchTouchEvent(ev)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun OfflineDocAIApp(mainViewModel: MainViewModel = viewModel()) {
    val themeMode by mainViewModel.themeMode.collectAsState()
    val isDark = when (themeMode) {
        MainViewModel.ThemeMode.SYSTEM -> androidx.compose.foundation.isSystemInDarkTheme()
        MainViewModel.ThemeMode.DARK -> true
        MainViewModel.ThemeMode.LIGHT -> false
    }

    MaterialTheme(colorScheme = if (isDark) darkColorScheme() else lightColorScheme()) {
        val navController = rememberNavController()
        val items = listOf(
            NavItem("home", "Home", Icons.Default.Home),
            NavItem("chats", "Chats", Icons.Default.Chat),
            NavItem("documents", "Documents", Icons.Default.Description),
            NavItem("settings", "Settings", Icons.Default.Settings)
        )
        Scaffold(
            bottomBar = {
                NavigationBar {
                    val currentDestination = navController.currentBackStackEntryAsState().value?.destination
                    items.forEach { item ->
                        NavigationBarItem(
                            icon = { Icon(item.icon, contentDescription = item.label) },
                            label = { Text(item.label) },
                            selected = currentDestination?.hierarchy?.any { it.route == item.route } == true,
                            onClick = {
                                navController.navigate(item.route) {
                                    popUpTo(navController.graph.findStartDestination().id) { saveState = true }
                                    launchSingleTop = true
                                    restoreState = true
                                }
                            }
                        )
                    }
                }
            }
        ) { innerPadding ->
            NavHost(
                navController = navController,
                startDestination = "home",
                modifier = Modifier.padding(innerPadding)
            ) {
                composable("home") { HomeScreen(mainViewModel, navController) }
                composable("chats") { ChatsScreen(mainViewModel, navController) }
                composable("documents") { DocumentsScreen(mainViewModel, navController) }
                composable("settings") { SettingsScreen(mainViewModel) }
                composable("documentDetail/{docId}") { backStackEntry ->
                    val docId = backStackEntry.arguments?.getString("docId")?.toLongOrNull() ?: return@composable
                    DocumentDetailScreen(docId, mainViewModel, navController)
                }
                composable("chat/{convId}") { backStackEntry ->
                    val convId = backStackEntry.arguments?.getString("convId")?.toLongOrNull() ?: return@composable
                    ChatScreen(convId, mainViewModel, navController)
                }
                composable("newChat") { NewChatScreen(mainViewModel, navController) }
                composable("importProcessing") { ImportProcessingScreen(mainViewModel, navController) }
            }
        }
    }
}

data class NavItem(val route: String, val label: String, val icon: ImageVector)

@Composable
fun HomeScreen(viewModel: MainViewModel, navController: NavHostController) {
    val documents by viewModel.documents.collectAsState()
    val conversations by viewModel.conversations.collectAsState()
    val isEmbedDownloaded by viewModel.isEmbeddingModelDownloaded.collectAsState()
    val isLlmDownloaded by viewModel.isLlmModelDownloaded.collectAsState()

    // Trigger check on layout load
    LaunchedEffect(Unit) {
        viewModel.checkModels()
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.Top,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(modifier = Modifier.height(24.dp))
        Text("Offline DocAI", style = MaterialTheme.typography.headlineLarge, fontWeight = FontWeight.Bold)
        Spacer(modifier = Modifier.height(4.dp))
        Text("Your secure, fully local RAG assistant", style = MaterialTheme.typography.bodyMedium, color = Color.Gray)
        Spacer(modifier = Modifier.height(24.dp))

        // Check if models need to be downloaded
        if (!isEmbedDownloaded || !isLlmDownloaded) {
            ModelDownloadBanner(viewModel)
            Spacer(modifier = Modifier.height(16.dp))
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Card(
                modifier = Modifier
                    .weight(1f)
                    .clickable { navController.navigate("documents") }
            ) {
                Column(modifier = Modifier.padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("${documents.size}", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                    Text("Documents", style = MaterialTheme.typography.bodyMedium)
                }
            }
            Card(
                modifier = Modifier
                    .weight(1f)
                    .clickable { navController.navigate("chats") }
            ) {
                Column(modifier = Modifier.padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("${conversations.size}", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                    Text("Chats", style = MaterialTheme.typography.bodyMedium)
                }
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        Button(
            onClick = { navController.navigate("newChat") },
            modifier = Modifier.fillMaxWidth(),
            enabled = isLlmDownloaded && isEmbedDownloaded
        ) {
            Icon(Icons.Default.Chat, contentDescription = null)
            Spacer(modifier = Modifier.width(8.dp))
            Text("Start New Conversation")
        }
    }
}

@Composable
fun ModelDownloadBanner(viewModel: MainViewModel) {
    val embedDownloadState by viewModel.embeddingDownloadState.collectAsState()
    val llmDownloadState by viewModel.llmDownloadState.collectAsState()
    val isEmbedDownloaded by viewModel.isEmbeddingModelDownloaded.collectAsState()
    val isLlmDownloaded by viewModel.isLlmModelDownloaded.collectAsState()

    Card(
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.8f)
        ),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Action Required: Models Missing", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Spacer(modifier = Modifier.height(8.dp))

            if (!isEmbedDownloaded) {
                Text("• Embedding Model (MiniLM):", fontWeight = FontWeight.Bold)
                Spacer(modifier = Modifier.height(4.dp))
                when (embedDownloadState) {
                    is DownloadState.Idle -> {
                        Text("Waiting to start...")
                    }
                    is DownloadState.Progress -> {
                        val progress = embedDownloadState as DownloadState.Progress
                        Column {
                            LinearProgressIndicator(
                                progress = { progress.percent / 100f },
                                modifier = Modifier.fillMaxWidth()
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            Text("Downloading: ${progress.percent}% (${(progress.bytesDownloaded / 1024 / 1024)}MB / ${(progress.totalBytes / 1024 / 1024)}MB)", style = MaterialTheme.typography.bodySmall)
                        }
                    }
                    is DownloadState.Completed -> {
                        Text("Finished!", color = Color(0xFF4CAF50))
                    }
                    is DownloadState.Error -> {
                        val error = embedDownloadState as DownloadState.Error
                        Text("Error: ${error.message}", color = Color.Red, style = MaterialTheme.typography.bodySmall)
                        Spacer(modifier = Modifier.height(4.dp))
                        Button(onClick = { viewModel.downloadEmbeddingModel() }) {
                            Text("Retry Embedding Download")
                        }
                    }
                }
                Spacer(modifier = Modifier.height(12.dp))
            }

            if (!isLlmDownloaded) {
                Text("• Local LLM (SLM): Required for chatting", fontWeight = FontWeight.Bold)
                Spacer(modifier = Modifier.height(4.dp))

                when (llmDownloadState) {
                    is DownloadState.Idle -> {
                        Text("Select a model to download below:")
                        Spacer(modifier = Modifier.height(8.dp))
                        Button(
                            onClick = {
                                viewModel.downloadLlmModel(
                                    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
                                    "qwen2.5-0.5b-instruct-q4_k_m.gguf"
                                )
                            },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("Download Qwen 2.5 0.5B (390 MB)")
                        }
                        Spacer(modifier = Modifier.height(8.dp))
                        Button(
                            onClick = {
                                viewModel.downloadLlmModel(
                                    "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                                    "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
                                )
                            },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("Download Llama 3.2 1B (700 MB)")
                        }
                    }
                    is DownloadState.Progress -> {
                        val progress = llmDownloadState as DownloadState.Progress
                        Column {
                            LinearProgressIndicator(
                                progress = { progress.percent / 100f },
                                modifier = Modifier.fillMaxWidth()
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            Text("Downloading: ${progress.percent}% (${(progress.bytesDownloaded / 1024 / 1024)}MB / ${(progress.totalBytes / 1024 / 1024)}MB)", style = MaterialTheme.typography.bodySmall)
                        }
                    }
                    is DownloadState.Completed -> {
                        Text("Download finished! Initializing...", style = MaterialTheme.typography.bodyMedium, color = Color(0xFF4CAF50))
                        Button(onClick = { viewModel.resetLlmDownloadState() }) {
                            Text("OK")
                        }
                    }
                    is DownloadState.Error -> {
                        val error = llmDownloadState as DownloadState.Error
                        Text("Error: ${error.message}", color = Color.Red, style = MaterialTheme.typography.bodySmall)
                        Spacer(modifier = Modifier.height(8.dp))
                        Button(onClick = { viewModel.resetLlmDownloadState() }) {
                            Text("Retry")
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun DocumentsScreen(viewModel: MainViewModel, navController: NavHostController) {
    val documents by viewModel.documents.collectAsState()
    val isEmbedDownloaded by viewModel.isEmbeddingModelDownloaded.collectAsState()
    val filePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            viewModel.importDocument(it)
            navController.navigate("importProcessing")
        }
    }

    Scaffold(
        floatingActionButton = {
            if (isEmbedDownloaded) {
                FloatingActionButton(onClick = { filePicker.launch("*/*") }) {
                    Icon(Icons.Default.Add, contentDescription = "Import Document")
                }
            }
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
        ) {
            Text("Documents", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
            Spacer(modifier = Modifier.height(16.dp))

            if (!isEmbedDownloaded) {
                Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)) {
                    Text("Please wait for the embedding model to finish downloading before importing documents.", modifier = Modifier.padding(16.dp))
                }
            } else if (documents.isEmpty()) {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Text("No documents imported yet.", color = Color.Gray)
                }
            } else {
                LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(documents) { doc ->
                        Card(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable { navController.navigate("documentDetail/${doc.id}") }
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(16.dp),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Column(modifier = Modifier.weight(1f)) {
                                    Text(doc.name, fontWeight = FontWeight.SemiBold)
                                    Spacer(modifier = Modifier.height(4.dp))
                                    val isReady = doc.status == "COMPLETED"
                                    val statusColor = if (isReady) Color(0xFF4CAF50) else Color(0xFFFF9800)
                                    Text(
                                        text = "${doc.pageCount} pages • ${if (isReady) "Ready for RAG" else "Processing..."}",
                                        style = MaterialTheme.typography.bodySmall,
                                        color = statusColor
                                    )
                                }
                                IconButton(onClick = { viewModel.deleteDocument(doc.id) }) {
                                    Icon(Icons.Default.Delete, contentDescription = "Delete", tint = Color.Red)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DocumentDetailScreen(docId: Long, viewModel: MainViewModel, navController: NavHostController) {
    val documents by viewModel.documents.collectAsState()
    val doc = documents.find { it.id == docId }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(doc?.name ?: "Document Detail") },
                navigationIcon = {
                    IconButton(onClick = { navController.popBackStack() }) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
                }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
        ) {
            if (doc == null) {
                Text("Document not found.")
            } else {
                Text("File Name: ${doc.name}", style = MaterialTheme.typography.titleLarge)
                Spacer(modifier = Modifier.height(8.dp))
                Text("Status: ${doc.status}", style = MaterialTheme.typography.bodyMedium)
                Spacer(modifier = Modifier.height(8.dp))
                Text("Pages: ${doc.pageCount}", style = MaterialTheme.typography.bodyMedium)
                Spacer(modifier = Modifier.height(8.dp))
                Text("URI: ${doc.uri}", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
            }
        }
    }
}

@Composable
fun ImportProcessingScreen(viewModel: MainViewModel, navController: NavHostController) {
    val importState by viewModel.importState.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text("Importing Document", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
        Spacer(modifier = Modifier.height(32.dp))

        when (importState) {
            is DocumentRepository.ImportState.Idle -> {
                CircularProgressIndicator()
                Spacer(modifier = Modifier.height(16.dp))
                Text("Initializing...")
            }
            is DocumentRepository.ImportState.Reading -> {
                CircularProgressIndicator()
                Spacer(modifier = Modifier.height(16.dp))
                Text("Reading text content...")
            }
            is DocumentRepository.ImportState.Chunking -> {
                CircularProgressIndicator()
                Spacer(modifier = Modifier.height(16.dp))
                Text("Splitting document pages into semantic chunks...")
            }
            is DocumentRepository.ImportState.Embedding -> {
                CircularProgressIndicator()
                Spacer(modifier = Modifier.height(16.dp))
                Text("Generating vector embeddings locally...")
            }
            is DocumentRepository.ImportState.Saving -> {
                CircularProgressIndicator()
                Spacer(modifier = Modifier.height(16.dp))
                Text("Saving vectors to local SQLite store...")
            }
            is DocumentRepository.ImportState.Completed -> {
                Text("🎉 Import Completed Successfully!", style = MaterialTheme.typography.titleMedium, color = Color(0xFF4CAF50))
                Spacer(modifier = Modifier.height(32.dp))
                Button(onClick = { navController.navigate("documents") { popUpTo("home") } }) {
                    Text("Go to Documents")
                }
            }
            is DocumentRepository.ImportState.Error -> {
                val errMsg = (importState as DocumentRepository.ImportState.Error).message
                Text("❌ Import Failed", style = MaterialTheme.typography.titleMedium, color = Color.Red)
                Spacer(modifier = Modifier.height(8.dp))
                Text(errMsg, color = Color.Gray)
                Spacer(modifier = Modifier.height(32.dp))
                Button(onClick = { navController.popBackStack() }) {
                    Text("Go Back")
                }
            }
        }
    }
}

@Composable
fun ChatsScreen(viewModel: MainViewModel, navController: NavHostController) {
    val conversations by viewModel.conversations.collectAsState()
    val isLlmDownloaded by viewModel.isLlmModelDownloaded.collectAsState()
    val isEmbedDownloaded by viewModel.isEmbeddingModelDownloaded.collectAsState()

    Scaffold(
        floatingActionButton = {
            if (isLlmDownloaded && isEmbedDownloaded) {
                FloatingActionButton(onClick = { navController.navigate("newChat") }) {
                    Icon(Icons.Default.Add, contentDescription = "New Chat")
                }
            }
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
        ) {
            Text("Conversations", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
            Spacer(modifier = Modifier.height(16.dp))

            if (!isLlmDownloaded || !isEmbedDownloaded) {
                Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)) {
                    Text("Please download missing models on the Home screen to start conversations.", modifier = Modifier.padding(16.dp))
                }
            } else if (conversations.isEmpty()) {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Text("No chats active yet.", color = Color.Gray)
                }
            } else {
                LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(conversations) { chat ->
                        Card(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable { navController.navigate("chat/${chat.id}") }
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(16.dp),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Row(
                                    modifier = Modifier.weight(1f),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Icon(Icons.Default.Chat, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                                    Spacer(modifier = Modifier.width(16.dp))
                                    Column {
                                        Text(chat.title, fontWeight = FontWeight.SemiBold)
                                        Text("Created at: ${java.text.SimpleDateFormat("yyyy-MM-dd HH:mm").format(java.util.Date(chat.createdAt))}", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
                                    }
                                }
                                IconButton(onClick = { viewModel.deleteConversation(chat.id) }) {
                                    Icon(Icons.Default.Delete, contentDescription = "Delete Conversation", tint = Color.Red)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun NewChatScreen(viewModel: MainViewModel, navController: NavHostController) {
    val documents by viewModel.documents.collectAsState()
    var title by remember { mutableStateOf("") }
    val selectedDocIds = remember { mutableStateListOf<Long>() }
    val coroutineScope = rememberCoroutineScope()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("New Conversation") },
                navigationIcon = {
                    IconButton(onClick = { navController.popBackStack() }) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
                }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
        ) {
            OutlinedTextField(
                value = title,
                onValueChange = { title = it },
                label = { Text("Chat Title") },
                placeholder = { Text("e.g. Finance Report Q2") },
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(24.dp))
            Text("Select Documents to Reference:", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Spacer(modifier = Modifier.height(8.dp))

            if (documents.isEmpty()) {
                Text("No documents available. Import some first.", color = Color.Gray)
            } else {
                LazyColumn(modifier = Modifier.weight(1f)) {
                    items(documents) { doc ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable {
                                    if (selectedDocIds.contains(doc.id)) {
                                        selectedDocIds.remove(doc.id)
                                    } else {
                                        selectedDocIds.add(doc.id)
                                    }
                                }
                                .padding(vertical = 4.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Checkbox(
                                checked = selectedDocIds.contains(doc.id),
                                onCheckedChange = {
                                    if (it == true) selectedDocIds.add(doc.id) else selectedDocIds.remove(doc.id)
                                }
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(doc.name)
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            Button(
                onClick = {
                    val finalTitle = title.ifBlank { "New Chat" }
                    coroutineScope.launch {
                        val convId = viewModel.startConversation(finalTitle, selectedDocIds.toList())
                        navController.navigate("chat/$convId") {
                            popUpTo("chats") { inclusive = false }
                        }
                    }
                },
                modifier = Modifier.fillMaxWidth(),
                enabled = title.isNotBlank()
            ) {
                Text("Create Conversation")
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(convId: Long, viewModel: MainViewModel, navController: NavHostController) {
    val conversations by viewModel.conversations.collectAsState()
    val chat = conversations.find { it.id == convId }
    val messages by viewModel.messagesForConversation(convId).collectAsState()
    val chatState by viewModel.chatState.collectAsState()
    var textInput by remember { mutableStateOf("") }
    val selectedAnswerLength by viewModel.selectedAnswerLength.collectAsState()
    var lengthMenuExpanded by remember { mutableStateOf(false) }

    val linkedDocIds = chat?.documentId?.let { listOf(it) } ?: emptyList()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(chat?.title ?: "Chat") },
                navigationIcon = {
                    IconButton(onClick = { navController.popBackStack() }) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
                }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
        ) {
            LazyColumn(
                modifier = Modifier
                    .weight(1f)
                    .padding(horizontal = 16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(messages) { msg ->
                    val isUser = msg.role == "user"
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 4.dp),
                        horizontalAlignment = if (isUser) Alignment.End else Alignment.Start
                    ) {
                        Card(
                            colors = CardDefaults.cardColors(
                                containerColor = if (isUser) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant
                            ),
                            modifier = Modifier.fillMaxWidth(0.85f)
                        ) {
                            Column(modifier = Modifier.padding(12.dp)) {
                                Text(msg.content)
                                if (!msg.sources.isNullOrBlank()) {
                                    val chunksList = remember(msg.sources) {
                                        msg.sources.split("|||").filter { it.isNotBlank() }
                                    }
                                    if (chunksList.isNotEmpty()) {
                                        var expanded by remember { mutableStateOf(false) }
                                        Spacer(modifier = Modifier.height(8.dp))
                                        Divider(color = Color.LightGray.copy(alpha = 0.5f))
                                        Spacer(modifier = Modifier.height(4.dp))
                                        Row(
                                            modifier = Modifier
                                                .fillMaxWidth()
                                                .clickable { expanded = !expanded }
                                                .padding(vertical = 4.dp),
                                            verticalAlignment = Alignment.CenterVertically,
                                            horizontalArrangement = Arrangement.SpaceBetween
                                        ) {
                                            Text(
                                                text = if (expanded) "▼ Hide Retained Chunks (${chunksList.size})" else "▶ View Extracted Context Chunks (${chunksList.size})",
                                                style = MaterialTheme.typography.labelMedium,
                                                color = MaterialTheme.colorScheme.primary,
                                                fontWeight = FontWeight.Bold
                                            )
                                        }
                                        if (expanded) {
                                            Column(
                                                modifier = Modifier
                                                    .fillMaxWidth()
                                                    .padding(top = 4.dp),
                                                verticalArrangement = Arrangement.spacedBy(6.dp)
                                            ) {
                                                chunksList.forEachIndexed { idx, chunk ->
                                                    Surface(
                                                        shape = RoundedCornerShape(6.dp),
                                                        color = MaterialTheme.colorScheme.background,
                                                        modifier = Modifier.fillMaxWidth()
                                                    ) {
                                                        Column(modifier = Modifier.padding(8.dp)) {
                                                            Text(
                                                                text = "Chunk #${idx + 1}",
                                                                style = MaterialTheme.typography.labelSmall,
                                                                fontWeight = FontWeight.Bold,
                                                                color = MaterialTheme.colorScheme.secondary
                                                            )
                                                            Spacer(modifier = Modifier.height(2.dp))
                                                            Text(
                                                                text = chunk,
                                                                style = MaterialTheme.typography.bodySmall
                                                            )
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                when (val state = chatState) {
                    is ChatRepository.ChatState.Retrieving -> {
                        item {
                            Row(
                                modifier = Modifier.fillMaxWidth().padding(8.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                                Spacer(modifier = Modifier.width(8.dp))
                                Text("Searching document and retrieving context...", color = Color.Gray, style = MaterialTheme.typography.bodyMedium)
                            }
                        }
                    }
                    is ChatRepository.ChatState.Generating -> {
                        item {
                            Row(
                                modifier = Modifier.fillMaxWidth().padding(8.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(
                                    if (state.elapsedSeconds > 0) "Generating answer (${state.elapsedSeconds}s)..." else "Generating answer...",
                                    color = Color.Gray,
                                    style = MaterialTheme.typography.bodyMedium
                                )
                            }
                        }
                    }
                    is ChatRepository.ChatState.Thinking -> {
                        item {
                            Row(
                                modifier = Modifier.fillMaxWidth().padding(8.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                                Spacer(modifier = Modifier.width(8.dp))
                                Text("Assistant is thinking...", color = Color.Gray, style = MaterialTheme.typography.bodyMedium)
                            }
                        }
                    }
                    is ChatRepository.ChatState.Error -> {
                        item {
                            Box(modifier = Modifier.fillMaxWidth().padding(8.dp), contentAlignment = Alignment.CenterStart) {
                                Text("⚠️ ${state.message}", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodyMedium)
                            }
                        }
                    }
                    else -> {}
                }
            }

            val isBusy = chatState !is ChatRepository.ChatState.Idle && chatState !is ChatRepository.ChatState.Error
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedTextField(
                    value = textInput,
                    onValueChange = { textInput = it },
                    placeholder = { Text("Ask about document...") },
                    modifier = Modifier.weight(1f),
                    maxLines = 3
                )
                Spacer(modifier = Modifier.width(4.dp))
                Box {
                    OutlinedButton(
                        onClick = { lengthMenuExpanded = true },
                        modifier = Modifier.height(48.dp),
                        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 0.dp),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text(
                            text = if (selectedAnswerLength == AnswerLength.SHORT) "Short ▾" else "Long ▾",
                            style = MaterialTheme.typography.bodySmall
                        )
                    }
                    DropdownMenu(
                        expanded = lengthMenuExpanded,
                        onDismissRequest = { lengthMenuExpanded = false }
                    ) {
                        DropdownMenuItem(
                            text = { Text("Short") },
                            onClick = {
                                viewModel.setAnswerLength(AnswerLength.SHORT)
                                lengthMenuExpanded = false
                            },
                            trailingIcon = if (selectedAnswerLength == AnswerLength.SHORT) {
                                { Icon(Icons.Default.Check, contentDescription = "Selected", modifier = Modifier.size(16.dp)) }
                            } else null
                        )
                        DropdownMenuItem(
                            text = { Text("Long") },
                            onClick = {
                                viewModel.setAnswerLength(AnswerLength.LONG)
                                lengthMenuExpanded = false
                            },
                            trailingIcon = if (selectedAnswerLength == AnswerLength.LONG) {
                                { Icon(Icons.Default.Check, contentDescription = "Selected", modifier = Modifier.size(16.dp)) }
                            } else null
                        )
                    }
                }
                Spacer(modifier = Modifier.width(4.dp))
                IconButton(
                    onClick = {
                        if (textInput.isNotBlank()) {
                            viewModel.sendMessage(convId, textInput, linkedDocIds, selectedAnswerLength)
                            textInput = ""
                        }
                    },
                    enabled = textInput.isNotBlank() && !isBusy
                ) {
                    Icon(Icons.Default.Send, contentDescription = "Send")
                }
            }
        }
    }
}

@Composable
fun SettingsScreen(viewModel: MainViewModel) {
    val isEmbedDownloaded by viewModel.isEmbeddingModelDownloaded.collectAsState()
    val installedModels by viewModel.installedModels.collectAsState()
    val activeModelItem by viewModel.activeLlmModelItem.collectAsState()
    val importProgress by viewModel.modelImportProgress.collectAsState()
    val importError by viewModel.modelImportError.collectAsState()
    val themeMode by viewModel.themeMode.collectAsState()

    var modelPendingDeletion by remember { mutableStateOf<com.siddu.offlinedocai.data.ai.ModelItem?>(null) }

    val importModelLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            viewModel.importLlmModel(uri)
        }
    }

    if (modelPendingDeletion != null) {
        val target = modelPendingDeletion!!
        AlertDialog(
            onDismissRequest = { modelPendingDeletion = null },
            title = { Text("Remove Model") },
            text = { Text("Are you sure you want to remove '${target.filename}' (${target.sizeBytes / (1024 * 1024)} MB) from device storage?") },
            confirmButton = {
                TextButton(
                    onClick = {
                        viewModel.deleteModelItem(target)
                        modelPendingDeletion = null
                    }
                ) {
                    Text("Remove", color = MaterialTheme.colorScheme.error)
                }
            },
            dismissButton = {
                TextButton(onClick = { modelPendingDeletion = null }) {
                    Text("Cancel")
                }
            }
        )
    }

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 20.dp, vertical = 16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        item {
            Text("Settings & Model Management", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
            Text("100% On-Device Local AI Pipeline", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
        }

        // 1. Theme Configuration
        item {
            Text("Appearance", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Spacer(modifier = Modifier.height(8.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                MainViewModel.ThemeMode.values().forEach { mode ->
                    val selected = themeMode == mode
                    FilterChip(
                        selected = selected,
                        onClick = { viewModel.setThemeMode(mode) },
                        label = {
                            Text(when (mode) {
                                MainViewModel.ThemeMode.SYSTEM -> "System"
                                MainViewModel.ThemeMode.LIGHT -> "Light"
                                MainViewModel.ThemeMode.DARK -> "Dark"
                            })
                        },
                        leadingIcon = if (selected) {
                            { Icon(Icons.Default.Check, contentDescription = null, modifier = Modifier.size(16.dp)) }
                        } else null
                    )
                }
            }
        }

        // 2. Embedding Model Card
        item {
            Text("Embedding Engine", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Spacer(modifier = Modifier.height(8.dp))
            Card(
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                modifier = Modifier.fillMaxWidth()
            ) {
                Row(
                    modifier = Modifier.padding(16.dp).fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text("all-MiniLM-L6-v2", fontWeight = FontWeight.Bold)
                        Text("ONNX Runtime • 384 dimensions", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
                    }
                    Surface(
                        color = if (isEmbedDownloaded) Color(0xFFE8F5E9) else Color(0xFFFFEBEE),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Text(
                            text = if (isEmbedDownloaded) "READY" else "MISSING",
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
                            color = if (isEmbedDownloaded) Color(0xFF2E7D32) else Color(0xFFC62828),
                            fontWeight = FontWeight.Bold,
                            style = MaterialTheme.typography.labelSmall
                        )
                    }
                }
            }
        }

        // 3. Local GGUF LLM Models Section
        item {
            Spacer(modifier = Modifier.height(8.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text("LOCAL LLM MODELS", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                if (installedModels.isNotEmpty()) {
                    Text("${installedModels.size} installed", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
                }
            }
        }

        // Import in progress card
        if (importProgress != null) {
            val prog = importProgress!!
            item {
                Card(
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer),
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text("Importing GGUF Model...", fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleSmall)
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(prog.filename, style = MaterialTheme.typography.bodySmall)
                        Spacer(modifier = Modifier.height(8.dp))
                        LinearProgressIndicator(
                            progress = if (prog.percent > 0) prog.percent / 100f else 0.05f,
                            modifier = Modifier.fillMaxWidth()
                        )
                        Spacer(modifier = Modifier.height(6.dp))
                        val copiedMb = prog.bytesCopied / (1024 * 1024)
                        val totalMb = prog.totalBytes / (1024 * 1024)
                        Text(
                            text = if (totalMb > 0) "$copiedMb MB / $totalMb MB (${prog.percent}%)" else "$copiedMb MB copied",
                            style = MaterialTheme.typography.labelSmall,
                            color = Color.Gray
                        )
                    }
                }
            }
        }

        // Import error alert
        if (importError != null) {
            item {
                Card(
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text("⚠️ Import Failed", fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onErrorContainer)
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(importError!!, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onErrorContainer)
                        Spacer(modifier = Modifier.height(8.dp))
                        TextButton(onClick = { viewModel.modelManager.clearImportError() }) {
                            Text("Dismiss", color = MaterialTheme.colorScheme.onErrorContainer, fontWeight = FontWeight.Bold)
                        }
                    }
                }
            }
        }

        // Installed Models List
        if (installedModels.isEmpty()) {
            item {
                Card(
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text("No GGUF Model Installed", fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleSmall)
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            "OfflineDocAI requires a local GGUF model for generative neural inference.\nTarget: qwen2.5-1.5b-instruct-q4_k_m.gguf (1.1 GB).\n\nTap '+ Import GGUF Model' below to select the file from your device storage.",
                            style = MaterialTheme.typography.bodySmall,
                            color = Color.Gray
                        )
                    }
                }
            }
        } else {
            items(installedModels) { model ->
                val isActive = model.isActive
                val sizeMb = model.sizeBytes / (1024 * 1024)
                val sizeGb = String.format(java.util.Locale.US, "%.2f", model.sizeBytes / (1024.0 * 1024.0 * 1024.0))

                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = if (isActive) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant
                    ),
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = model.filename,
                                fontWeight = FontWeight.Bold,
                                style = MaterialTheme.typography.titleSmall,
                                modifier = Modifier.weight(1f)
                            )
                            if (isActive) {
                                Surface(
                                    color = Color(0xFF2E7D32),
                                    shape = RoundedCornerShape(8.dp)
                                ) {
                                    Text(
                                        text = "ACTIVE MODEL",
                                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp),
                                        color = Color.White,
                                        fontWeight = FontWeight.Bold,
                                        style = MaterialTheme.typography.labelSmall
                                    )
                                }
                            }
                        }
                        Spacer(modifier = Modifier.height(6.dp))
                        Text("Display Name: ${model.displayName}", style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.SemiBold)
                        Text("Status: ${if (isActive) "Active / Loaded" else model.status.name.lowercase().replaceFirstChar { it.uppercase() }}", style = MaterialTheme.typography.bodySmall)
                        Text("Size: $sizeMb MB ($sizeGb GB)", style = MaterialTheme.typography.bodySmall)
                        Text("Type: GGUF", style = MaterialTheme.typography.bodySmall)
                        Text("Architecture: ${model.architecture} • Quantization: ${model.quantization}", style = MaterialTheme.typography.bodySmall)
                        if (model.parameterCount.isNotBlank()) {
                            Text("Parameters: ${model.parameterCount}", style = MaterialTheme.typography.bodySmall)
                        }

                        Spacer(modifier = Modifier.height(12.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            if (!isActive) {
                                Button(
                                    onClick = { viewModel.selectLlmModel(model.file) },
                                    shape = RoundedCornerShape(8.dp)
                                ) {
                                    Text("Use as active model")
                                }
                            } else {
                                OutlinedButton(
                                    onClick = {},
                                    enabled = false,
                                    shape = RoundedCornerShape(8.dp)
                                ) {
                                    Icon(Icons.Default.Check, contentDescription = null, modifier = Modifier.size(16.dp))
                                    Spacer(modifier = Modifier.width(4.dp))
                                    Text("Active")
                                }
                            }

                            val isGenerating = viewModel.modelManager.isGenerationActive()
                            IconButton(
                                onClick = { modelPendingDeletion = model },
                                enabled = !isGenerating
                            ) {
                                Icon(
                                    Icons.Default.Delete,
                                    contentDescription = "Remove Model",
                                    tint = if (!isGenerating) MaterialTheme.colorScheme.error else Color.Gray
                                )
                            }
                        }
                    }
                }
            }
        }

        // Import GGUF Model Action
        item {
            Spacer(modifier = Modifier.height(4.dp))
            Button(
                onClick = { importModelLauncher.launch(arrayOf("*/*")) },
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp)
            ) {
                Icon(Icons.Default.Add, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("+ Import GGUF Model")
            }
            Spacer(modifier = Modifier.height(16.dp))
        }
    }
}
