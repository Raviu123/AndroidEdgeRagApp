package com.siddu.offlinedocai

import android.app.Activity
import android.os.Bundle
import android.net.Uri
import android.view.MotionEvent
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavHostController
import androidx.navigation.NavType
import androidx.navigation.compose.*
import androidx.navigation.navArgument
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import androidx.core.view.WindowCompat
import com.siddu.offlinedocai.ai.llm.AnswerLength
import com.siddu.offlinedocai.data.entities.ConversationEntity
import com.siddu.offlinedocai.data.entities.DocumentEntity
import com.siddu.offlinedocai.data.entities.MessageEntity
import com.siddu.offlinedocai.data.ai.ModelItem
import com.siddu.offlinedocai.domain.repository.ChatRepository
import com.siddu.offlinedocai.domain.repository.DocumentRepository
import com.siddu.offlinedocai.ai.llm.AvailableLlmModel
import com.siddu.offlinedocai.ai.llm.AvailableLlmModels
import com.siddu.offlinedocai.ai.export.ChatExporter
import com.siddu.offlinedocai.ai.queue.QuestionQueueParser
import com.siddu.offlinedocai.ai.queue.QuestionQueueState
import com.siddu.offlinedocai.ui.components.MarkdownMessage
import com.siddu.offlinedocai.data.download.DownloadStatus
import com.siddu.offlinedocai.ai.llm.LlmLoadState
import com.siddu.offlinedocai.ai.embeddings.AvailableEmbeddingModels
import com.siddu.offlinedocai.ai.embeddings.EmbeddingModelSpec
import com.siddu.offlinedocai.ai.embeddings.EmbeddingModelStatus
import com.siddu.offlinedocai.data.ai.ModelStatus
import kotlinx.coroutines.launch
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        installSplashScreen()
        super.onCreate(savedInstanceState)
        try {
            com.tom_roush.pdfbox.android.PDFBoxResourceLoader.init(applicationContext)
        } catch (_: Exception) {}
        setContent { OfflineDocAIApp() }
    }

    override fun dispatchGenericMotionEvent(event: MotionEvent?): Boolean {
        if (event?.actionMasked == MotionEvent.ACTION_HOVER_EXIT) {
            return true
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

// =========================================================================
// CHATGPT COLOR PALETTE & ROOT APP
// =========================================================================

private val ChatGPTGreen = Color(0xFF10A37F)
private val DarkBg = Color(0xFF171717)
private val DarkSurface = Color(0xFF1E1E1E)
private val DarkCard = Color(0xFF2A2A2A)
private val DarkText = Color(0xFFECECEC)
private val DarkTextSecondary = Color(0xFF9E9E9E)

private val LightBg = Color(0xFFFFFFFF)
private val LightSurface = Color(0xFFF7F7F8)
private val LightCard = Color(0xFFECECEC)
private val LightText = Color(0xFF0D0D0D)
private val LightTextSecondary = Color(0xFF6B6B6B)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun OfflineDocAIApp(mainViewModel: MainViewModel = viewModel()) {
    val themeMode by mainViewModel.themeMode.collectAsState()
    val isDark = when (themeMode) {
        MainViewModel.ThemeMode.SYSTEM -> isSystemInDarkTheme()
        MainViewModel.ThemeMode.DARK -> true
        MainViewModel.ThemeMode.LIGHT -> false
    }

    val customColors = if (isDark) {
        darkColorScheme(
            primary = ChatGPTGreen,
            onPrimary = Color.White,
            secondary = ChatGPTGreen,
            background = DarkBg,
            surface = DarkSurface,
            surfaceVariant = DarkCard,
            onBackground = DarkText,
            onSurface = DarkText,
            onSurfaceVariant = DarkTextSecondary
        )
    } else {
        lightColorScheme(
            primary = ChatGPTGreen,
            onPrimary = Color.White,
            secondary = ChatGPTGreen,
            background = LightBg,
            surface = LightSurface,
            surfaceVariant = LightCard,
            onBackground = LightText,
            onSurface = LightText,
            onSurfaceVariant = LightTextSecondary
        )
    }

    MaterialTheme(colorScheme = customColors) {
        val view = LocalView.current
        val window = (view.context as Activity).window

        SideEffect {
            window.statusBarColor = customColors.surface.toArgb()
            window.navigationBarColor = customColors.surface.toArgb()
            WindowCompat.getInsetsController(window, view).apply {
                isAppearanceLightStatusBars = !isDark
                isAppearanceLightNavigationBars = !isDark
            }
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
                window.isStatusBarContrastEnforced = false
                window.isNavigationBarContrastEnforced = false
            }
        }

        val navController = rememberNavController()
        val drawerState = rememberDrawerState(initialValue = DrawerValue.Closed)
        val scope = rememberCoroutineScope()

        val conversations by mainViewModel.conversations.collectAsState()
        val documents by mainViewModel.documents.collectAsState()
        val activeModel by mainViewModel.activeLlmModelItem.collectAsState()
        val llmLoadState by mainViewModel.llmLoadState.collectAsState()

        var convToRename by remember { mutableStateOf<ConversationEntity?>(null) }
        var convToDelete by remember { mutableStateOf<ConversationEntity?>(null) }

        // Rename Dialog
        if (convToRename != null) {
            val target = convToRename!!
            var newTitle by remember { mutableStateOf(target.title) }
            AlertDialog(
                onDismissRequest = { convToRename = null },
                title = { Text("Rename Chat") },
                text = {
                    OutlinedTextField(
                        value = newTitle,
                        onValueChange = { newTitle = it },
                        singleLine = true,
                        modifier = Modifier.fillMaxWidth()
                    )
                },
                confirmButton = {
                    TextButton(
                        onClick = {
                            if (newTitle.isNotBlank()) {
                                mainViewModel.renameConversation(target.id, newTitle)
                            }
                            convToRename = null
                        }
                    ) {
                        Text("Save")
                    }
                },
                dismissButton = {
                    TextButton(onClick = { convToRename = null }) {
                        Text("Cancel")
                    }
                }
            )
        }

        // Delete Dialog
        if (convToDelete != null) {
            val target = convToDelete!!
            AlertDialog(
                onDismissRequest = { convToDelete = null },
                title = { Text("Delete Chat") },
                text = { Text("Are you sure you want to delete \"${target.title}\"? This action cannot be undone.") },
                confirmButton = {
                    TextButton(
                        onClick = {
                            mainViewModel.deleteConversation(target.id)
                            convToDelete = null
                            navController.navigate("chat?convId=-1") {
                                popUpTo("chat?convId=-1") { inclusive = true }
                            }
                        }
                    ) {
                        Text("Delete", color = MaterialTheme.colorScheme.error)
                    }
                },
                dismissButton = {
                    TextButton(onClick = { convToDelete = null }) {
                        Text("Cancel")
                    }
                }
            )
        }

        ModalNavigationDrawer(
            drawerState = drawerState,
            drawerContent = {
                ModalDrawerSheet(
                    modifier = Modifier.width(310.dp),
                    drawerContainerColor = MaterialTheme.colorScheme.surface
                ) {
                    ChatGPTDrawerContent(
                        conversations = conversations,
                        documentsCount = documents.size,
                        activeModel = activeModel,
                        llmLoadState = llmLoadState,
                        themeMode = themeMode,
                        onNewChat = {
                            scope.launch {
                                drawerState.close()
                                navController.navigate("chat?convId=-1") {
                                    launchSingleTop = true
                                }
                            }
                        },
                        onSelectConversation = { convId ->
                            scope.launch {
                                drawerState.close()
                                navController.navigate("chat?convId=$convId") {
                                    launchSingleTop = true
                                }
                            }
                        },
                        onOpenLibrary = {
                            scope.launch {
                                drawerState.close()
                                navController.navigate("documents") {
                                    launchSingleTop = true
                                }
                            }
                        },
                        onOpenSettings = {
                            scope.launch {
                                drawerState.close()
                                navController.navigate("settings") {
                                    launchSingleTop = true
                                }
                            }
                        },
                        onRenameConversation = { conv -> convToRename = conv },
                        onDeleteConversation = { conv -> convToDelete = conv },
                        onShareConversation = { convId ->
                            mainViewModel.shareConversationPdf(navController.context, convId)
                        },
                        onToggleTheme = {
                            val nextMode = when (themeMode) {
                                MainViewModel.ThemeMode.DARK -> MainViewModel.ThemeMode.LIGHT
                                MainViewModel.ThemeMode.LIGHT -> MainViewModel.ThemeMode.DARK
                                MainViewModel.ThemeMode.SYSTEM -> if (isDark) MainViewModel.ThemeMode.LIGHT else MainViewModel.ThemeMode.DARK
                            }
                            mainViewModel.setThemeMode(nextMode)
                        }
                    )
                }
            }
        ) {
          Box(modifier = Modifier.fillMaxSize()) {
            NavHost(
                navController = navController,
                startDestination = "chat?convId=-1"
            ) {
                composable(
                    route = "chat?convId={convId}",
                    arguments = listOf(
                        navArgument("convId") {
                            type = NavType.LongType
                            defaultValue = -1L
                        }
                    )
                ) { backStackEntry ->
                    val convIdArg = backStackEntry.arguments?.getLong("convId") ?: -1L
                    ChatScreen(
                        convId = if (convIdArg > 0L) convIdArg else null,
                        viewModel = mainViewModel,
                        navController = navController,
                        onOpenDrawer = { scope.launch { drawerState.open() } },
                        onRenameChat = { id, title ->
                            convToRename = conversations.firstOrNull { it.id == id }
                        },
                        onDeleteChat = { id ->
                            convToDelete = conversations.firstOrNull { it.id == id }
                        }
                    )
                }

                composable("documents") {
                    DocumentsScreen(
                        viewModel = mainViewModel,
                        navController = navController,
                        onOpenDrawer = { scope.launch { drawerState.open() } }
                    )
                }

                composable("documentDetail/{docId}") { backStackEntry ->
                    val docId = backStackEntry.arguments?.getString("docId")?.toLongOrNull() ?: return@composable
                    DocumentDetailScreen(
                        docId = docId,
                        viewModel = mainViewModel,
                        navController = navController
                    )
                }

                composable("settings") {
                    SettingsScreen(
                        viewModel = mainViewModel,
                        navController = navController,
                        onOpenDrawer = { scope.launch { drawerState.open() } }
                    )
                }
            }

            // App-wide messages (model problems found at startup, download/switch results).
            val snackbarHostState = remember { SnackbarHostState() }
            val userMessage by mainViewModel.userMessage.collectAsState()
            LaunchedEffect(userMessage) {
                userMessage?.let {
                    snackbarHostState.showSnackbar(it, withDismissAction = true, duration = SnackbarDuration.Long)
                    mainViewModel.consumeUserMessage()
                }
            }
            SnackbarHost(
                hostState = snackbarHostState,
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 88.dp)
            )

            // "Download finished — use it now?"
            val pendingSwitch by mainViewModel.pendingSwitch.collectAsState()
            pendingSwitch?.let { pending ->
                val reindexCount = if (pending.isEmbedding) mainViewModel.documentsToReindex() else 0
                AlertDialog(
                    onDismissRequest = { mainViewModel.dismissPendingSwitch() },
                    title = { Text("${pending.name} is ready") },
                    text = {
                        Text(
                            if (pending.isEmbedding) {
                                "Use it as the search model now?" + if (reindexCount > 0) " Your $reindexCount document(s) will be re-indexed in the background." else ""
                            } else {
                                "Use it as the language model now?"
                            }
                        )
                    },
                    confirmButton = { TextButton(onClick = { mainViewModel.confirmPendingSwitch() }) { Text("Use now") } },
                    dismissButton = { TextButton(onClick = { mainViewModel.dismissPendingSwitch() }) { Text("Later") } }
                )
            }
          }
        }
    }
}

// =========================================================================
// CHATGPT LEFT NAVIGATION DRAWER
// =========================================================================

@Composable
fun ChatGPTDrawerContent(
    conversations: List<ConversationEntity>,
    documentsCount: Int,
    activeModel: ModelItem?,
    llmLoadState: LlmLoadState = LlmLoadState.NoModel,
    themeMode: MainViewModel.ThemeMode,
    onNewChat: () -> Unit,
    onSelectConversation: (Long) -> Unit,
    onOpenLibrary: () -> Unit,
    onOpenSettings: () -> Unit,
    onRenameConversation: (ConversationEntity) -> Unit,
    onDeleteConversation: (ConversationEntity) -> Unit,
    onShareConversation: (Long) -> Unit,
    onToggleTheme: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .statusBarsPadding()
            .navigationBarsPadding()
            .padding(horizontal = 14.dp, vertical = 12.dp)
    ) {
        // App Title & Badge
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 6.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    modifier = Modifier
                        .size(32.dp)
                        .clip(CircleShape)
                        .background(ChatGPTGreen),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Default.AutoAwesome,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(18.dp)
                    )
                }
                Spacer(modifier = Modifier.width(10.dp))
                Column {
                    Text(
                        text = "OfflineDocAI",
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                    )
                    Text(
                        text = "100% On-Device RAG",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            IconButton(onClick = onToggleTheme, modifier = Modifier.size(32.dp)) {
                Icon(
                    imageVector = if (themeMode == MainViewModel.ThemeMode.DARK) Icons.Default.LightMode else Icons.Default.DarkMode,
                    contentDescription = "Toggle Theme",
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.size(18.dp)
                )
            }
        }

        Spacer(modifier = Modifier.height(10.dp))

        // "+ New Chat" Button
        Surface(
            onClick = onNewChat,
            shape = RoundedCornerShape(12.dp),
            color = MaterialTheme.colorScheme.surfaceVariant,
            modifier = Modifier.fillMaxWidth()
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 14.dp, vertical = 12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = Icons.Default.Add,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(20.dp)
                )
                Spacer(modifier = Modifier.width(12.dp))
                Text(
                    text = "New Chat",
                    style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold)
                )
            }
        }

        Spacer(modifier = Modifier.height(14.dp))

        // Navigation Links: Library & Settings
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(10.dp))
                .clickable { onOpenLibrary() }
                .padding(horizontal = 10.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = Icons.Default.Folder,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.size(20.dp)
            )
            Spacer(modifier = Modifier.width(12.dp))
            Text(
                text = "Library",
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.weight(1f)
            )
            if (documentsCount > 0) {
                Surface(
                    shape = RoundedCornerShape(10.dp),
                    color = MaterialTheme.colorScheme.surfaceVariant
                ) {
                    Text(
                        text = "$documentsCount",
                        style = MaterialTheme.typography.labelSmall,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp)
                    )
                }
            }
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(10.dp))
                .clickable { onOpenSettings() }
                .padding(horizontal = 10.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = Icons.Default.Settings,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.size(20.dp)
            )
            Spacer(modifier = Modifier.width(12.dp))
            Text(
                text = "Settings & Models",
                style = MaterialTheme.typography.bodyMedium
            )
        }

        HorizontalDivider(
            modifier = Modifier.padding(vertical = 8.dp),
            color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f)
        )

        // Section: Chats & Search
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 10.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text = "Recent Chats",
                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            if (conversations.isNotEmpty()) {
                Text(
                    text = "${conversations.size}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)
                )
            }
        }

        var chatSearchQuery by remember { mutableStateOf("") }

        if (conversations.size > 2) {
            OutlinedTextField(
                value = chatSearchQuery,
                onValueChange = { chatSearchQuery = it },
                placeholder = { Text("Search chats...", style = MaterialTheme.typography.bodySmall) },
                leadingIcon = {
                    Icon(
                        Icons.Default.Search,
                        contentDescription = null,
                        modifier = Modifier.size(16.dp),
                        tint = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                },
                trailingIcon = {
                    if (chatSearchQuery.isNotBlank()) {
                        IconButton(onClick = { chatSearchQuery = "" }, modifier = Modifier.size(20.dp)) {
                            Icon(Icons.Default.Close, contentDescription = "Clear", modifier = Modifier.size(14.dp))
                        }
                    }
                },
                singleLine = true,
                textStyle = MaterialTheme.typography.bodySmall,
                shape = RoundedCornerShape(10.dp),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedContainerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.4f),
                    unfocusedContainerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.2f),
                    focusedBorderColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.4f),
                    unfocusedBorderColor = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.2f)
                ),
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 4.dp, vertical = 4.dp)
            )
        }

        val filteredConversations = remember(conversations, chatSearchQuery) {
            if (chatSearchQuery.isBlank()) conversations
            else conversations.filter { it.title.contains(chatSearchQuery, ignoreCase = true) }
        }

        if (filteredConversations.isEmpty()) {
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = if (chatSearchQuery.isNotBlank()) "No matching chats" else "No recent chats",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
                )
            }
        } else {
            LazyColumn(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(2.dp)
            ) {
                items(filteredConversations, key = { it.id }) { conv ->
                    DrawerConversationItem(
                        conversation = conv,
                        onClick = { onSelectConversation(conv.id) },
                        onRename = { onRenameConversation(conv) },
                        onShare = { onShareConversation(conv.id) },
                        onDelete = { onDeleteConversation(conv) }
                    )
                }
            }
        }

        HorizontalDivider(
            modifier = Modifier.padding(vertical = 8.dp),
            color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f)
        )

        // Drawer Footer: Model Status
        Surface(
            shape = RoundedCornerShape(10.dp),
            color = MaterialTheme.colorScheme.surfaceVariant,
            modifier = Modifier.fillMaxWidth()
        ) {
            Row(
                modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                val (statusText, statusColor) = llmStatusLabel(llmLoadState)
                Box(
                    modifier = Modifier
                        .size(8.dp)
                        .clip(CircleShape)
                        .background(statusColor)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = activeModel?.displayName ?: "No Model Active",
                    style = MaterialTheme.typography.labelMedium,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.weight(1f)
                )
                Text(
                    text = statusText,
                    style = MaterialTheme.typography.labelSmall,
                    color = statusColor,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}

@Composable
fun DrawerConversationItem(
    conversation: ConversationEntity,
    onClick: () -> Unit,
    onRename: () -> Unit,
    onShare: () -> Unit,
    onDelete: () -> Unit
) {
    var menuExpanded by remember { mutableStateOf(false) }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .clickable { onClick() }
            .padding(horizontal = 10.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Row(
            modifier = Modifier.weight(1f),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = Icons.Default.Chat,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                modifier = Modifier.size(16.dp)
            )
            Spacer(modifier = Modifier.width(10.dp))
            Text(
                text = conversation.title,
                style = MaterialTheme.typography.bodyMedium,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }

        Box {
            IconButton(
                onClick = { menuExpanded = true },
                modifier = Modifier.size(24.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.MoreVert,
                    contentDescription = "Options",
                    tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
                    modifier = Modifier.size(16.dp)
                )
            }

            DropdownMenu(
                expanded = menuExpanded,
                onDismissRequest = { menuExpanded = false }
            ) {
                DropdownMenuItem(
                    text = { Text("Rename") },
                    onClick = {
                        menuExpanded = false
                        onRename()
                    },
                    leadingIcon = { Icon(Icons.Default.Edit, contentDescription = null, modifier = Modifier.size(18.dp)) }
                )
                DropdownMenuItem(
                    text = { Text("Share as PDF") },
                    onClick = {
                        menuExpanded = false
                        onShare()
                    },
                    leadingIcon = { Icon(Icons.Default.Share, contentDescription = null, modifier = Modifier.size(18.dp)) }
                )
                DropdownMenuItem(
                    text = { Text("Delete", color = MaterialTheme.colorScheme.error) },
                    onClick = {
                        menuExpanded = false
                        onDelete()
                    },
                    leadingIcon = { Icon(Icons.Default.Delete, contentDescription = null, tint = MaterialTheme.colorScheme.error, modifier = Modifier.size(18.dp)) }
                )
            }
        }
    }
}

// =========================================================================
// CHATGPT CHAT HOME & CONVERSATION SCREEN
// =========================================================================

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(
    convId: Long?,
    viewModel: MainViewModel,
    navController: NavHostController,
    onOpenDrawer: () -> Unit,
    onRenameChat: (Long, String) -> Unit,
    onDeleteChat: (Long) -> Unit
) {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()

    val conversations by viewModel.conversations.collectAsState()
    val documents by viewModel.documents.collectAsState()
    val chatState by viewModel.chatState.collectAsState()
    val selectedAnswerLength by viewModel.selectedAnswerLength.collectAsState()
    val questionQueueState by viewModel.questionQueueState.collectAsState()

    val activeConv = conversations.firstOrNull { it.id == convId }
    val messagesState = remember(convId) {
        if (convId != null && convId > 0L) {
            viewModel.messagesForConversation(convId)
        } else null
    }
    val messages = messagesState?.collectAsState()?.value ?: emptyList()

    var textInput by remember { mutableStateOf("") }
    var selectedDocId by remember(convId, activeConv?.documentId) {
        mutableStateOf(activeConv?.documentId)
    }
    var showDocumentPicker by remember { mutableStateOf(false) }
    var showBatchQueueDialog by remember { mutableStateOf(false) }
    var chatMenuExpanded by remember { mutableStateOf(false) }

    val listState = rememberLazyListState()

    // Auto-scroll to bottom on new message
    LaunchedEffect(messages.size, chatState) {
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(messages.size - 1)
        }
    }

    val isBusy = chatState is ChatRepository.ChatState.Retrieving || chatState is ChatRepository.ChatState.Preparing ||
        chatState is ChatRepository.ChatState.Generating
    val documentPickerState = rememberModalBottomSheetState(skipPartiallyExpanded = true)

    if (showBatchQueueDialog) {
        BatchQuestionsDialog(
            onDismiss = { showBatchQueueDialog = false },
            onStartQueue = { questions ->
                val effectiveDocIds = if (selectedDocId != null) listOf(selectedDocId!!) else emptyList()
                if (convId == null || convId <= 0L) {
                    coroutineScope.launch {
                        val title = questions.firstOrNull()?.take(30)?.trim()?.ifBlank { "Batch Chat" } ?: "Batch Chat"
                        val newId = viewModel.startConversation(title, effectiveDocIds)
                        viewModel.startQuestionQueue(newId, questions, effectiveDocIds)
                        navController.navigate("chat?convId=$newId") {
                            popUpTo("chat?convId=-1") { inclusive = true }
                        }
                    }
                } else {
                    viewModel.startQuestionQueue(convId, questions, effectiveDocIds)
                }
            }
        )
    }

    if (showDocumentPicker) {
        ModalBottomSheet(
            onDismissRequest = { showDocumentPicker = false },
            sheetState = documentPickerState,
            containerColor = MaterialTheme.colorScheme.surface
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                            .padding(horizontal = 20.dp)
                            .fillMaxHeight(0.85f)
                            .padding(bottom = 24.dp)
                    ) {
                        Text(
                            text = "Choose a document",
                            style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold)
                        )
                        Spacer(modifier = Modifier.height(12.dp))
                        LazyColumn(modifier = Modifier.weight(1f)) {
                            item {
                                Surface(
                                    onClick = {
                                        selectedDocId = null
                                        showDocumentPicker = false
                                    },
                                    shape = RoundedCornerShape(12.dp),
                                    color = MaterialTheme.colorScheme.surfaceVariant,
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    Text(
                                        text = "No document (general chat)",
                                        style = MaterialTheme.typography.bodyLarge,
                                        modifier = Modifier.padding(16.dp)
                                    )
                                }
                            }
                            if (documents.isEmpty()) {
                                item {
                                    Text(
                                        text = "No documents in your library yet.",
                                        style = MaterialTheme.typography.bodyMedium,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        modifier = Modifier.padding(vertical = 12.dp)
                                    )
                                }
                            } else {
                                items(documents, key = { it.id }) { document ->
                                    Surface(
                                        onClick = {
                                            selectedDocId = document.id
                                            showDocumentPicker = false
                                        },
                                        shape = RoundedCornerShape(12.dp),
                                        color = if (selectedDocId == document.id) {
                                            MaterialTheme.colorScheme.primaryContainer
                                        } else {
                                            MaterialTheme.colorScheme.surface
                                        },
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(top = 6.dp)
                                    ) {
                                        Row(
                                            modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp),
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            Icon(
                                                Icons.Default.Description,
                                                contentDescription = null,
                                                tint = ChatGPTGreen
                                            )
                                            Spacer(modifier = Modifier.width(12.dp))
                                            Text(
                                                text = document.name,
                                                maxLines = 1,
                                                overflow = TextOverflow.Ellipsis,
                                                modifier = Modifier.weight(1f)
                                            )
                                            if (selectedDocId == document.id) {
                                                Icon(Icons.Default.Check, contentDescription = "Selected")
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        TextButton(
                    onClick = {
                        showDocumentPicker = false
                        navController.navigate("documents") { launchSingleTop = true }
                    },
                    modifier = Modifier.align(Alignment.End)
                ) {
                    Icon(Icons.Default.Folder, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Open Library")
                }
            }
        }
    }

    fun submitMessage(promptText: String) {
        if (promptText.isBlank() || isBusy) return
        // Only use document RAG context when the user has explicitly selected a document.
        // If selectedDocId is null (no document chosen), route as normal conversational chat.
        val effectiveDocIds = if (selectedDocId != null) listOf(selectedDocId!!) else emptyList()

        if (convId == null || convId <= 0L) {
            // Create new conversation and navigate
            coroutineScope.launch {
                val title = promptText.take(30).trim().ifBlank { "New Chat" }
                val newId = viewModel.startConversation(title, effectiveDocIds)
                viewModel.sendMessage(newId, promptText, effectiveDocIds, selectedAnswerLength)
                navController.navigate("chat?convId=$newId") {
                    popUpTo("chat?convId=-1") { inclusive = true }
                }
            }
        } else {
            viewModel.sendMessage(convId, promptText, effectiveDocIds, selectedAnswerLength)
        }
        textInput = ""
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = activeConv?.title ?: "OfflineDocAI",
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onOpenDrawer) {
                        Icon(Icons.Default.Menu, contentDescription = "Menu")
                    }
                },
                actions = {
                    IconButton(onClick = { showBatchQueueDialog = true }) {
                        Icon(
                            Icons.Default.PlaylistPlay,
                            contentDescription = "Batch Questions Queue",
                            tint = if (questionQueueState.isActive) ChatGPTGreen else MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }

                    if (convId != null && convId > 0L) {
                        Box {
                            IconButton(onClick = { chatMenuExpanded = true }) {
                                Icon(Icons.Default.MoreVert, contentDescription = "More Options")
                            }
                            DropdownMenu(
                                expanded = chatMenuExpanded,
                                onDismissRequest = { chatMenuExpanded = false }
                            ) {
                                DropdownMenuItem(
                                    text = { Text("Batch Question Queue") },
                                    onClick = {
                                        chatMenuExpanded = false
                                        showBatchQueueDialog = true
                                    },
                                    leadingIcon = { Icon(Icons.Default.PlaylistPlay, contentDescription = null, modifier = Modifier.size(18.dp)) }
                                )
                                DropdownMenuItem(
                                    text = { Text("Rename Chat") },
                                    onClick = {
                                        chatMenuExpanded = false
                                        activeConv?.let { onRenameChat(it.id, it.title) }
                                    },
                                    leadingIcon = { Icon(Icons.Default.Edit, contentDescription = null, modifier = Modifier.size(18.dp)) }
                                )
                                DropdownMenuItem(
                                    text = { Text("Copy Full Chat (JSON + Chunks)") },
                                    onClick = {
                                        chatMenuExpanded = false
                                        ChatExporter.copyChatToClipboard(
                                            context = context,
                                            conversationId = convId,
                                            conversationTitle = activeConv?.title ?: "Chat",
                                            documentId = activeConv?.documentId,
                                            messages = messages,
                                            includeChunks = true
                                        )
                                    },
                                    leadingIcon = { Icon(Icons.Default.ContentCopy, contentDescription = null, modifier = Modifier.size(18.dp)) }
                                )
                                DropdownMenuItem(
                                    text = { Text("Copy Q&A (JSON)") },
                                    onClick = {
                                        chatMenuExpanded = false
                                        ChatExporter.copyChatToClipboard(
                                            context = context,
                                            conversationId = convId,
                                            conversationTitle = activeConv?.title ?: "Chat",
                                            documentId = activeConv?.documentId,
                                            messages = messages,
                                            includeChunks = false
                                        )
                                    },
                                    leadingIcon = { Icon(Icons.Default.CopyAll, contentDescription = null, modifier = Modifier.size(18.dp)) }
                                )
                                DropdownMenuItem(
                                    text = { Text("Copy Q&A (Plain Text)") },
                                    onClick = {
                                        chatMenuExpanded = false
                                        ChatExporter.copyPlainTextQnAToClipboard(
                                            context = context,
                                            conversationTitle = activeConv?.title ?: "Chat",
                                            messages = messages
                                        )
                                    },
                                    leadingIcon = { Icon(Icons.Default.Description, contentDescription = null, modifier = Modifier.size(18.dp)) }
                                )
                                DropdownMenuItem(
                                    text = { Text("Share as PDF") },
                                    onClick = {
                                        chatMenuExpanded = false
                                        viewModel.shareConversationPdf(context, convId)
                                    },
                                    leadingIcon = { Icon(Icons.Default.Share, contentDescription = null, modifier = Modifier.size(18.dp)) }
                                )
                                DropdownMenuItem(
                                    text = { Text("Delete Chat", color = MaterialTheme.colorScheme.error) },
                                    onClick = {
                                        chatMenuExpanded = false
                                        onDeleteChat(convId)
                                    },
                                    leadingIcon = { Icon(Icons.Default.Delete, contentDescription = null, tint = MaterialTheme.colorScheme.error, modifier = Modifier.size(18.dp)) }
                                )
                            }
                        }
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .imePadding()
        ) {
            // Batch Queue Active Banner
            if (questionQueueState.isActive) {
                QuestionQueueBanner(
                    queueState = questionQueueState,
                    onPause = { viewModel.pauseQuestionQueue() },
                    onResume = { viewModel.resumeQuestionQueue() },
                    onStop = { viewModel.stopQuestionQueue() }
                )
            }

            // Main Chat Area
            if (messages.isEmpty() && !isBusy && chatState !is ChatRepository.ChatState.Error) {
                // ChatGPT Hero Home Screen
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth()
                        .padding(horizontal = 24.dp),
                    contentAlignment = Alignment.Center
                ) {
                    ChatGPTHeroEmptyState()
                }
            } else {
                // Messages List
                LazyColumn(
                    state = listState,
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp),
                    verticalArrangement = Arrangement.spacedBy(16.dp),
                    contentPadding = PaddingValues(vertical = 12.dp)
                ) {
                    items(messages, key = { it.id }) { msg ->
                        val isUser = msg.role.equals("user", ignoreCase = true)
                        if (isUser) {
                            UserMessageBubble(text = msg.content)
                        } else {
                            val msgIndex = messages.indexOf(msg)
                            val userQuestion = if (msgIndex > 0 && messages[msgIndex - 1].role.equals("user", ignoreCase = true)) {
                                messages[msgIndex - 1].content
                            } else null
                            AssistantMessageItem(
                                text = msg.content,
                                metricsJson = msg.metrics,
                                userQuestion = userQuestion
                            )
                        }
                    }

                    // Generation Indicator State with live timer
                    if (isBusy) {
                        item {
                            AssistantThinkingItem(chatState = chatState)
                        }
                    }

                    // Error State
                    if (chatState is ChatRepository.ChatState.Error) {
                        item {
                            val errorState = chatState as ChatRepository.ChatState.Error
                            Card(
                                colors = CardDefaults.cardColors(
                                    containerColor = MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.7f)
                                ),
                                shape = RoundedCornerShape(12.dp),
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Row(
                                    modifier = Modifier.padding(14.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Icon(
                                        imageVector = Icons.Default.Close,
                                        contentDescription = null,
                                        tint = MaterialTheme.colorScheme.error,
                                        modifier = Modifier.size(20.dp)
                                    )
                                    Spacer(modifier = Modifier.width(10.dp))
                                    Text(
                                        text = errorState.message,
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onErrorContainer,
                                        modifier = Modifier.weight(1f)
                                    )
                                }
                            }
                        }
                    }
                }
            }

            // ChatGPT Message Composer
            ChatGPTComposer(
                textInput = textInput,
                onTextChanged = { textInput = it },
                onSend = { submitMessage(textInput) },
                isBusy = isBusy,
                onStop = { viewModel.stopGeneration() },
                selectedDocument = documents.firstOrNull { it.id == selectedDocId },
                onOpenDocumentPicker = { showDocumentPicker = true },
                onClearSelectedDoc = { selectedDocId = null },
                answerLength = selectedAnswerLength,
                onToggleAnswerLength = {
                    val next = when (selectedAnswerLength) {
                        AnswerLength.DEFAULT -> AnswerLength.SHORT
                        AnswerLength.SHORT -> AnswerLength.LONG
                        AnswerLength.LONG -> AnswerLength.DEFAULT
                    }
                    viewModel.setAnswerLength(next)
                }
            )
        }
    }
}

// =========================================================================
// CHATGPT HERO EMPTY STATE
// =========================================================================

@Composable
fun ChatGPTHeroEmptyState() {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
        modifier = Modifier.fillMaxWidth()
    ) {
        // Hero Icon
        Box(
            modifier = Modifier
                .size(60.dp)
                .clip(CircleShape)
                .background(ChatGPTGreen),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = Icons.Default.AutoAwesome,
                contentDescription = null,
                tint = Color.White,
                modifier = Modifier.size(32.dp)
            )
        }

        Spacer(modifier = Modifier.height(18.dp))

        Text(
            text = "How can I help you today?",
            style = MaterialTheme.typography.headlineSmall.copy(fontWeight = FontWeight.Bold),
            color = MaterialTheme.colorScheme.onBackground
        )

    }
}


// =========================================================================
// CHAT BUBBLES & THINKING INDICATOR
// =========================================================================

@Composable
fun UserMessageBubble(text: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.End
    ) {
        Surface(
            shape = RoundedCornerShape(18.dp),
            color = MaterialTheme.colorScheme.surfaceVariant,
            modifier = Modifier.widthIn(max = 280.dp)
        ) {
            Text(
                text = text,
                style = MaterialTheme.typography.bodyMedium.copy(lineHeight = 20.sp),
                color = MaterialTheme.colorScheme.onSurface,
                modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp)
            )
        }
    }
}

@Composable
fun AssistantMessageItem(
    text: String,
    metricsJson: String? = null,
    userQuestion: String? = null
) {
    val metrics = remember(metricsJson) { com.siddu.offlinedocai.domain.model.ResponseMetrics.fromJson(metricsJson) }
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.Start
    ) {
        Box(
            modifier = Modifier
                .padding(top = 2.dp)
                .size(26.dp)
                .clip(CircleShape)
                .background(ChatGPTGreen),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = Icons.Default.AutoAwesome,
                contentDescription = null,
                tint = Color.White,
                modifier = Modifier.size(14.dp)
            )
        }
        Spacer(modifier = Modifier.width(10.dp))
        Column(modifier = Modifier.weight(1f)) {
            MarkdownMessage(
                text = text,
                modifier = Modifier.fillMaxWidth(),
                isAssistant = true
            )
            if (metrics != null) {
                com.siddu.offlinedocai.ui.components.ResponseMetricsDropdown(
                    metrics = metrics,
                    userQuestion = userQuestion,
                    assistantAnswer = text
                )
            }
        }
    }
}

@Composable
fun AssistantThinkingItem(chatState: ChatRepository.ChatState = ChatRepository.ChatState.Retrieving) {
    var elapsedSeconds by remember { mutableIntStateOf(0) }
    LaunchedEffect(Unit) {
        while (true) {
            kotlinx.coroutines.delay(1000)
            elapsedSeconds++
        }
    }

    val stateSeconds = if (chatState is ChatRepository.ChatState.Generating && chatState.elapsedSeconds > 0) {
        chatState.elapsedSeconds
    } else {
        elapsedSeconds
    }

    val statusLabel = when (chatState) {
        is ChatRepository.ChatState.Retrieving -> "Searching documents"
        is ChatRepository.ChatState.Preparing -> "Preparing the model"
        is ChatRepository.ChatState.Generating -> "Generating response"
        else -> "Thinking"
    }
    val partialText = (chatState as? ChatRepository.ChatState.Generating)?.partialText.orEmpty()

    // Animated dots
    var dotCount by remember { mutableIntStateOf(0) }
    LaunchedEffect(Unit) {
        while (true) {
            kotlinx.coroutines.delay(500)
            dotCount = (dotCount + 1) % 4
        }
    }
    val dots = ".".repeat(dotCount)

    Column(modifier = Modifier.fillMaxWidth()) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Pulsing logo
        val infiniteTransition = rememberInfiniteTransition(label = "thinking")
        val alpha by infiniteTransition.animateFloat(
            initialValue = 0.4f,
            targetValue = 1f,
            animationSpec = infiniteRepeatable(
                animation = tween(800, easing = androidx.compose.animation.core.FastOutSlowInEasing),
                repeatMode = RepeatMode.Reverse
            ),
            label = "pulse"
        )

        Box(
            modifier = Modifier
                .size(26.dp)
                .clip(CircleShape)
                .background(ChatGPTGreen.copy(alpha = alpha)),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = Icons.Default.AutoAwesome,
                contentDescription = null,
                tint = Color.White,
                modifier = Modifier.size(14.dp)
            )
        }
        Spacer(modifier = Modifier.width(10.dp))
        Text(
            text = "$statusLabel$dots (${stateSeconds}s)",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
    // The answer streams in while it is generated.
    if (partialText.isNotBlank()) {
        AssistantMessageItem(text = partialText)
    }
    }
}

// =========================================================================
// BATCH QUESTIONS QUEUE DIALOG & BANNER
// =========================================================================

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BatchQuestionsDialog(
    onDismiss: () -> Unit,
    onStartQueue: (List<String>) -> Unit
) {
    var rawInput by remember {
        mutableStateOf("[\n  \"What is the summary of this document?\",\n  \"What are the key findings?\",\n  \"What are the conclusions?\"\n]")
    }
    val parsedQuestions = remember(rawInput) {
        QuestionQueueParser.parseQuestions(rawInput)
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = Icons.Default.PlaylistPlay,
                    contentDescription = null,
                    tint = ChatGPTGreen,
                    modifier = Modifier.size(24.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = "Batch Question Queue",
                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                )
            }
        },
        text = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 4.dp)
            ) {
                Text(
                    text = "Enter a JSON array of strings or one question per line. The AI will process each question sequentially.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Spacer(modifier = Modifier.height(10.dp))

                OutlinedTextField(
                    value = rawInput,
                    onValueChange = { rawInput = it },
                    placeholder = {
                        Text(
                            "[\n  \"What is the main topic?\",\n  \"What are the key conclusions?\"\n]",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
                        )
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(180.dp),
                    textStyle = MaterialTheme.typography.bodySmall,
                    shape = RoundedCornerShape(12.dp)
                )

                Spacer(modifier = Modifier.height(8.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Surface(
                        shape = RoundedCornerShape(8.dp),
                        color = if (parsedQuestions.isNotEmpty()) ChatGPTGreen.copy(alpha = 0.15f) else MaterialTheme.colorScheme.surfaceVariant
                    ) {
                        Text(
                            text = if (parsedQuestions.isNotEmpty()) "${parsedQuestions.size} questions detected" else "0 questions detected",
                            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                            color = if (parsedQuestions.isNotEmpty()) ChatGPTGreen else MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                        )
                    }

                    TextButton(
                        onClick = {
                            rawInput = "[\n  \"What is the main summary?\",\n  \"What are the key findings?\",\n  \"What are the conclusions?\"\n]"
                        }
                    ) {
                        Text("Reset Example", style = MaterialTheme.typography.labelSmall)
                    }
                }
            }
        },
        confirmButton = {
            Button(
                onClick = {
                    if (parsedQuestions.isNotEmpty()) {
                        onStartQueue(parsedQuestions)
                        onDismiss()
                    }
                },
                enabled = parsedQuestions.isNotEmpty(),
                colors = ButtonDefaults.buttonColors(containerColor = ChatGPTGreen)
            ) {
                Text("Start Queue (${parsedQuestions.size})", color = Color.White)
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        }
    )
}

@Composable
fun QuestionQueueBanner(
    queueState: QuestionQueueState,
    onPause: () -> Unit,
    onResume: () -> Unit,
    onStop: () -> Unit
) {
    if (!queueState.isActive) return

    Surface(
        color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.85f),
        shape = RoundedCornerShape(12.dp),
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 6.dp)
    ) {
        Column(
            modifier = Modifier.padding(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(
                        imageVector = if (queueState.isPaused) Icons.Default.PauseCircle else Icons.Default.PlaylistPlay,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = if (queueState.isPaused) "Queue Paused (${queueState.completedCount}/${queueState.totalQuestions})"
                               else "Queue: ${queueState.completedCount + 1}/${queueState.totalQuestions}",
                        style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                        color = MaterialTheme.colorScheme.onPrimaryContainer
                    )
                }

                Row(verticalAlignment = Alignment.CenterVertically) {
                    IconButton(
                        onClick = { if (queueState.isPaused) onResume() else onPause() },
                        modifier = Modifier.size(32.dp)
                    ) {
                        Icon(
                            imageVector = if (queueState.isPaused) Icons.Default.PlayArrow else Icons.Default.Pause,
                            contentDescription = if (queueState.isPaused) "Resume" else "Pause",
                            tint = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.size(18.dp)
                        )
                    }
                    IconButton(
                        onClick = onStop,
                        modifier = Modifier.size(32.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Stop,
                            contentDescription = "Stop Queue",
                            tint = MaterialTheme.colorScheme.error,
                            modifier = Modifier.size(18.dp)
                        )
                    }
                }
            }

            if (!queueState.currentQuestion.isNullOrBlank()) {
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "Current: \"${queueState.currentQuestion}\"",
                    style = MaterialTheme.typography.bodySmall,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.8f)
                )
            }

            Spacer(modifier = Modifier.height(6.dp))
            LinearProgressIndicator(
                progress = {
                    if (queueState.totalQuestions > 0) {
                        queueState.completedCount.toFloat() / queueState.totalQuestions
                    } else 0f
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(4.dp)
                    .clip(RoundedCornerShape(2.dp)),
                color = ChatGPTGreen,
                trackColor = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.2f)
            )
        }
    }
}

// =========================================================================
// CHATGPT-STYLE MESSAGE COMPOSER
// =========================================================================

@Composable
fun ChatGPTComposer(
    textInput: String,
    onTextChanged: (String) -> Unit,
    onSend: () -> Unit,
    isBusy: Boolean,
    onStop: (() -> Unit)? = null,
    selectedDocument: DocumentEntity?,
    onOpenDocumentPicker: () -> Unit,
    onClearSelectedDoc: () -> Unit,
    answerLength: AnswerLength,
    onToggleAnswerLength: () -> Unit
) {
    Surface(
        color = MaterialTheme.colorScheme.surface,
        tonalElevation = 3.dp,
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 8.dp)
        ) {
            if (selectedDocument != null) {
                Surface(
                    shape = RoundedCornerShape(10.dp),
                    color = MaterialTheme.colorScheme.primaryContainer,
                    modifier = Modifier.padding(bottom = 8.dp)
                ) {
                    Row(
                        modifier = Modifier.padding(start = 10.dp, end = 4.dp, top = 4.dp, bottom = 4.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            Icons.Default.Description,
                            contentDescription = null,
                            tint = ChatGPTGreen,
                            modifier = Modifier.size(14.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = selectedDocument.name,
                            style = MaterialTheme.typography.labelMedium,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.widthIn(max = 220.dp)
                        )
                        IconButton(
                            onClick = onClearSelectedDoc,
                            modifier = Modifier.size(28.dp)
                        ) {
                            Icon(
                                Icons.Default.Close,
                                contentDescription = "Remove selected document",
                                modifier = Modifier.size(16.dp)
                            )
                        }
                    }
                }
            }

            // Keep response length available without taking space from the message row.
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 6.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Surface(
                    onClick = onToggleAnswerLength,
                    shape = RoundedCornerShape(12.dp),
                    color = MaterialTheme.colorScheme.surfaceVariant,
                    modifier = Modifier.height(28.dp)
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        val (label, color) = when (answerLength) {
                            AnswerLength.DEFAULT -> "Default" to MaterialTheme.colorScheme.onSurfaceVariant
                            AnswerLength.SHORT -> "Short" to MaterialTheme.colorScheme.onSurface
                            AnswerLength.LONG -> "Long" to ChatGPTGreen
                        }
                        Text(
                            text = "Length: $label",
                            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                            color = color
                        )
                    }
                }
            }

            // Composer Pill Container
            Surface(
                shape = RoundedCornerShape(26.dp),
                color = MaterialTheme.colorScheme.surfaceVariant,
                border = androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f)),
                modifier = Modifier.fillMaxWidth()
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 8.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    IconButton(
                        onClick = onOpenDocumentPicker,
                        modifier = Modifier.size(40.dp)
                    ) {
                        Icon(
                            Icons.Default.Add,
                            contentDescription = "Choose a document",
                            tint = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }

                    TextField(
                        value = textInput,
                        onValueChange = onTextChanged,
                        keyboardOptions = KeyboardOptions(
                            capitalization = KeyboardCapitalization.Sentences,
                            imeAction = ImeAction.Send
                        ),
                        keyboardActions = KeyboardActions(onSend = { if (!isBusy) onSend() }),
                        placeholder = {
                            Text(
                                text = "Message...",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
                            )
                        },
                        colors = TextFieldDefaults.colors(
                            focusedContainerColor = Color.Transparent,
                            unfocusedContainerColor = Color.Transparent,
                            disabledContainerColor = Color.Transparent,
                            focusedIndicatorColor = Color.Transparent,
                            unfocusedIndicatorColor = Color.Transparent,
                            disabledIndicatorColor = Color.Transparent
                        ),
                        minLines = 1,
                        maxLines = 5,
                        modifier = Modifier.weight(1f)
                    )

                    Spacer(modifier = Modifier.width(6.dp))

                    // Circular Send Button
                    val canSend = textInput.isNotBlank() && !isBusy
                    val canStop = isBusy && onStop != null
                    Box(
                        modifier = Modifier
                            .size(36.dp)
                            .clip(CircleShape)
                            .background(
                                if (canSend || canStop) ChatGPTGreen else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.2f)
                            )
                            .clickable(enabled = canSend || canStop) { if (isBusy) onStop?.invoke() else onSend() },
                        contentAlignment = Alignment.Center
                    ) {
                        if (isBusy) {
                            // Stop button: ends the current answer and keeps what was written so far.
                            Box(modifier = Modifier.size(12.dp).background(Color.White, RoundedCornerShape(2.dp)))
                        } else {
                            Icon(
                                imageVector = Icons.Default.Send,
                                contentDescription = "Send",
                                tint = if (canSend) Color.White else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f),
                                modifier = Modifier.size(18.dp)
                            )
                        }
                    }
                }
            }
        }
    }
}

// =========================================================================
// LIBRARY (DOCUMENTS SCREEN)
// =========================================================================

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DocumentsScreen(
    viewModel: MainViewModel,
    navController: NavHostController,
    onOpenDrawer: () -> Unit
) {
    val coroutineScope = rememberCoroutineScope()
    val documents by viewModel.documents.collectAsState()
    val importState by viewModel.importState.collectAsState()
    var searchQuery by remember { mutableStateOf("") }
    var docToDelete by remember { mutableStateOf<DocumentEntity?>(null) }

    val filePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            viewModel.importDocument(it)
        }
    }

    if (docToDelete != null) {
        val target = docToDelete!!
        AlertDialog(
            onDismissRequest = { docToDelete = null },
            title = { Text("Delete Document") },
            text = { Text("Are you sure you want to delete \"${target.name}\"? All indexed chunks and embeddings for this document will be removed.") },
            confirmButton = {
                TextButton(
                    onClick = {
                        viewModel.deleteDocument(target.id)
                        docToDelete = null
                    }
                ) {
                    Text("Delete", color = MaterialTheme.colorScheme.error)
                }
            },
            dismissButton = {
                TextButton(onClick = { docToDelete = null }) {
                    Text("Cancel")
                }
            }
        )
    }

    val filteredDocs = remember(documents, searchQuery) {
        if (searchQuery.isBlank()) documents
        else documents.filter { it.name.contains(searchQuery, ignoreCase = true) }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Library", style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold)) },
                navigationIcon = {
                    IconButton(onClick = onOpenDrawer) {
                        Icon(Icons.Default.Menu, contentDescription = "Menu")
                    }
                },
                actions = {
                    IconButton(onClick = { filePicker.launch("*/*") }) {
                        Icon(Icons.Default.Add, contentDescription = "Import Document", tint = ChatGPTGreen)
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            )
        },
        floatingActionButton = {
            FloatingActionButton(
                onClick = { filePicker.launch("*/*") },
                containerColor = ChatGPTGreen,
                contentColor = Color.White
            ) {
                Icon(Icons.Default.Add, contentDescription = "Import Document")
            }
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(horizontal = 16.dp)
        ) {
            // Live Import Progress Banner
            when (val state = importState) {
                is DocumentRepository.ImportState.Reading -> {
                    Surface(
                        shape = RoundedCornerShape(12.dp),
                        color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.7f),
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 6.dp)
                    ) {
                        Row(
                            modifier = Modifier.padding(12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            CircularProgressIndicator(strokeWidth = 2.dp, modifier = Modifier.size(16.dp), color = ChatGPTGreen)
                            Spacer(modifier = Modifier.width(10.dp))
                            Text(
                                text = "Reading \"${state.documentName}\"...",
                                style = MaterialTheme.typography.bodySmall,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                        }
                    }
                }
                is DocumentRepository.ImportState.Chunking -> {
                    Surface(
                        shape = RoundedCornerShape(12.dp),
                        color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.7f),
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 6.dp)
                    ) {
                        Row(
                            modifier = Modifier.padding(12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            CircularProgressIndicator(strokeWidth = 2.dp, modifier = Modifier.size(16.dp), color = ChatGPTGreen)
                            Spacer(modifier = Modifier.width(10.dp))
                            Text(
                                text = "Chunking \"${state.documentName}\" (${state.pageCount} pages)...",
                                style = MaterialTheme.typography.bodySmall,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                        }
                    }
                }
                is DocumentRepository.ImportState.Embedding -> {
                    Surface(
                        shape = RoundedCornerShape(12.dp),
                        color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.7f),
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 6.dp)
                    ) {
                        Column(modifier = Modifier.padding(12.dp)) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text(
                                    text = "Indexing \"${state.documentName}\"",
                                    style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.SemiBold),
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis,
                                    modifier = Modifier.weight(1f)
                                )
                                Text(
                                    text = "${state.current}/${state.total} (${state.percent}%)",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = ChatGPTGreen
                                )
                            }
                            Spacer(modifier = Modifier.height(6.dp))
                            LinearProgressIndicator(
                                progress = { state.percent / 100f },
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(4.dp)
                                    .clip(RoundedCornerShape(2.dp)),
                                color = ChatGPTGreen,
                                trackColor = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.2f)
                            )
                        }
                    }
                }
                is DocumentRepository.ImportState.Saving -> {
                    Surface(
                        shape = RoundedCornerShape(12.dp),
                        color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.7f),
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 6.dp)
                    ) {
                        Row(
                            modifier = Modifier.padding(12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            CircularProgressIndicator(strokeWidth = 2.dp, modifier = Modifier.size(16.dp), color = ChatGPTGreen)
                            Spacer(modifier = Modifier.width(10.dp))
                            Text(
                                text = "Saving chunks & vectors for \"${state.documentName}\"...",
                                style = MaterialTheme.typography.bodySmall,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                        }
                    }
                }
                is DocumentRepository.ImportState.Error -> {
                    Surface(
                        shape = RoundedCornerShape(12.dp),
                        color = MaterialTheme.colorScheme.errorContainer,
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 6.dp)
                    ) {
                        Row(
                            modifier = Modifier.padding(12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(Icons.Default.Error, contentDescription = null, tint = MaterialTheme.colorScheme.error, modifier = Modifier.size(16.dp))
                            Spacer(modifier = Modifier.width(10.dp))
                            Text(
                                text = "Import failed: ${state.message}",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onErrorContainer
                            )
                        }
                    }
                }
                else -> {}
            }

            // Search Bar
            OutlinedTextField(
                value = searchQuery,
                onValueChange = { searchQuery = it },
                placeholder = { Text("Search documents...") },
                leadingIcon = { Icon(Icons.Default.Search, contentDescription = null, modifier = Modifier.size(18.dp)) },
                shape = RoundedCornerShape(14.dp),
                singleLine = true,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 10.dp)
            )

            if (filteredDocs.isEmpty()) {
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth(),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Icon(
                            imageVector = Icons.Default.Folder,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f),
                            modifier = Modifier.size(48.dp)
                        )
                        Spacer(modifier = Modifier.height(10.dp))
                        Text(
                            text = if (searchQuery.isNotBlank()) "No matching documents" else "No documents in library",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            } else {
                LazyColumn(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                    contentPadding = PaddingValues(vertical = 8.dp)
                ) {
                    items(filteredDocs, key = { it.id }) { doc ->
                        DocumentCardItem(
                            document = doc,
                            onAsk = {
                                coroutineScope.launch {
                                    val newId = viewModel.startConversation("Chat: ${doc.name.take(20)}", listOf(doc.id))
                                    navController.navigate("chat?convId=$newId") {
                                        launchSingleTop = true
                                    }
                                }
                            },
                            onViewDetail = {
                                navController.navigate("documentDetail/${doc.id}")
                            },
                            onDelete = { docToDelete = doc },
                            onReprocess = { viewModel.reprocessDocument(doc) }
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun DocumentCardItem(
    document: DocumentEntity,
    onAsk: () -> Unit,
    onViewDetail: () -> Unit,
    onDelete: () -> Unit,
    onReprocess: () -> Unit = {}
) {
    val isReady = document.status == "COMPLETED" || document.status == "READY"
    val isProcessing = document.status == "PROCESSING"
    val isFailed = document.status == "FAILED"

    val statusColor = when {
        isReady -> ChatGPTGreen
        isFailed -> MaterialTheme.colorScheme.error
        else -> Color(0xFFFF9800)
    }

    Surface(
        onClick = onViewDetail,
        shape = RoundedCornerShape(16.dp),
        color = MaterialTheme.colorScheme.surfaceVariant,
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(36.dp)
                        .clip(RoundedCornerShape(8.dp))
                        .background(statusColor.copy(alpha = 0.15f)),
                    contentAlignment = Alignment.Center
                ) {
                    if (isProcessing) {
                        CircularProgressIndicator(
                            strokeWidth = 2.dp,
                            color = statusColor,
                            modifier = Modifier.size(18.dp)
                        )
                    } else {
                        Icon(
                            imageVector = if (isFailed) Icons.Default.Error else Icons.Default.Description,
                            contentDescription = null,
                            tint = statusColor,
                            modifier = Modifier.size(20.dp)
                        )
                    }
                }

                Spacer(modifier = Modifier.width(12.dp))

                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = document.name,
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                    val statusText = when {
                        isReady -> "${document.pageCount} pages • Indexed"
                        isFailed -> "Processing failed"
                        else -> if (document.pageCount > 0) "${document.pageCount} pages • Processing..." else "Reading & processing..."
                    }
                    Text(
                        text = statusText,
                        style = MaterialTheme.typography.labelSmall,
                        color = statusColor
                    )
                }

                IconButton(onClick = onDelete, modifier = Modifier.size(28.dp)) {
                    Icon(
                        imageVector = Icons.Default.Delete,
                        contentDescription = "Delete",
                        tint = MaterialTheme.colorScheme.error.copy(alpha = 0.7f),
                        modifier = Modifier.size(18.dp)
                    )
                }
            }

            if (isProcessing) {
                Spacer(modifier = Modifier.height(8.dp))
                LinearProgressIndicator(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(3.dp)
                        .clip(RoundedCornerShape(1.5.dp)),
                    color = ChatGPTGreen,
                    trackColor = ChatGPTGreen.copy(alpha = 0.15f)
                )
            }

            Spacer(modifier = Modifier.height(10.dp))

            // Action Row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End,
                verticalAlignment = Alignment.CenterVertically
            ) {
                if (isFailed) {
                    OutlinedButton(
                        onClick = onReprocess,
                        shape = RoundedCornerShape(12.dp),
                        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp)
                    ) {
                        Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(14.dp))
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Retry", style = MaterialTheme.typography.labelMedium)
                    }
                    Spacer(modifier = Modifier.width(8.dp))
                }

                TextButton(onClick = onViewDetail) {
                    Text("Details", style = MaterialTheme.typography.labelMedium)
                }
                Spacer(modifier = Modifier.width(8.dp))
                Button(
                    onClick = onAsk,
                    enabled = isReady,
                    shape = RoundedCornerShape(12.dp),
                    contentPadding = PaddingValues(horizontal = 14.dp, vertical = 6.dp)
                ) {
                    Icon(Icons.Default.Chat, contentDescription = null, modifier = Modifier.size(14.dp))
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("Ask", style = MaterialTheme.typography.labelMedium)
                }
            }
        }
    }
}

// =========================================================================
// DOCUMENT DETAIL SCREEN
// =========================================================================

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DocumentDetailScreen(
    docId: Long,
    viewModel: MainViewModel,
    navController: NavHostController
) {
    val coroutineScope = rememberCoroutineScope()
    val documents by viewModel.documents.collectAsState()
    val doc = documents.firstOrNull { it.id == docId }

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
                .padding(20.dp)
        ) {
            if (doc == null) {
                Text("Document not found.")
            } else {
                Surface(
                    shape = RoundedCornerShape(16.dp),
                    color = MaterialTheme.colorScheme.surfaceVariant,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(18.dp)) {
                        Text(
                            text = doc.name,
                            style = MaterialTheme.typography.headlineSmall.copy(fontWeight = FontWeight.Bold)
                        )
                        Spacer(modifier = Modifier.height(14.dp))
                        DetailRow(label = "Status", value = doc.status)
                        DetailRow(label = "Total Pages", value = "${doc.pageCount}")
                        DetailRow(label = "MIME Type", value = doc.type)
                        DetailRow(label = "Storage URI", value = doc.uri)
                    }
                }

                Spacer(modifier = Modifier.height(24.dp))

                val isReady = doc.status == "COMPLETED" || doc.status == "READY"
                if (isReady) {
                    Button(
                        onClick = {
                            coroutineScope.launch {
                                val newId = viewModel.startConversation("Chat: ${doc.name.take(20)}", listOf(doc.id))
                                navController.navigate("chat?convId=$newId") {
                                    launchSingleTop = true
                                }
                            }
                        },
                        shape = RoundedCornerShape(16.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Icon(Icons.Default.Chat, contentDescription = null)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Start Conversation with this Document")
                    }
                } else {
                    Button(
                        onClick = {
                            viewModel.reprocessDocument(doc)
                        },
                        shape = RoundedCornerShape(16.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Icon(Icons.Default.Refresh, contentDescription = null)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Re-index / Retry Document Processing")
                    }
                }
            }
        }
    }
}

@Composable
fun DetailRow(label: String, value: String) {
    Column(modifier = Modifier.padding(vertical = 4.dp)) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold)
        )
    }
}

// =========================================================================
// SETTINGS & MODEL MANAGEMENT SCREEN
// =========================================================================

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    viewModel: MainViewModel,
    navController: NavHostController,
    onOpenDrawer: () -> Unit
) {
    val context = LocalContext.current
    val themeMode by viewModel.themeMode.collectAsState()
    val installedModels by viewModel.installedModels.collectAsState()
    val activeModel by viewModel.activeLlmModelItem.collectAsState()
    val llmLoadState by viewModel.llmLoadState.collectAsState()
    val downloadStatuses by viewModel.downloadStatuses.collectAsState()
    val embeddingStatuses by viewModel.embeddingStatuses.collectAsState()
    val activeEmbeddingId by viewModel.activeEmbeddingId.collectAsState()
    val embeddingSwitching by viewModel.embeddingSwitchInProgress.collectAsState()
    val importProgress by viewModel.modelImportProgress.collectAsState()
    val importError by viewModel.modelImportError.collectAsState()
    val compatibilityPreferred by viewModel.compatibilityPreferred.collectAsState()
    val speedMode by viewModel.speedMode.collectAsState()

    var modelToDelete by remember { mutableStateOf<ModelItem?>(null) }
    var embeddingToActivate by remember { mutableStateOf<EmbeddingModelSpec?>(null) }
    var embeddingToDelete by remember { mutableStateOf<EmbeddingModelSpec?>(null) }

    val importModelLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) viewModel.importLlmModel(uri)
    }
    // Android 13+: allow the download progress notification (downloads continue in the background either way).
    val notificationPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { }
    val startDownload: (String) -> Unit = { id ->
        if (android.os.Build.VERSION.SDK_INT >= 33 &&
            androidx.core.content.ContextCompat.checkSelfPermission(context, android.Manifest.permission.POST_NOTIFICATIONS) !=
            android.content.pm.PackageManager.PERMISSION_GRANTED
        ) {
            notificationPermission.launch(android.Manifest.permission.POST_NOTIFICATIONS)
        }
        viewModel.downloadModel(id)
    }

    modelToDelete?.let { target ->
        AlertDialog(
            onDismissRequest = { modelToDelete = null },
            title = { Text("Delete model") },
            text = { Text("Delete '${target.displayName}' (${target.sizeBytes / 1_000_000} MB)? It is unloaded from memory first, so its storage is freed right away.") },
            confirmButton = {
                TextButton(onClick = {
                    viewModel.deleteModelItem(target)
                    modelToDelete = null
                }) { Text("Delete", color = MaterialTheme.colorScheme.error) }
            },
            dismissButton = { TextButton(onClick = { modelToDelete = null }) { Text("Cancel") } }
        )
    }

    embeddingToActivate?.let { spec ->
        val count = viewModel.documentsToReindex()
        AlertDialog(
            onDismissRequest = { embeddingToActivate = null },
            title = { Text("Switch search model?") },
            text = {
                Text(
                    "The app first checks that ${spec.name} works on this phone." +
                        if (count > 0) " Then your $count document(s) are re-indexed in the background; they can't be searched until that finishes." else ""
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    viewModel.activateEmbeddingModel(spec.id)
                    embeddingToActivate = null
                }) { Text("Switch") }
            },
            dismissButton = { TextButton(onClick = { embeddingToActivate = null }) { Text("Cancel") } }
        )
    }

    embeddingToDelete?.let { spec ->
        AlertDialog(
            onDismissRequest = { embeddingToDelete = null },
            title = { Text("Delete search model") },
            text = { Text("Delete ${spec.name} (${spec.downloadSize})?") },
            confirmButton = {
                TextButton(onClick = {
                    viewModel.deleteEmbeddingModel(spec.id)
                    embeddingToDelete = null
                }) { Text("Delete", color = MaterialTheme.colorScheme.error) }
            },
            dismissButton = { TextButton(onClick = { embeddingToDelete = null }) { Text("Cancel") } }
        )
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Settings & Models", style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold)) },
                navigationIcon = {
                    IconButton(onClick = onOpenDrawer) { Icon(Icons.Default.Menu, contentDescription = "Menu") }
                }
            )
        }
    ) { padding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(horizontal = 20.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(18.dp)
        ) {
            // Appearance
            item {
                SectionTitle("Appearance")
                Surface(shape = RoundedCornerShape(16.dp), color = MaterialTheme.colorScheme.surfaceVariant, modifier = Modifier.fillMaxWidth()) {
                    Row(
                        modifier = Modifier.fillMaxWidth().padding(14.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text("Theme Mode", style = MaterialTheme.typography.bodyMedium)
                        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            MainViewModel.ThemeMode.values().forEach { mode ->
                                FilterChip(
                                    selected = themeMode == mode,
                                    onClick = { viewModel.setThemeMode(mode) },
                                    label = { Text(mode.name.lowercase().replaceFirstChar { it.uppercase() }) }
                                )
                            }
                        }
                    }
                }
            }

            // Active language model: real load state
            item {
                SectionTitle("Language model (answers)")
                Surface(shape = RoundedCornerShape(16.dp), color = MaterialTheme.colorScheme.surfaceVariant, modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    text = activeModel?.displayName ?: "No model installed",
                                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                                )
                                val detail = when (val s = llmLoadState) {
                                    is LlmLoadState.Ready -> "Context ${s.contextSize} tokens • CPU ${s.cpuVariant}" +
                                        if (s.performance.isNotBlank()) "\n${s.performance}" else ""
                                    is LlmLoadState.Failed -> s.reason
                                    is LlmLoadState.Loading -> "Loading into memory…"
                                    is LlmLoadState.NotLoaded -> "Loads when you ask a question"
                                    LlmLoadState.NoModel -> "Download a model below to enable offline AI chat."
                                }
                                Text(
                                    text = detail,
                                    style = MaterialTheme.typography.labelSmall,
                                    color = if (llmLoadState is LlmLoadState.Failed) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                            val (label, color) = llmStatusLabel(llmLoadState)
                            StatusPill(label, color)
                        }
                        if (llmLoadState is LlmLoadState.Loading) {
                            Spacer(modifier = Modifier.height(8.dp))
                            LinearProgressIndicator(modifier = Modifier.fillMaxWidth(), color = ChatGPTGreen)
                        }
                        if (llmLoadState is LlmLoadState.Failed) {
                            Spacer(modifier = Modifier.height(8.dp))
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                OutlinedButton(onClick = { viewModel.retryLoadLlm() }) { Text("Retry") }
                                activeModel?.let { m -> OutlinedButton(onClick = { modelToDelete = m }) { Text("Delete") } }
                            }
                        }
                    }
                }
            }

            // Language model catalog
            item {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text("Available language models", style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold))
                    TextButton(onClick = { importModelLauncher.launch(arrayOf("*/*")) }, enabled = importProgress == null) {
                        Icon(Icons.Default.Add, contentDescription = null, modifier = Modifier.size(16.dp))
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Import GGUF")
                    }
                }
                importProgress?.let { p ->
                    Text("Importing ${p.filename}: ${p.percent}%", style = MaterialTheme.typography.bodySmall)
                    LinearProgressIndicator(progress = { p.percent / 100f }, modifier = Modifier.fillMaxWidth(), color = ChatGPTGreen)
                }
                importError?.let { err -> Text(err, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }
                Spacer(modifier = Modifier.height(8.dp))
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    AvailableLlmModels.list.forEach { model ->
                        val installed = installedModels.firstOrNull { it.filename.equals(model.fileName, ignoreCase = true) }
                        val isActive = installed != null && activeModel?.id == installed.id
                        ModelCard(
                            title = model.name,
                            subtitle = "${model.parameterSize} params • ${model.downloadSize} • ${model.ramRequirement}",
                            badge = model.badge,
                            description = model.description,
                            state = cardState(
                                installed = installed != null,
                                active = isActive,
                                error = installed?.takeIf { it.status == ModelStatus.ERROR }?.let { it.errorMessage ?: "Damaged or unsupported file" },
                                download = downloadStatuses[model.id],
                                activeLabel = llmStatusLabel(llmLoadState).first
                            ),
                            onDownload = { startDownload(model.id) },
                            onCancel = { viewModel.cancelDownload(model.id) },
                            onDismiss = { viewModel.dismissDownloadStatus(model.id) },
                            onUse = { installed?.let { viewModel.selectLlmModel(it) } },
                            onDelete = { installed?.let { modelToDelete = it } }
                        )
                    }
                }
            }

            // Imported models that are not in the catalog
            val customModels = installedModels.filter { inst ->
                AvailableLlmModels.list.none { it.fileName.equals(inst.filename, ignoreCase = true) }
            }
            if (customModels.isNotEmpty()) {
                item {
                    Text("Imported models", style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold))
                    Spacer(modifier = Modifier.height(8.dp))
                    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        customModels.forEach { m ->
                            ModelCard(
                                title = m.displayName.ifBlank { m.filename },
                                subtitle = "${m.sizeBytes / 1_000_000} MB • ${m.quantization.ifBlank { "GGUF" }}",
                                badge = null,
                                description = null,
                                state = cardState(
                                    installed = true,
                                    active = activeModel?.id == m.id,
                                    error = m.takeIf { it.status == ModelStatus.ERROR }?.let { it.errorMessage ?: "Damaged or unsupported file" },
                                    download = null,
                                    activeLabel = llmStatusLabel(llmLoadState).first
                                ),
                                onDownload = {},
                                onCancel = {},
                                onDismiss = {},
                                onUse = { viewModel.selectLlmModel(m) },
                                onDelete = { modelToDelete = m }
                            )
                        }
                    }
                }
            }

            // Search (embedding) models
            item {
                SectionTitle("Search model (embeddings)")
                Text(
                    "Used to index documents and find the passages each answer is based on. If search does not work on your phone, switch to another model.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Spacer(modifier = Modifier.height(8.dp))
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    AvailableEmbeddingModels.list.forEach { spec ->
                        val status = embeddingStatuses[spec.id]
                        val isActive = spec.id == activeEmbeddingId
                        val switching = embeddingSwitching == spec.id
                        val installed = status != null && status != EmbeddingModelStatus.NotInstalled
                        val activeLabel = when {
                            status is EmbeddingModelStatus.Loading || switching -> "Checking…"
                            status is EmbeddingModelStatus.Ready -> "Ready"
                            else -> "Active"
                        }
                        ModelCard(
                            title = spec.name,
                            subtitle = "${spec.backendLabel} • ${spec.dimension}-dim • " + if (spec.isBundled) "Built-in" else spec.downloadSize,
                            badge = spec.badge,
                            description = spec.description,
                            state = when {
                                status is EmbeddingModelStatus.Unsupported -> CardState.Failed("Not supported on this device: ${status.reason}", canRetry = true, canDelete = !spec.isBundled && !isActive, retryDownloads = false)
                                switching -> CardState.Busy("Checking that it works on this phone…")
                                else -> cardState(
                                    installed = installed,
                                    active = isActive,
                                    error = null,
                                    download = downloadStatuses[spec.id],
                                    activeLabel = activeLabel,
                                    deletable = !spec.isBundled
                                )
                            },
                            onDownload = { startDownload(spec.id) },
                            onCancel = { viewModel.cancelDownload(spec.id) },
                            onDismiss = { viewModel.dismissDownloadStatus(spec.id) },
                            onUse = { if (!isActive) embeddingToActivate = spec else viewModel.activateEmbeddingModel(spec.id) },
                            onDelete = { embeddingToDelete = spec }
                        )
                    }
                }
            }

            // Answer speed: how long the phone may spend reading the sources (adapts to measured speed)
            item {
                SectionTitle("Answer speed")
                Surface(shape = RoundedCornerShape(16.dp), color = MaterialTheme.colorScheme.surfaceVariant, modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            com.siddu.offlinedocai.ai.llm.SpeedMode.entries.forEach { mode ->
                                FilterChip(
                                    selected = speedMode == mode,
                                    onClick = { viewModel.setSpeedMode(mode) },
                                    label = { Text(mode.label) }
                                )
                            }
                        }
                        Spacer(modifier = Modifier.height(6.dp))
                        Text(
                            "The app measures this phone's speed and sends the model only as much document text as it can read in about " +
                                "${speedMode.targetPrefillSeconds} s. Fastest = quicker answers from fewer passages; Thorough = more passages, longer wait. " +
                                "Uses fewer CPU cores automatically when the phone is hot or in battery saver.",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        TextButton(onClick = { viewModel.remeasureSpeed() }) { Text("Re-measure this phone's speed") }
                    }
                }
            }

            // Advanced
            item {
                SectionTitle("Advanced")
                Surface(shape = RoundedCornerShape(16.dp), color = MaterialTheme.colorScheme.surfaceVariant, modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text("Compatibility mode", style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold))
                                Text(
                                    "Uses the most basic CPU instructions for the local models. Slower, but avoids crashes on some phones. Applies after restarting the app.",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                            Switch(checked = compatibilityPreferred || viewModel.compatibilityModeActive, onCheckedChange = { viewModel.setCompatibilityMode(it) })
                        }
                        viewModel.compatibilityReason?.let {
                            Spacer(modifier = Modifier.height(6.dp))
                            Text(it, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.error)
                        }
                        Spacer(modifier = Modifier.height(6.dp))
                        Text(
                            "CPU kernels in use: ${viewModel.cpuVariant.ifBlank { "not loaded yet" }}",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun SectionTitle(text: String) {
    Text(text = text, style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold))
    Spacer(modifier = Modifier.height(8.dp))
}

@Composable
private fun StatusPill(label: String, color: Color) {
    Surface(shape = RoundedCornerShape(12.dp), color = color.copy(alpha = 0.15f)) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
            color = color,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
        )
    }
}

/** Label and colour for the language model's real load state. */
@Composable
fun llmStatusLabel(state: LlmLoadState): Pair<String, Color> = when (state) {
    is LlmLoadState.Ready -> "Ready" to ChatGPTGreen
    is LlmLoadState.Loading -> "Loading…" to MaterialTheme.colorScheme.tertiary
    is LlmLoadState.Failed -> "Failed" to MaterialTheme.colorScheme.error
    is LlmLoadState.NotLoaded -> "Installed" to MaterialTheme.colorScheme.primary
    LlmLoadState.NoModel -> "No model" to MaterialTheme.colorScheme.outline
}

/** Everything a model card can show; exactly one applies at a time. */
sealed class CardState {
    object NotInstalled : CardState()
    data class Queued(val note: String?) : CardState()
    data class Downloading(val percent: Int, val bytes: Long, val total: Long) : CardState()
    data class Busy(val message: String) : CardState()
    data class Installed(val deletable: Boolean) : CardState()
    data class Active(val label: String, val deletable: Boolean) : CardState()
    data class Failed(val message: String, val canRetry: Boolean, val canDelete: Boolean, val retryDownloads: Boolean = true) : CardState()
}

private fun cardState(
    installed: Boolean,
    active: Boolean,
    error: String?,
    download: DownloadStatus?,
    activeLabel: String,
    deletable: Boolean = true
): CardState = when {
    download is DownloadStatus.Queued -> CardState.Queued(download.note)
    download is DownloadStatus.Downloading -> CardState.Downloading(download.percent, download.bytes, download.total)
    download is DownloadStatus.Verifying -> CardState.Busy("Verifying the download…")
    download is DownloadStatus.Failed && !installed -> CardState.Failed(download.message, canRetry = true, canDelete = false)
    error != null -> CardState.Failed(error, canRetry = false, canDelete = true)
    active -> CardState.Active(activeLabel, deletable)
    installed -> CardState.Installed(deletable)
    else -> CardState.NotInstalled
}

@Composable
fun ModelCard(
    title: String,
    subtitle: String,
    badge: String?,
    description: String?,
    state: CardState,
    onDownload: () -> Unit,
    onCancel: () -> Unit,
    onDismiss: () -> Unit,
    onUse: () -> Unit,
    onDelete: () -> Unit
) {
    val isActive = state is CardState.Active
    Card(
        colors = CardDefaults.cardColors(
            containerColor = when {
                isActive -> MaterialTheme.colorScheme.primaryContainer
                state is CardState.Installed -> MaterialTheme.colorScheme.surfaceVariant
                else -> MaterialTheme.colorScheme.surface
            }
        ),
        shape = RoundedCornerShape(12.dp),
        border = if (isActive) androidx.compose.foundation.BorderStroke(2.dp, ChatGPTGreen) else androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(text = title, fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleSmall)
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(text = subtitle, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                if (badge != null) StatusPill(badge, MaterialTheme.colorScheme.primary)
            }
            if (!description.isNullOrBlank()) {
                Spacer(modifier = Modifier.height(6.dp))
                Text(text = description, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Spacer(modifier = Modifier.height(10.dp))
            when (state) {
                CardState.NotInstalled -> Button(onClick = onDownload, shape = RoundedCornerShape(8.dp)) {
                    Icon(Icons.Default.Download, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("Download")
                }
                is CardState.Queued -> Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(state.note ?: "Waiting to download…", style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
                    TextButton(onClick = onCancel) { Text("Cancel") }
                }
                is CardState.Downloading -> Column {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            "Downloading ${state.percent}% (${state.bytes / 1_000_000} of ${state.total / 1_000_000} MB)",
                            style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier.weight(1f)
                        )
                        TextButton(onClick = onCancel) { Text("Cancel") }
                    }
                    LinearProgressIndicator(progress = { state.percent / 100f }, modifier = Modifier.fillMaxWidth(), color = ChatGPTGreen)
                }
                is CardState.Busy -> Column {
                    Text(state.message, style = MaterialTheme.typography.bodySmall)
                    Spacer(modifier = Modifier.height(4.dp))
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth(), color = ChatGPTGreen)
                }
                is CardState.Installed -> Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                    Button(onClick = onUse, shape = RoundedCornerShape(8.dp)) { Text("Use this model") }
                    if (state.deletable) TextButton(onClick = onDelete) { Text("Delete", color = MaterialTheme.colorScheme.error) }
                }
                is CardState.Active -> Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                    StatusPill("● Active · ${state.label}", ChatGPTGreen)
                    Spacer(modifier = Modifier.weight(1f))
                    if (state.deletable) TextButton(onClick = onDelete) { Text("Delete", color = MaterialTheme.colorScheme.error) }
                }
                is CardState.Failed -> Column {
                    Text(state.message, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        if (state.canRetry) TextButton(onClick = { if (state.retryDownloads) { onDismiss(); onDownload() } else onUse() }) { Text("Retry") }
                        if (state.canDelete) TextButton(onClick = onDelete) { Text("Delete", color = MaterialTheme.colorScheme.error) }
                        if (!state.canRetry) TextButton(onClick = onDismiss) { Text("Dismiss") }
                    }
                }
            }
        }
    }
}
