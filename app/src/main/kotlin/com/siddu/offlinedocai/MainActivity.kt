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
import com.siddu.offlinedocai.ai.embeddings.AvailableEmbeddingModel
import com.siddu.offlinedocai.ai.embeddings.AvailableEmbeddingModels
import com.siddu.offlinedocai.ai.llm.AvailableLlmModel
import com.siddu.offlinedocai.ai.llm.AvailableLlmModels
import com.siddu.offlinedocai.ai.export.ChatExporter
import com.siddu.offlinedocai.ui.components.MarkdownMessage
import com.siddu.offlinedocai.domain.repository.DownloadState
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
                Box(
                    modifier = Modifier
                        .size(8.dp)
                        .clip(CircleShape)
                        .background(ChatGPTGreen)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = activeModel?.displayName ?: "Qwen 2.5 1.5B",
                    style = MaterialTheme.typography.labelMedium,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.weight(1f)
                )
                Text(
                    text = "Local",
                    style = MaterialTheme.typography.labelSmall,
                    color = ChatGPTGreen,
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
    var chatMenuExpanded by remember { mutableStateOf(false) }

    val listState = rememberLazyListState()

    // Auto-scroll to bottom on new message
    LaunchedEffect(messages.size, chatState) {
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(messages.size - 1)
        }
    }

    val isBusy = chatState is ChatRepository.ChatState.Thinking ||
        chatState is ChatRepository.ChatState.Retrieving ||
        chatState is ChatRepository.ChatState.Generating
    val documentPickerState = rememberModalBottomSheetState(skipPartiallyExpanded = true)

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
                                        text = "All documents",
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
                            AssistantMessageItem(text = msg.content)
                        }
                    }

                    // Generation Indicator State
                    if (isBusy) {
                        item {
                            AssistantThinkingItem()
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
                selectedDocument = documents.firstOrNull { it.id == selectedDocId },
                onOpenDocumentPicker = { showDocumentPicker = true },
                onClearSelectedDoc = { selectedDocId = null },
                answerLength = selectedAnswerLength,
                onToggleAnswerLength = {
                    val next = if (selectedAnswerLength == AnswerLength.SHORT) AnswerLength.LONG else AnswerLength.SHORT
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
fun AssistantMessageItem(text: String) {
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
        MarkdownMessage(
            text = text,
            modifier = Modifier.weight(1f),
            isAssistant = true
        )
    }
}

@Composable
fun AssistantThinkingItem() {
    val statusLabel = "Thinking"

    // Animated dots
    var dotCount by remember { mutableIntStateOf(0) }
    LaunchedEffect(Unit) {
        while (true) {
            kotlinx.coroutines.delay(500)
            dotCount = (dotCount + 1) % 4
        }
    }
    val dots = ".".repeat(dotCount)

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
            text = "$statusLabel$dots",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
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
                        Text(
                            text = if (answerLength == AnswerLength.SHORT) "Short" else "Long",
                            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                            color = if (answerLength == AnswerLength.LONG) ChatGPTGreen else MaterialTheme.colorScheme.onSurface
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
                    Box(
                        modifier = Modifier
                            .size(36.dp)
                            .clip(CircleShape)
                            .background(
                                if (canSend) ChatGPTGreen else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.2f)
                            )
                            .clickable(enabled = canSend) { onSend() },
                        contentAlignment = Alignment.Center
                    ) {
                        if (isBusy) {
                            CircularProgressIndicator(
                                strokeWidth = 2.dp,
                                color = Color.White,
                                modifier = Modifier.size(16.dp)
                            )
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
                            onDelete = { docToDelete = doc }
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
    onDelete: () -> Unit
) {
    val isReady = document.status == "COMPLETED" || document.status == "READY"

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
                        .background(if (isReady) ChatGPTGreen.copy(alpha = 0.15f) else Color(0xFFFF9800).copy(alpha = 0.15f)),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Default.Description,
                        contentDescription = null,
                        tint = if (isReady) ChatGPTGreen else Color(0xFFFF9800),
                        modifier = Modifier.size(20.dp)
                    )
                }

                Spacer(modifier = Modifier.width(12.dp))

                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = document.name,
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                    Text(
                        text = "${document.pageCount} pages • ${if (isReady) "Indexed" else "Processing..."}",
                        style = MaterialTheme.typography.labelSmall,
                        color = if (isReady) ChatGPTGreen else Color(0xFFFF9800)
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

            Spacer(modifier = Modifier.height(10.dp))

            // Action Row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End,
                verticalAlignment = Alignment.CenterVertically
            ) {
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
    val themeMode by viewModel.themeMode.collectAsState()
    val installedModels by viewModel.installedModels.collectAsState()
    val activeModel by viewModel.activeLlmModelItem.collectAsState()
    val isEmbedReady by viewModel.isEmbeddingModelDownloaded.collectAsState()

    val downloadedLlmModels by viewModel.downloadedLlmModels.collectAsState()
    val selectedLlmModel by viewModel.selectedLlmModel.collectAsState()
    val currentlyDownloadingLlm by viewModel.currentlyDownloadingModel.collectAsState()
    val llmDownloadState by viewModel.llmDownloadState.collectAsState()

    val downloadedEmbeddingModels by viewModel.downloadedEmbeddingModels.collectAsState()
    val selectedEmbeddingModel by viewModel.selectedEmbeddingModel.collectAsState()
    val currentlyDownloadingEmbed by viewModel.currentlyDownloadingEmbeddingModel.collectAsState()
    val embeddingDownloadState by viewModel.embeddingDownloadState.collectAsState()

    var modelToDelete by remember { mutableStateOf<ModelItem?>(null) }

    val importModelLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            viewModel.importLlmModel(uri)
        }
    }

    if (modelToDelete != null) {
        val target = modelToDelete!!
        AlertDialog(
            onDismissRequest = { modelToDelete = null },
            title = { Text("Remove Model") },
            text = { Text("Are you sure you want to remove '${target.filename}' (${target.sizeBytes / (1024 * 1024)} MB)?") },
            confirmButton = {
                TextButton(
                    onClick = {
                        viewModel.deleteModelItem(target)
                        modelToDelete = null
                    }
                ) {
                    Text("Remove", color = MaterialTheme.colorScheme.error)
                }
            },
            dismissButton = {
                TextButton(onClick = { modelToDelete = null }) {
                    Text("Cancel")
                }
            }
        )
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Settings & Models", style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold)) },
                navigationIcon = {
                    IconButton(onClick = onOpenDrawer) {
                        Icon(Icons.Default.Menu, contentDescription = "Menu")
                    }
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
            // Theme Section
            item {
                Text(
                    text = "Appearance",
                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                )
                Spacer(modifier = Modifier.height(8.dp))
                Surface(
                    shape = RoundedCornerShape(16.dp),
                    color = MaterialTheme.colorScheme.surfaceVariant,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(14.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text("Theme Mode", style = MaterialTheme.typography.bodyMedium)
                            Row {
                                FilterChip(
                                    selected = themeMode == MainViewModel.ThemeMode.SYSTEM,
                                    onClick = { viewModel.setThemeMode(MainViewModel.ThemeMode.SYSTEM) },
                                    label = { Text("System") }
                                )
                                Spacer(modifier = Modifier.width(6.dp))
                                FilterChip(
                                    selected = themeMode == MainViewModel.ThemeMode.LIGHT,
                                    onClick = { viewModel.setThemeMode(MainViewModel.ThemeMode.LIGHT) },
                                    label = { Text("Light") }
                                )
                                Spacer(modifier = Modifier.width(6.dp))
                                FilterChip(
                                    selected = themeMode == MainViewModel.ThemeMode.DARK,
                                    onClick = { viewModel.setThemeMode(MainViewModel.ThemeMode.DARK) },
                                    label = { Text("Dark") }
                                )
                            }
                        }
                    }
                }
            }

            // Active LLM Model Card
            item {
                Text(
                    text = "Active LLM Model",
                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                )
                Spacer(modifier = Modifier.height(8.dp))
                Surface(
                    shape = RoundedCornerShape(16.dp),
                    color = MaterialTheme.colorScheme.surfaceVariant,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    text = activeModel?.displayName ?: selectedLlmModel?.name ?: "Qwen 2.5 1.5B Instruct",
                                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                                )
                                val sizeMb = (activeModel?.sizeBytes ?: 1117320736L) / (1024 * 1024)
                                Text(
                                    text = "Format: GGUF Q4_K_M • Size: ${sizeMb}MB",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                            Surface(
                                shape = RoundedCornerShape(12.dp),
                                color = ChatGPTGreen.copy(alpha = 0.15f)
                            ) {
                                Text(
                                    text = "Ready",
                                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                                    color = ChatGPTGreen,
                                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
                                )
                            }
                        }
                    }
                }
            }

            // Installed Models & Import
            item {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Installed GGUF Models",
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                    )
                    TextButton(onClick = { importModelLauncher.launch(arrayOf("*/*")) }) {
                        Icon(Icons.Default.Add, contentDescription = null, modifier = Modifier.size(16.dp))
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Import GGUF")
                    }
                }

                if (installedModels.isEmpty()) {
                    Text(
                        text = "Using bundled Qwen 2.5 1.5B GGUF model.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                } else {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        installedModels.forEach { model ->
                            Surface(
                                shape = RoundedCornerShape(12.dp),
                                color = MaterialTheme.colorScheme.surfaceVariant,
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Row(
                                    modifier = Modifier.padding(12.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Column(modifier = Modifier.weight(1f)) {
                                        Text(
                                            text = model.filename,
                                            style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold)
                                        )
                                        Text(
                                            text = "${model.sizeBytes / (1024 * 1024)} MB",
                                            style = MaterialTheme.typography.labelSmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                    }
                                    IconButton(onClick = { modelToDelete = model }, modifier = Modifier.size(28.dp)) {
                                        Icon(
                                            imageVector = Icons.Default.Delete,
                                            contentDescription = "Delete",
                                            tint = MaterialTheme.colorScheme.error,
                                            modifier = Modifier.size(18.dp)
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // SLM / LLM Models Catalog
            item {
                Text(
                    text = "SLM Model Catalog (One-Click Downloads)",
                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                )
                Spacer(modifier = Modifier.height(8.dp))
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    AvailableLlmModels.list.forEach { model ->
                        val isDownloaded = downloadedLlmModels.any { it.name == model.fileName }
                        val isSelected = selectedLlmModel?.name == model.fileName
                        val isDownloading = currentlyDownloadingLlm == model.fileName && llmDownloadState is DownloadState.Progress
                        val progress = if (isDownloading) llmDownloadState as DownloadState.Progress else null

                        ModelOptionCard(
                            model = model,
                            isDownloaded = isDownloaded,
                            isSelected = isSelected,
                            isDownloading = isDownloading,
                            downloadProgress = progress,
                            onDownload = {
                                viewModel.downloadLlmModel(model.downloadUrl, model.fileName)
                            },
                            onSelect = {
                                val file = downloadedLlmModels.firstOrNull { it.name == model.fileName }
                                if (file != null) viewModel.selectLlmModel(file)
                            }
                        )
                    }
                }
            }

            // Embedding Models Catalog
            item {
                Text(
                    text = "Embedding Model Catalog",
                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                )
                Spacer(modifier = Modifier.height(8.dp))
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    AvailableEmbeddingModels.list.forEach { model ->
                        val isDownloaded = downloadedEmbeddingModels.any { it.name == model.fileName }
                        val isSelected = selectedEmbeddingModel?.name == model.fileName
                        val isDownloading = currentlyDownloadingEmbed == model.fileName && embeddingDownloadState is DownloadState.Progress
                        val progress = if (isDownloading) embeddingDownloadState as DownloadState.Progress else null

                        EmbeddingModelOptionCard(
                            model = model,
                            isDownloaded = isDownloaded,
                            isSelected = isSelected,
                            isDownloading = isDownloading,
                            downloadProgress = progress,
                            onDownload = {
                                viewModel.downloadEmbeddingModel(model.downloadUrl, model.fileName)
                            },
                            onSelect = {
                                val file = downloadedEmbeddingModels.firstOrNull { it.name == model.fileName }
                                if (file != null) viewModel.selectEmbeddingModel(file)
                            }
                        )
                    }
                }
            }

            // 100% On-Device Privacy Card
            item {
                Surface(
                    shape = RoundedCornerShape(16.dp),
                    color = ChatGPTGreen.copy(alpha = 0.1f),
                    border = androidx.compose.foundation.BorderStroke(1.dp, ChatGPTGreen.copy(alpha = 0.3f)),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Row(
                        modifier = Modifier.padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            imageVector = Icons.Default.Check,
                            contentDescription = null,
                            tint = ChatGPTGreen,
                            modifier = Modifier.size(24.dp)
                        )
                        Spacer(modifier = Modifier.width(12.dp))
                        Column {
                            Text(
                                text = "100% Local & Private",
                                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold),
                                color = ChatGPTGreen
                            )
                            Text(
                                text = "All text chunking, embeddings, vector retrieval, and neural inference execute entirely on-device without internet access.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurface
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun ModelOptionCard(
    model: AvailableLlmModel,
    isDownloaded: Boolean,
    isSelected: Boolean,
    isDownloading: Boolean,
    downloadProgress: DownloadState.Progress?,
    onDownload: () -> Unit,
    onSelect: () -> Unit,
    modifier: Modifier = Modifier
) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = when {
                isSelected -> MaterialTheme.colorScheme.primaryContainer
                isDownloaded -> MaterialTheme.colorScheme.surfaceVariant
                else -> MaterialTheme.colorScheme.surface
            }
        ),
        shape = RoundedCornerShape(12.dp),
        border = if (isSelected) androidx.compose.foundation.BorderStroke(2.dp, ChatGPTGreen) else null,
        modifier = modifier
            .fillMaxWidth()
            .then(if (isDownloaded) Modifier.clickable { onSelect() } else Modifier)
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = model.name,
                        fontWeight = FontWeight.Bold,
                        style = MaterialTheme.typography.titleSmall
                    )
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = "${model.parameterSize} params • Size: ${model.downloadSize}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                if (isDownloading && downloadProgress != null) {
                    Text(
                        "${downloadProgress.percent}%",
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Bold,
                        color = ChatGPTGreen
                    )
                } else if (isDownloaded) {
                    Surface(
                        shape = RoundedCornerShape(8.dp),
                        color = if (isSelected) ChatGPTGreen else MaterialTheme.colorScheme.secondaryContainer
                    ) {
                        Text(
                            text = if (isSelected) "Active" else "Select",
                            style = MaterialTheme.typography.labelSmall,
                            color = if (isSelected) Color.White else MaterialTheme.colorScheme.onSecondaryContainer,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                        )
                    }
                } else {
                    Button(
                        onClick = onDownload,
                        shape = RoundedCornerShape(8.dp),
                        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp)
                    ) {
                        Icon(Icons.Default.Download, contentDescription = null, modifier = Modifier.size(14.dp))
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Download", style = MaterialTheme.typography.labelSmall)
                    }
                }
            }

            if (isDownloading && downloadProgress != null) {
                Spacer(modifier = Modifier.height(8.dp))
                LinearProgressIndicator(
                    progress = { downloadProgress.percent / 100f },
                    modifier = Modifier.fillMaxWidth(),
                    color = ChatGPTGreen
                )
            }
        }
    }
}

@Composable
fun EmbeddingModelOptionCard(
    model: AvailableEmbeddingModel,
    isDownloaded: Boolean,
    isSelected: Boolean,
    isDownloading: Boolean,
    downloadProgress: DownloadState.Progress?,
    onDownload: () -> Unit,
    onSelect: () -> Unit,
    modifier: Modifier = Modifier
) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = when {
                isSelected -> MaterialTheme.colorScheme.primaryContainer
                isDownloaded -> MaterialTheme.colorScheme.surfaceVariant
                else -> MaterialTheme.colorScheme.surface
            }
        ),
        shape = RoundedCornerShape(12.dp),
        border = if (isSelected) androidx.compose.foundation.BorderStroke(2.dp, ChatGPTGreen) else null,
        modifier = modifier
            .fillMaxWidth()
            .then(if (isDownloaded) Modifier.clickable { onSelect() } else Modifier)
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = model.name,
                        fontWeight = FontWeight.Bold,
                        style = MaterialTheme.typography.titleSmall
                    )
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = "${model.format} • ${model.dimension}-dim • Size: ${model.downloadSize}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                if (isDownloading && downloadProgress != null) {
                    Text(
                        "${downloadProgress.percent}%",
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Bold,
                        color = ChatGPTGreen
                    )
                } else if (isDownloaded) {
                    Surface(
                        shape = RoundedCornerShape(8.dp),
                        color = if (isSelected) ChatGPTGreen else MaterialTheme.colorScheme.secondaryContainer
                    ) {
                        Text(
                            text = if (isSelected) "Active" else "Select",
                            style = MaterialTheme.typography.labelSmall,
                            color = if (isSelected) Color.White else MaterialTheme.colorScheme.onSecondaryContainer,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                        )
                    }
                } else {
                    Button(
                        onClick = onDownload,
                        shape = RoundedCornerShape(8.dp),
                        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp)
                    ) {
                        Icon(Icons.Default.Download, contentDescription = null, modifier = Modifier.size(14.dp))
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Download", style = MaterialTheme.typography.labelSmall)
                    }
                }
            }

            if (isDownloading && downloadProgress != null) {
                Spacer(modifier = Modifier.height(8.dp))
                LinearProgressIndicator(
                    progress = { downloadProgress.percent / 100f },
                    modifier = Modifier.fillMaxWidth(),
                    color = ChatGPTGreen
                )
            }
        }
    }
}
