package com.siddu.offlinedocai

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Chat
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.Settings
import androidx.compose.ui.graphics.vector.ImageVector

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { OfflineDocAIApp() }
    }
}

@Composable
fun OfflineDocAIApp(mainViewModel: MainViewModel = viewModel()) {
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
            composable("home") { HomeScreen(mainViewModel) }
            composable("chats") { ChatsScreen(mainViewModel) }
            composable("documents") { DocumentsScreen(mainViewModel) }
            composable("settings") { SettingsScreen(mainViewModel) }
            composable("documentDetail/{docId}") { backStackEntry ->
                val docId = backStackEntry.arguments?.getString("docId")?.toLongOrNull() ?: return@composable
                DocumentDetailScreen(docId, mainViewModel)
            }
            composable("chat/{convId}") { backStackEntry ->
                val convId = backStackEntry.arguments?.getString("convId")?.toLongOrNull() ?: return@composable
                ChatScreen(convId, mainViewModel)
            }
            composable("newChat") { NewChatScreen(mainViewModel, navController) }
            composable("importProcessing") { ImportProcessingScreen(mainViewModel) }
        }
    }
}

data class NavItem(val route: String, val label: String, val icon: ImageVector)

@Composable fun HomeScreen(viewModel: MainViewModel) = PlaceholderScreen("Home")
@Composable fun ChatsScreen(viewModel: MainViewModel) = PlaceholderScreen("Chats")
@Composable fun DocumentsScreen(viewModel: MainViewModel) = PlaceholderScreen("Documents")
@Composable fun SettingsScreen(viewModel: MainViewModel) = PlaceholderScreen("Settings")
@Composable fun DocumentDetailScreen(docId: Long, viewModel: MainViewModel) = PlaceholderScreen("Document Detail")
@Composable fun ImportProcessingScreen(viewModel: MainViewModel) = PlaceholderScreen("Import Processing")
@Composable fun ChatScreen(convId: Long, viewModel: MainViewModel) = PlaceholderScreen("Chat")
@Composable fun NewChatScreen(viewModel: MainViewModel, navController: NavHostController) = PlaceholderScreen("New Chat")

@Composable
fun PlaceholderScreen(name: String) {
    Surface(modifier = Modifier.fillMaxSize()) {
        Text("$name Screen – loading…", style = MaterialTheme.typography.bodyLarge, modifier = Modifier.padding(16.dp))
    }
}
