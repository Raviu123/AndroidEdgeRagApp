package com.example.offlinedocai.ui.settings

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.ModelTraining
import androidx.compose.material.icons.filled.SettingsInputComponent
import androidx.compose.material.icons.filled.Storage
import androidx.compose.material.icons.filled.UploadFile
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import java.io.File

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    viewModel: SettingsViewModel = viewModel(),
    onBack: (() -> Unit)? = null
) {
    val scrollState = rememberScrollState()
    val settings = viewModel.settingsManager

    var themeMode by remember { mutableStateOf(settings.themeMode) }
    var ggufPath by remember { mutableStateOf(settings.ggufModelPath) }
    var onnxPath by remember { mutableStateOf(settings.onnxModelPath) }

    var chunkSize by remember { mutableStateOf(settings.chunkSize.toFloat()) }
    var chunkOverlap by remember { mutableStateOf(settings.chunkOverlap.toFloat()) }
    var topK by remember { mutableStateOf(settings.topK.toFloat()) }
    var temperature by remember { mutableStateOf(settings.temperature) }

    val importState by viewModel.importState.collectAsState()

    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let {
            viewModel.importModel(it, isGguf = true)
        }
    }

    val onnxPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let {
            viewModel.importModel(uri, isGguf = false)
        }
    }

    LaunchedEffect(importState) {
        if (importState is SettingsViewModel.ImportStatus.Success) {
            ggufPath = settings.ggufModelPath
            onnxPath = settings.onnxModelPath
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Settings") },
                navigationIcon = {
                    if (onBack != null) {
                        IconButton(onClick = onBack) {
                            Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                        }
                    }
                }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .padding(padding)
                .fillMaxSize()
                .verticalScroll(scrollState)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Import Status Banner
            when (val state = importState) {
                is SettingsViewModel.ImportStatus.Processing -> {
                    Card(
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Row(
                            modifier = Modifier.padding(16.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            CircularProgressIndicator(modifier = Modifier.size(24.dp))
                            Spacer(modifier = Modifier.width(16.dp))
                            Text("Copying model file to internal storage...")
                        }
                    }
                }
                is SettingsViewModel.ImportStatus.Success -> {
                    Card(
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.tertiaryContainer),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Row(
                            modifier = Modifier.padding(16.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(Icons.Default.Info, contentDescription = "Success")
                            Spacer(modifier = Modifier.width(16.dp))
                            Column {
                                Text("Success", style = MaterialTheme.typography.titleMedium)
                                Text("${state.fileName} imported successfully.")
                            }
                            Spacer(modifier = Modifier.weight(1f))
                            TextButton(onClick = { viewModel.resetImportState() }) {
                                Text("Dismiss")
                            }
                        }
                    }
                }
                is SettingsViewModel.ImportStatus.Error -> {
                    Card(
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Row(
                            modifier = Modifier.padding(16.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(Icons.Default.Info, contentDescription = "Error")
                            Spacer(modifier = Modifier.width(16.dp))
                            Text(state.message, modifier = Modifier.weight(1f))
                            TextButton(onClick = { viewModel.resetImportState() }) {
                                Text("Dismiss")
                            }
                        }
                    }
                }
                else -> {}
            }

            // Section 1: AI Models
            Text("AI Models (GGUF & ONNX)", style = MaterialTheme.typography.titleMedium, color = MaterialTheme.colorScheme.primary)
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    // LLM Model GGUF
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text("Local LLM (GGUF)", style = MaterialTheme.typography.bodyLarge)
                            Text(
                                text = if (ggufPath.isNotEmpty()) File(ggufPath).name else "No model imported",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        Button(onClick = { filePickerLauncher.launch("*/*") }) {
                            Icon(Icons.Default.UploadFile, contentDescription = "Import GGUF")
                            Spacer(modifier = Modifier.width(4.dp))
                            Text("Import")
                        }
                    }

                    HorizontalDivider()

                    // Embedding Model ONNX
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text("Embedding Model (ONNX)", style = MaterialTheme.typography.bodyLarge)
                            Text(
                                text = if (onnxPath.isNotEmpty()) File(onnxPath).name else "Using internal word-hashing fallback",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        Button(onClick = { onnxPickerLauncher.launch("*/*") }) {
                            Icon(Icons.Default.UploadFile, contentDescription = "Import ONNX")
                            Spacer(modifier = Modifier.width(4.dp))
                            Text("Import")
                        }
                    }
                }
            }

            // Section 2: Retrieval-Augmented Generation (RAG)
            Text("RAG Configuration", style = MaterialTheme.typography.titleMedium, color = MaterialTheme.colorScheme.primary)
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    // Chunk Size
                    Column {
                        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text("Chunk Size (Words)", style = MaterialTheme.typography.bodyMedium)
                            Text("${chunkSize.toInt()}", style = MaterialTheme.typography.bodyMedium)
                        }
                        Slider(
                            value = chunkSize,
                            onValueChange = {
                                chunkSize = it
                                settings.chunkSize = it.toInt()
                            },
                            valueRange = 100f..1000f,
                            steps = 9
                        )
                    }

                    // Chunk Overlap
                    Column {
                        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text("Chunk Overlap (Words)", style = MaterialTheme.typography.bodyMedium)
                            Text("${chunkOverlap.toInt()}", style = MaterialTheme.typography.bodyMedium)
                        }
                        Slider(
                            value = chunkOverlap,
                            onValueChange = {
                                chunkOverlap = it
                                settings.chunkOverlap = it.toInt()
                            },
                            valueRange = 10f..200f,
                            steps = 19
                        )
                    }

                    // Retrieved Chunks (Top K)
                    Column {
                        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text("Retrieved Context Chunks (Top-K)", style = MaterialTheme.typography.bodyMedium)
                            Text("${topK.toInt()}", style = MaterialTheme.typography.bodyMedium)
                        }
                        Slider(
                            value = topK,
                            onValueChange = {
                                topK = it
                                settings.topK = it.toInt()
                            },
                            valueRange = 1f..10f,
                            steps = 9
                        )
                    }
                }
            }

            // Section 3: LLM Parameters
            Text("Generation parameters", style = MaterialTheme.typography.titleMedium, color = MaterialTheme.colorScheme.primary)
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    Column {
                        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text("Temperature", style = MaterialTheme.typography.bodyMedium)
                            Text(String.format("%.2f", temperature), style = MaterialTheme.typography.bodyMedium)
                        }
                        Slider(
                            value = temperature,
                            onValueChange = {
                                temperature = it
                                settings.temperature = it
                            },
                            valueRange = 0.0f..1.5f,
                            steps = 15
                        )
                    }
                }
            }

            // Section 4: Offline Status & App Info
            Text("About & Offline Status", style = MaterialTheme.typography.titleMedium, color = MaterialTheme.colorScheme.primary)
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.ModelTraining, contentDescription = "Offline Mode")
                        Spacer(modifier = Modifier.width(16.dp))
                        Column {
                            Text("Inference Status", style = MaterialTheme.typography.bodyLarge)
                            Text(
                                text = if (ggufPath.isNotEmpty()) "● Offline AI Ready" else "Model missing – Import GGUF Model",
                                style = MaterialTheme.typography.bodyMedium,
                                color = if (ggufPath.isNotEmpty()) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error
                            )
                        }
                    }

                    HorizontalDivider()

                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.Storage, contentDescription = "Privacy")
                        Spacer(modifier = Modifier.width(16.dp))
                        Column {
                            Text("Data Security", style = MaterialTheme.typography.bodyLarge)
                            Text("All document processing, embedding, database storage, and text generation are executed strictly offline on-device.", style = MaterialTheme.typography.bodyMedium)
                        }
                    }
                }
            }
        }
    }
}
