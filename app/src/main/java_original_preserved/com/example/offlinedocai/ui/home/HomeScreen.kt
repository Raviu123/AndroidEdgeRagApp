package com.example.offlinedocai.ui.home

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.Divider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

// Simple data class representing a document for UI purposes

data class UiDocument(
    val id: String,
    val name: String,
    val type: String,
    val size: String
)

@Composable
fun HomeScreen(
    documents: List<UiDocument>,
    onAddDocument: () -> Unit,
    onOpenDocument: (UiDocument) -> Unit,
    onOpenChat: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text(
            text = "OfflineDocAI",
            style = MaterialTheme.typography.headlineMedium
        )
        Text(
            text = "Your documents. Your device. Your AI.",
            style = MaterialTheme.typography.bodyMedium
        )
        Button(onClick = onAddDocument) {
            Text(text = "+ Add Document")
        }
        Divider()
        if (documents.isEmpty()) {
            // Empty state composable
            Text(text = "Your documents will appear here", style = MaterialTheme.typography.bodySmall)
        } else {
            LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                items(documents) { doc ->
                    Card(onClick = { onOpenDocument(doc) }) {
                        Column(modifier = Modifier.padding(8.dp)) {
                            Text(text = doc.name, style = MaterialTheme.typography.titleMedium)
                            Text(text = "${doc.type} • ${doc.size}", style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
            }
        }
        Divider()
        Button(onClick = onOpenChat) {
            Text(text = "Open Chats")
        }
    }
}
