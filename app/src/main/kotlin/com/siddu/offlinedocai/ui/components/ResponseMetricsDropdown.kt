package com.siddu.offlinedocai.ui.components

import android.widget.Toast
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material.icons.filled.Timer
import androidx.compose.material3.Divider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.siddu.offlinedocai.domain.model.ResponseMetrics
import java.util.Locale

@Composable
fun ResponseMetricsDropdown(
    metrics: ResponseMetrics,
    userQuestion: String? = null,
    assistantAnswer: String,
    modifier: Modifier = Modifier
) {
    var expanded by remember { mutableStateOf(false) }
    val rotationAngle by animateFloatAsState(targetValue = if (expanded) 180f else 0f, label = "arrowRotation")
    val context = LocalContext.current
    val clipboardManager = LocalClipboardManager.current

    val totalSecStr = String.format(Locale.US, "%.2fs", metrics.totalDurationMs / 1000.0)
    val tpsStr = if (metrics.tokensPerSecond > 0) String.format(Locale.US, "%.1f tok/s", metrics.tokensPerSecond) else ""
    val ttftStr = if (metrics.ttftMs > 0) "${metrics.ttftMs}ms TTFT" else ""

    val summaryPillText = listOf(totalSecStr, tpsStr, ttftStr).filter { it.isNotBlank() }.joinToString(" • ")

    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(top = 6.dp, bottom = 2.dp)
    ) {
        // Collapsible Pill Header
        Row(
            modifier = Modifier
                .clip(RoundedCornerShape(12.dp))
                .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.65f))
                .border(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f), RoundedCornerShape(12.dp))
                .clickable { expanded = !expanded }
                .padding(horizontal = 10.dp, vertical = 5.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = Icons.Default.Speed,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(13.dp)
                )
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    text = summaryPillText.ifBlank { "Performance Breakdown" },
                    fontSize = 11.5.sp,
                    fontWeight = FontWeight.Medium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            Spacer(modifier = Modifier.width(8.dp))
            Icon(
                imageVector = Icons.Default.ExpandMore,
                contentDescription = if (expanded) "Collapse" else "Expand",
                tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.75f),
                modifier = Modifier
                    .size(16.dp)
                    .rotate(rotationAngle)
            )
        }

        // Expanded Breakdown Card
        AnimatedVisibility(
            visible = expanded,
            enter = expandVertically() + fadeIn(),
            exit = shrinkVertically() + fadeOut()
        ) {
            Surface(
                modifier = Modifier
                    .padding(top = 6.dp)
                    .fillMaxWidth(),
                shape = RoundedCornerShape(10.dp),
                color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.45f),
                border = androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.4f))
            ) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                imageVector = Icons.Default.Timer,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.size(14.dp)
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = "Pipeline Latency Breakdown",
                                fontSize = 12.sp,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.onSurface
                            )
                        }

                        // Copy Q&A + Performance Breakdown button
                        Row(
                            modifier = Modifier
                                .clip(RoundedCornerShape(6.dp))
                                .clickable {
                                    val formatted = metrics.toMarkdownSummary(
                                        question = userQuestion ?: "Query",
                                        answer = assistantAnswer
                                    )
                                    clipboardManager.setText(AnnotatedString(formatted))
                                    Toast.makeText(context, "Copied Q&A with Performance Breakdown!", Toast.LENGTH_SHORT).show()
                                }
                                .padding(horizontal = 6.dp, vertical = 2.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(
                                imageVector = Icons.Default.ContentCopy,
                                contentDescription = "Copy Q&A with Breakdown",
                                tint = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.size(12.dp)
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            Text(
                                text = "Copy Q&A + Metrics",
                                fontSize = 11.sp,
                                fontWeight = FontWeight.SemiBold,
                                color = MaterialTheme.colorScheme.primary
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(8.dp))
                    Divider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.3f))
                    Spacer(modifier = Modifier.height(8.dp))

                    MetricRow("Total Turnaround", totalSecStr, isPrimary = true)
                    if (metrics.tokensPerSecond > 0) {
                        MetricRow("Decoding Speed", "${String.format(Locale.US, "%.1f", metrics.tokensPerSecond)} tok/s (${metrics.generatedTokens} tokens in ${metrics.decodeMs}ms)")
                    }
                    if (metrics.ttftMs > 0) {
                        MetricRow("Time-to-First-Token (TTFT)", "${metrics.ttftMs} ms")
                    }
                    if (metrics.intent != null) {
                        MetricRow("Intent Routing", "${metrics.intentRoutingMs} ms (${metrics.intent})")
                    }
                    if (metrics.queryEmbeddingMs > 0) {
                        MetricRow("Query Vector Embedding", "${metrics.queryEmbeddingMs} ms")
                    }
                    if (metrics.searchMs > 0) {
                        MetricRow("Hybrid Search (Dense+BM25)", "${metrics.searchMs} ms (${metrics.candidateChunksCount} chunks)")
                    }
                    if (metrics.contextAssemblyMs > 0 || metrics.assembledTokens > 0) {
                        MetricRow("Context Assembly", "${metrics.contextAssemblyMs} ms (${metrics.assembledTokens} tokens)")
                    }
                    if (!metrics.modelName.isNullOrBlank()) {
                        MetricRow("Active Model", metrics.modelName)
                    }
                }
            }
        }
    }
}

@Composable
private fun MetricRow(label: String, value: String, isPrimary: Boolean = false) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = label,
            fontSize = 11.sp,
            color = if (isPrimary) MaterialTheme.colorScheme.onSurface else MaterialTheme.colorScheme.onSurfaceVariant,
            fontWeight = if (isPrimary) FontWeight.SemiBold else FontWeight.Normal
        )
        Text(
            text = value,
            fontSize = 11.sp,
            fontFamily = FontFamily.Monospace,
            fontWeight = if (isPrimary) FontWeight.Bold else FontWeight.Medium,
            color = if (isPrimary) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface
        )
    }
}
