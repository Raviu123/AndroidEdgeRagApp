package com.siddu.offlinedocai.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Check
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

@Composable
fun MarkdownMessage(
    text: String,
    modifier: Modifier = Modifier,
    isAssistant: Boolean = true
) {
    val clipboardManager = LocalClipboardManager.current
    val coroutineScope = rememberCoroutineScope()
    var copied by remember { mutableStateOf(false) }

    Column(modifier = modifier.fillMaxWidth()) {
        val blocks = remember(text) { parseMarkdownBlocks(text) }

        for (block in blocks) {
            when (block) {
                is MarkdownBlock.CodeBlock -> {
                    CodeBlockView(language = block.language, code = block.code)
                    Spacer(modifier = Modifier.height(8.dp))
                }
                is MarkdownBlock.ListItem -> {
                    ListItemView(
                        number = block.prefix,
                        content = block.content,
                        citationColor = MaterialTheme.colorScheme.primary
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                }
                is MarkdownBlock.Paragraph -> {
                    ParagraphView(content = block.content, citationColor = MaterialTheme.colorScheme.primary)
                    Spacer(modifier = Modifier.height(8.dp))
                }
            }
        }

        if (isAssistant && text.isNotBlank()) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 4.dp),
                horizontalArrangement = Arrangement.Start,
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(
                    onClick = {
                        clipboardManager.setText(AnnotatedString(text))
                        copied = true
                        coroutineScope.launch {
                            delay(2000)
                            copied = false
                        }
                    },
                    modifier = Modifier.size(28.dp)
                ) {
                    Icon(
                        imageVector = if (copied) Icons.Default.Check else Icons.Default.ContentCopy,
                        contentDescription = "Copy message",
                        tint = if (copied) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
                        modifier = Modifier.size(16.dp)
                    )
                }
                if (copied) {
                    Text(
                        "Copied",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.primary
                    )
                }
            }
        }
    }
}

@Composable
private fun ListItemView(number: String, content: String, citationColor: Color) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp),
        verticalAlignment = Alignment.Top
    ) {
        val isBullet = number == "•" || number == "-" || number == "*"
        if (isBullet) {
            Box(
                modifier = Modifier
                    .padding(top = 6.dp, end = 10.dp)
                    .size(6.dp)
                    .clip(RoundedCornerShape(3.dp))
                    .background(MaterialTheme.colorScheme.primary)
            )
        } else {
            Text(
                text = number,
                style = MaterialTheme.typography.bodyMedium.copy(
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary
                ),
                modifier = Modifier.width(28.dp)
            )
        }
        Text(
            text = formatInlineMarkdown(content, citationColor),
            style = MaterialTheme.typography.bodyMedium.copy(
                lineHeight = 22.sp,
                color = MaterialTheme.colorScheme.onSurface
            ),
            modifier = Modifier.weight(1f)
        )
    }
}

@Composable
private fun ParagraphView(content: String, citationColor: Color) {
    Text(
        text = formatInlineMarkdown(content, citationColor),
        style = MaterialTheme.typography.bodyMedium.copy(
            lineHeight = 22.sp,
            color = MaterialTheme.colorScheme.onSurface
        ),
        modifier = Modifier.fillMaxWidth()
    )
}

@Composable
private fun CodeBlockView(language: String, code: String) {
    val clipboardManager = LocalClipboardManager.current
    var copiedCode by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(Color(0xFF1E1E1E))
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color(0xFF2D2D2D))
                .padding(horizontal = 12.dp, vertical = 6.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = language.ifBlank { "code" },
                color = Color(0xFFCCCCCC),
                fontSize = 12.sp,
                fontFamily = FontFamily.Monospace
            )
            IconButton(
                onClick = {
                    clipboardManager.setText(AnnotatedString(code))
                    copiedCode = true
                    scope.launch {
                        delay(2000)
                        copiedCode = false
                    }
                },
                modifier = Modifier.size(24.dp)
            ) {
                Icon(
                    imageVector = if (copiedCode) Icons.Default.Check else Icons.Default.ContentCopy,
                    contentDescription = "Copy code",
                    tint = if (copiedCode) Color(0xFF4CAF50) else Color(0xFFAAAAAA),
                    modifier = Modifier.size(14.dp)
                )
            }
        }
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState())
                .padding(12.dp)
        ) {
            Text(
                text = code,
                color = Color(0xFFE6E6E6),
                fontSize = 13.sp,
                fontFamily = FontFamily.Monospace,
                lineHeight = 18.sp
            )
        }
    }
}

private sealed class MarkdownBlock {
    data class Paragraph(val content: String) : MarkdownBlock()
    data class ListItem(val prefix: String, val content: String) : MarkdownBlock()
    data class CodeBlock(val language: String, val code: String) : MarkdownBlock()
}

private fun parseMarkdownBlocks(text: String): List<MarkdownBlock> {
    val blocks = mutableListOf<MarkdownBlock>()
    val lines = text.lines()
    var i = 0

    while (i < lines.size) {
        val line = lines[i]

        // Check for fenced code block ```
        if (line.trim().startsWith("```")) {
            val lang = line.trim().removePrefix("```").trim()
            val codeLines = mutableListOf<String>()
            i++
            while (i < lines.size && !lines[i].trim().startsWith("```")) {
                codeLines.add(lines[i])
                i++
            }
            if (i < lines.size) i++ // skip closing ```
            blocks.add(MarkdownBlock.CodeBlock(lang, codeLines.joinToString("\n")))
            continue
        }

        // Check for numbered list: e.g. "1. Item" or "1) Item"
        val listMatch = Regex("""^[ \t]*(\d+[\.\)])\s+(.*)$""").matchEntire(line)
        if (listMatch != null) {
            blocks.add(MarkdownBlock.ListItem(listMatch.groupValues[1], listMatch.groupValues[2]))
            i++
            continue
        }

        // Check for bullet list: e.g. "• Item", "- Item", "* Item"
        val bulletMatch = Regex("""^[ \t]*([•\-\*])\s+(.*)$""").matchEntire(line)
        if (bulletMatch != null) {
            blocks.add(MarkdownBlock.ListItem("•", bulletMatch.groupValues[2]))
            i++
            continue
        }

        // Normal text line
        if (line.isNotBlank()) {
            blocks.add(MarkdownBlock.Paragraph(line.trim()))
        }
        i++
    }
    return blocks
}

private fun formatInlineMarkdown(text: String, citationColor: Color): AnnotatedString {
    return buildAnnotatedString {
        var cursor = 0

        // Regex for bold **text**, inline `code`, and citations [Source X, Page Y] or [Page Y]
        val inlineRegex = Regex("""(\*\*(.+?)\*\*|`([^`]+)`|\[(?:Source\s*\d+,\s*)?Page\s*\d+(?:-\d+)?\])""")
        val matches = inlineRegex.findAll(text)

        for (match in matches) {
            if (match.range.first > cursor) {
                append(text.substring(cursor, match.range.first))
            }

            val value = match.value
            when {
                // Bold **text**
                value.startsWith("**") && value.endsWith("**") && value.length > 4 -> {
                    val content = value.substring(2, value.length - 2)
                    pushStyle(SpanStyle(fontWeight = FontWeight.Bold))
                    append(content)
                    pop()
                }
                // Inline `code`
                value.startsWith("`") && value.endsWith("`") && value.length > 2 -> {
                    val code = value.substring(1, value.length - 1)
                    pushStyle(
                        SpanStyle(
                            fontFamily = FontFamily.Monospace,
                            background = Color(0x22808080),
                            fontSize = 13.sp
                        )
                    )
                    append(" $code ")
                    pop()
                }
                // Citations [Source X, Page Y]
                value.startsWith("[") && value.endsWith("]") -> {
                    pushStyle(
                        SpanStyle(
                            color = citationColor,
                            fontWeight = FontWeight.SemiBold,
                            fontSize = 12.sp,
                            background = citationColor.copy(alpha = 0.12f)
                        )
                    )
                    append(" $value ")
                    pop()
                }
                else -> {
                    append(value)
                }
            }
            cursor = match.range.last + 1
        }

        if (cursor < text.length) {
            append(text.substring(cursor))
        }
    }
}
