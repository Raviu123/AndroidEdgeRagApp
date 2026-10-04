package com.siddu.offlinedocai.ui

import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.graphics.pdf.PdfDocument
import androidx.core.content.FileProvider
import com.siddu.offlinedocai.data.entities.MessageEntity
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object PdfExporter {

    fun exportAndShareConversation(
        context: Context,
        conversationTitle: String,
        messages: List<MessageEntity>
    ) {
        val pdfDocument = PdfDocument()
        val pageWidth = 595
        val pageHeight = 842
        val margin = 40f
        val contentWidth = pageWidth - (margin * 2)

        val titlePaint = Paint().apply {
            color = Color.rgb(20, 20, 20)
            textSize = 18f
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            isAntiAlias = true
        }

        val subPaint = Paint().apply {
            color = Color.rgb(100, 100, 100)
            textSize = 10f
            typeface = Typeface.DEFAULT
            isAntiAlias = true
        }

        val rolePaint = Paint().apply {
            textSize = 11f
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            isAntiAlias = true
        }

        val textPaint = Paint().apply {
            color = Color.rgb(30, 30, 30)
            textSize = 10.5f
            typeface = Typeface.DEFAULT
            isAntiAlias = true
        }

        val linePaint = Paint().apply {
            color = Color.rgb(220, 220, 220)
            strokeWidth = 1f
        }

        val userBoxPaint = Paint().apply {
            color = Color.rgb(240, 243, 246)
            style = Paint.Style.FILL
        }

        var pageNumber = 1
        var pageInfo = PdfDocument.PageInfo.Builder(pageWidth, pageHeight, pageNumber).create()
        var page = pdfDocument.startPage(pageInfo)
        var canvas = page.canvas
        var currentY = margin + 20f

        // Draw Header
        canvas.drawText("OfflineDocAI", margin, currentY, titlePaint)
        currentY += 16f

        val safeTitle = conversationTitle.ifBlank { "Conversation" }
        canvas.drawText("Chat: $safeTitle", margin, currentY, subPaint)
        currentY += 14f

        val dateStr = SimpleDateFormat("MMM dd, yyyy HH:mm", Locale.getDefault()).format(Date())
        canvas.drawText("Exported on: $dateStr  •  100% On-Device Local Intelligence", margin, currentY, subPaint)
        currentY += 16f

        canvas.drawLine(margin, currentY, pageWidth - margin, currentY, linePaint)
        currentY += 24f

        fun startNewPage() {
            pdfDocument.finishPage(page)
            pageNumber++
            pageInfo = PdfDocument.PageInfo.Builder(pageWidth, pageHeight, pageNumber).create()
            page = pdfDocument.startPage(pageInfo)
            canvas = page.canvas
            currentY = margin + 20f
        }

        for (msg in messages) {
            val isUser = msg.role.equals("user", ignoreCase = true)
            val roleLabel = if (isUser) "You" else "OfflineDocAI Assistant"
            rolePaint.color = if (isUser) Color.rgb(16, 120, 220) else Color.rgb(16, 160, 110)

            // Break message text into wrapped lines
            val rawLines = msg.content.lines()
            val wrappedLines = mutableListOf<String>()
            for (line in rawLines) {
                if (line.isBlank()) {
                    wrappedLines.add("")
                    continue
                }
                var remaining = line
                while (remaining.isNotEmpty()) {
                    val count = textPaint.breakText(remaining, true, contentWidth - 16f, null)
                    if (count <= 0) break
                    wrappedLines.add(remaining.substring(0, count))
                    remaining = remaining.substring(count)
                }
            }

            val requiredHeight = 24f + (wrappedLines.size * 15f) + 16f
            if (currentY + requiredHeight > pageHeight - margin) {
                startNewPage()
            }

            // Draw role
            canvas.drawText(roleLabel, margin, currentY, rolePaint)
            currentY += 16f

            // Draw box for user, or line for assistant
            if (isUser) {
                val boxBottom = currentY + (wrappedLines.size * 15f) + 6f
                canvas.drawRoundRect(margin - 4f, currentY - 12f, pageWidth - margin + 4f, boxBottom, 8f, 8f, userBoxPaint)
            }

            for (wLine in wrappedLines) {
                if (currentY + 16f > pageHeight - margin) {
                    startNewPage()
                }
                canvas.drawText(wLine, margin + 4f, currentY, textPaint)
                currentY += 15f
            }

            currentY += 16f
        }

        pdfDocument.finishPage(page)

        // Write to cache file
        val cleanTitle = conversationTitle.replace(Regex("""[^a-zA-Z0-9_\-]"""), "_").take(30)
        val outFile = File(context.cacheDir, "OfflineDocAI_${cleanTitle}_${System.currentTimeMillis()}.pdf")
        FileOutputStream(outFile).use { fos ->
            pdfDocument.writeTo(fos)
        }
        pdfDocument.close()

        // Share via Android Intent
        val fileUri = FileProvider.getUriForFile(
            context,
            "${context.packageName}.fileprovider",
            outFile
        )

        val shareIntent = Intent(Intent.ACTION_SEND).apply {
            type = "application/pdf"
            putExtra(Intent.EXTRA_STREAM, fileUri)
            putExtra(Intent.EXTRA_SUBJECT, "OfflineDocAI: $conversationTitle")
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }

        val chooser = Intent.createChooser(shareIntent, "Share Conversation PDF").apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        context.startActivity(chooser)
    }
}
