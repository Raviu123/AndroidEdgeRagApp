package com.example.offlinedocai.domain.rag

import android.util.Xml
import org.xmlpull.v1.XmlPullParser
import java.io.InputStream
import java.util.zip.ZipInputStream

object DocxExtractor {
    /**
     * Extracts text content from a DOCX file using a lightweight XML pull parser
     * on the internal zip file (word/document.xml). Requires zero extra dependencies.
     */
    fun extractText(inputStream: InputStream): String {
        val zipInputStream = ZipInputStream(inputStream)
        var entry = zipInputStream.nextEntry
        var documentXmlStream: InputStream? = null

        while (entry != null) {
            if (entry.name == "word/document.xml") {
                documentXmlStream = zipInputStream
                break
            }
            entry = zipInputStream.nextEntry
        }

        if (documentXmlStream == null) {
            return ""
        }

        val extractedText = StringBuilder()
        try {
            val parser = Xml.newPullParser()
            parser.setFeature(XmlPullParser.FEATURE_PROCESS_NAMESPACES, true)
            parser.setInput(documentXmlStream, "UTF-8")

            var eventType = parser.eventType
            while (eventType != XmlPullParser.END_DOCUMENT) {
                if (eventType == XmlPullParser.START_TAG) {
                    when (parser.name) {
                        "p" -> {
                            // Paragraph tag
                            if (extractedText.isNotEmpty() && !extractedText.endsWith("\n")) {
                                extractedText.append("\n")
                            }
                        }
                        "t" -> {
                            // Text tag
                            extractedText.append(parser.nextText())
                        }
                    }
                }
                eventType = parser.next()
            }
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            try {
                zipInputStream.close()
            } catch (e: Exception) {}
        }

        return extractedText.toString()
    }
}
