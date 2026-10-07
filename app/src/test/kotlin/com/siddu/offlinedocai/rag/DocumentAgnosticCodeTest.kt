package com.siddu.offlinedocai.rag

import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Guards against document-specific tuning creeping back into production code. The app must work on
 * any document, so production sources may not contain vocabulary from the documents it was tested on.
 */
class DocumentAgnosticCodeTest {

    private val banned = listOf(
        // Horticulture test guide
        "graft", "scion", "rootstock", "cambium", "callus", "aftercare", "sucker",
        // NASA technical-report boilerplate
        "sti program", "standard form 298", "aeronautics",
        // Résumé-specific section handling
        "certifications", "\"skills\"", "\"projects\"",
        // Device-spec heuristics
        "mah", "wattage"
    )

    @Test
    fun productionSourcesContainNoDocumentSpecificVocabulary() {
        val root = listOf(File("src/main/kotlin"), File("app/src/main/kotlin")).first { it.exists() }
        val offenders = root.walkTopDown()
            .filter { it.isFile && it.extension == "kt" }
            .flatMap { file ->
                file.readLines().withIndex()
                    .filter { (_, line) -> banned.any { word -> line.lowercase().contains(word) } }
                    .map { (i, line) -> "${file.name}:${i + 1}: ${line.trim()}" }
            }
            .toList()
        assertTrue("Document-specific vocabulary found:\n" + offenders.joinToString("\n"), offenders.isEmpty())
    }
}
