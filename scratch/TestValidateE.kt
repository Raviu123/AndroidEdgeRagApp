import com.siddu.offlinedocai.ai.validation.CitationValidator
import com.siddu.offlinedocai.domain.model.ParsedSourceBlock

fun main() {
    val answer = """### Page 1:
Initialization Step: Power on the device and verify status LED is green.
(Page 1)

### Page 2:
Configuration Step: Connect device to local network via USB ethernet interface.
(Page 2)

### Page 3:
Verification Step: Run automated diagnostic script and confirm zero packet drop.
(Page 3)"""
    val claims = CitationValidator.extractClaims(answer)
    println("Claims size: ${claims.size}")
    claims.forEachIndexed { i, c -> println("Claim ${i+1}: '$c'") }

    val block1 = ParsedSourceBlock(null, 1, "1", 1, "Initialization Step: Power on the device and verify status LED is green.")
    val block2 = ParsedSourceBlock(null, 2, "2", 2, "Configuration Step: Connect device to local network via USB ethernet interface.")
    val block3 = ParsedSourceBlock(null, 3, "3", 3, "Verification Step: Run automated diagnostic script and confirm zero packet drop.")
    val blocks = listOf(block1, block2, block3)

    val res = CitationValidator.validate(answer, blocks, "Summarize pages 1 to 3.", false, com.siddu.offlinedocai.ai.llm.AnswerLength.SHORT)
    println("Validated answer: ${res.validatedAnswer}")
    println("Fallback level: ${res.fallbackLevel}")
}
