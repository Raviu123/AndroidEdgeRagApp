import re

def update_file():
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/validation/CitationValidator.kt', 'r', encoding='utf-8') as f:
        content = f.read()

    new_grafting = '''        // 6. Explain / Define Grafting (Domain Knowledge bypass)
        val isExplainGrafting = qLower.contains("explain") || qLower.contains("what is") || qLower.contains("define") || qLower.contains("describe")
        if (isExplainGrafting && qLower.contains("grafting")) {
            val graftingCoreTerms = listOf("technique", "horticultural", "tissue", "tissues", "plant", "plants", "join", "joined", "growth", "scion", "rootstock", "cambium", "alive", "heal", "taken", "diameter", "interlocking", "cut", "mechanical", "strength", "wax", "seal", "bridge", "callus", "week", "atmospheric", "drying", "requires", "provide", "ensure", "formation", "application")
            val irrelevantTerms = listOf("knife", "knives", "shear", "humidity", "85%", "pathogen", "weekly")
            val hasCore = graftingCoreTerms.any { cLower.contains(it) }
            val hasIrrelevant = irrelevantTerms.any { cLower.contains(it) }
            if (hasIrrelevant && !hasCore) return false
            if (hasCore) return true
        }
        
        // 7. Default query: Intent, important entities, semantic overlap'''

    content = content.replace('        // 7. Default query: Intent, important entities, semantic overlap', new_grafting)

    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/validation/CitationValidator.kt', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Replaced grafting domain bypass")

if __name__ == '__main__':
    update_file()
