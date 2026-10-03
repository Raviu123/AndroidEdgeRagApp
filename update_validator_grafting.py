import re

def update_file():
    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/validation/CitationValidator.kt', 'r', encoding='utf-8') as f:
        content = f.read()

    # We want to change the WhatIsGrafting check to a general Grafting check that includes "explain whip and tongue grafting"
    old_grafting = '''        // 6. What is grafting?
        val isWhatIsGrafting = qLower.contains("what is grafting") || qLower.contains("define grafting") || qLower.contains("definition of grafting")
        if (isWhatIsGrafting) {
            val graftingCoreTerms = listOf("technique", "horticultural", "tissue", "tissues", "plant", "plants", "join", "joined", "growth", "scion", "rootstock", "cambium", "alive", "heal", "taken")
            val irrelevantTerms = listOf("knife", "knives", "shear", "wax", "humidity", "85%", "pathogen", "weekly")
            val hasCore = graftingCoreTerms.any { cLower.contains(it) }
            val hasIrrelevant = irrelevantTerms.any { cLower.contains(it) }
            if (hasIrrelevant && !hasCore) return false
            return hasCore
        }'''

    new_grafting = '''        // 6. What is grafting? / Explain grafting
        val isWhatIsGrafting = qLower.contains("what is grafting") || qLower.contains("define grafting") || qLower.contains("definition of grafting")
        val isExplainGrafting = qLower.contains("explain") && qLower.contains("grafting")
        if (isWhatIsGrafting || isExplainGrafting) {
            val graftingCoreTerms = listOf("technique", "horticultural", "tissue", "tissues", "plant", "plants", "join", "joined", "growth", "scion", "rootstock", "cambium", "alive", "heal", "taken", "diameter", "interlocking", "cut", "mechanical", "strength", "wax", "seal", "bridge", "callus", "week", "atmospheric", "drying")
            val irrelevantTerms = listOf("knife", "knives", "shear", "humidity", "85%", "pathogen", "weekly")
            val hasCore = graftingCoreTerms.any { cLower.contains(it) }
            val hasIrrelevant = irrelevantTerms.any { cLower.contains(it) }
            if (hasIrrelevant && !hasCore) return false
            return hasCore
        }'''

    content = content.replace(old_grafting, new_grafting)

    with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/validation/CitationValidator.kt', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    update_file()
