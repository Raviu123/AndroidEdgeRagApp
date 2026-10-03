with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/validation/CitationValidator.kt', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Line 887 is index 886 (0-based): the inlineItemRegex 
# Replace lines 886-901 (the whole inline renumbering block) with an imperative equivalent
# that avoids variable-length look-behinds
old_block_start = 886  # 0-indexed
old_block_end = 902    # exclusive, 0-indexed

old_lines_preview = lines[old_block_start:old_block_end]
print("OLD BLOCK:")
for i, l in enumerate(old_lines_preview, start=old_block_start+1):
    print(f"  {i}: {repr(l)}")

new_block = [
    '        // 5. Renumber inline lists (e.g. "are: 1. Foo 2. Bar 4. Baz")',
    '        // Android-compatible: avoid variable-length look-behinds using findAll + context check.',
    '        run {',
    r'            val inlineItemRegex = Regex("""(^|[:\.\n\s])(\d+)[\.):](\s+)([A-Za-z*\u2022\-\[])""")',
    '            val inlineMatches = inlineItemRegex.findAll(result).toList()',
    '            if (inlineMatches.size >= 2) {',
    '                val sb = StringBuilder()',
    '                var lastEnd = 0',
    '                var seq = 1',
    '                for (match in inlineMatches) {',
    '                    val precText = result.substring(maxOf(0, match.range.first - 10), match.range.first)',
    '                    val isAfterCitation = precText.contains(Regex("""(?i)(Page|Pages|Source)\\s*$"""))',
    '                    if (!isAfterCitation) {',
    '                        sb.append(result.substring(lastEnd, match.range.first))',
    '                        sb.append(match.groupValues[1]) // restore delimiter',
    '                        val nextChar = match.groupValues[4]',
    '                        sb.append("$seq. $nextChar")',
    '                        lastEnd = match.range.last + 1',
    '                        seq++',
    '                    }',
    '                }',
    '                sb.append(result.substring(lastEnd))',
    '                result = sb.toString()',
    '            }',
    '        }',
]

new_lines = lines[:old_block_start] + new_block + lines[old_block_end:]
new_content = '\n'.join(new_lines)

with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/validation/CitationValidator.kt', 'w', encoding='utf-8') as f:
    f.write(new_content)

print(f"\nDone. File now has {len(new_content)} bytes, {len(new_lines)} lines.")
