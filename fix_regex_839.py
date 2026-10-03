import re

with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/validation/CitationValidator.kt', encoding='utf-8') as f:
    content = f.read()

# The target is lines 841-842 (0-indexed 840-841)
lines = content.split('\n')
target_line = lines[841]  # line 842 (0-indexed)
print("LINE 842 content:", repr(target_line))

# Build replacement block
replacement = (
    '        // 3. Remove inline orphan numbers followed immediately by punctuation, bracket, or end-of-string\n'
    '        // Android-compatible: avoid stacked/variable-length look-behinds.\n'
    '        run {\n'
    '            val orphanRegex = Regex("""(^|[\\\\s])(\\\\d+)[\\.:]\\\\s*(?=[,\\.;:\\[\\]]|\\\\z)""", RegexOption.MULTILINE)\n'
    '            val sb = StringBuilder()\n'
    '            var last = 0\n'
    '            for (m in orphanRegex.findAll(cleaned)) {\n'
    '                val precText = cleaned.substring(maxOf(0, m.range.first - 10), m.range.first)\n'
    '                val isAfterCitation = precText.contains(Regex("""(?i)(Page|Pages|Source)\\\\s*$"""))\n'
    '                if (!isAfterCitation) {\n'
    '                    sb.append(cleaned.substring(last, m.range.first))\n'
    '                    sb.append(m.groupValues[1])\n'
    '                    last = m.range.last + 1\n'
    '                }\n'
    '            }\n'
    '            sb.append(cleaned.substring(last))\n'
    '            cleaned = sb.toString()\n'
    '        }'
)

# Replace lines 840-842 (0-indexed) -- the comment + the replace call
new_lines = lines[:840] + replacement.split('\n') + lines[843:]
new_content = '\n'.join(new_lines)

with open('app/src/main/kotlin/com/siddu/offlinedocai/ai/validation/CitationValidator.kt', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Done. Wrote", len(new_content), "bytes")
