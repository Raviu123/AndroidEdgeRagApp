package com.siddu.offlinedocai.ai.llm

/**
 * Size class of the loaded language model. Small models are easily distracted by extra passages and
 * long instructions, so prompt shape, evidence budget and answer length are tuned per tier.
 *
 *  - [TINY]: under ~0.9B parameters (e.g. 0.5B / 0.6B).
 *  - [SMALL]: ~0.9B to ~1.3B (e.g. 1B).
 *  - [STANDARD]: everything larger, and models whose size is unknown.
 */
enum class ModelTier {
    TINY,
    SMALL,
    STANDARD;

    companion object {
        private val SIZE_IN_TEXT = Regex("""(?<![\d.])(\d+(?:\.\d+)?)\s*([bm])(?![a-z])""")

        fun fromBillions(billions: Double?): ModelTier = when {
            billions == null || billions <= 0.0 -> STANDARD
            billions < 0.9 -> TINY
            billions < 1.3 -> SMALL
            else -> STANDARD
        }

        /** "0.5B", "494M", "1.5 B" -> billions of parameters; null if no size is written in [text]. */
        fun parseBillions(text: String?): Double? {
            val match = SIZE_IN_TEXT.find(text.orEmpty().lowercase().replace('_', '-')) ?: return null
            val value = match.groupValues[1].toDoubleOrNull() ?: return null
            return if (match.groupValues[2] == "m") value / 1000.0 else value
        }

        /**
         * Tier from GGUF metadata: the exact parameter count when present, else the size label, else a
         * size written in the model or file name. Unknown sizes are treated as [STANDARD].
         */
        fun detect(parameterCount: String?, sizeLabel: String?, nameHints: String): ModelTier {
            parameterCount?.trim()?.toLongOrNull()?.takeIf { it > 0 }?.let { return fromBillions(it / 1e9) }
            return fromBillions(parseBillions(sizeLabel) ?: parseBillions(nameHints))
        }
    }
}
