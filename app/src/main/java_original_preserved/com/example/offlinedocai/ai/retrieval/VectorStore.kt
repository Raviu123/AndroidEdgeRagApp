package com.example.offlinedocai.ai.retrieval

import com.example.offlinedocai.data.database.entities.DocumentChunkEntity
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

class VectorStore {

    /**
     * Finds the top K most similar document chunks
     * using cosine similarity.
     */
    fun findTopK(
        queryVector: FloatArray,
        chunks: List<DocumentChunkEntity>,
        topK: Int = 3
    ): List<SearchResult> {

        if (queryVector.isEmpty() || chunks.isEmpty()) {
            return emptyList()
        }

        val results = chunks.mapNotNull { chunk ->

            try {
                val chunkVector =
                    byteArrayToFloatArray(chunk.embedding)

                if (chunkVector.isEmpty()) {
                    null
                } else {
                    val score =
                        cosineSimilarity(
                            queryVector,
                            chunkVector
                        )

                    SearchResult(
                        chunk = chunk,
                        similarity = score
                    )
                }

            } catch (e: Exception) {
                null
            }
        }

        return results
            .sortedByDescending { it.similarity }
            .take(topK.coerceAtLeast(1))
    }

    /**
     * Calculates cosine similarity between two vectors.
     *
     * Result:
     *  1.0  = identical direction
     *  0.0  = unrelated/orthogonal
     * -1.0  = opposite direction
     */
    private fun cosineSimilarity(
        vectorA: FloatArray,
        vectorB: FloatArray
    ): Float {

        if (vectorA.isEmpty() || vectorB.isEmpty()) {
            return 0f
        }

        if (vectorA.size != vectorB.size) {
            return 0f
        }

        var dotProduct = 0.0
        var magnitudeA = 0.0
        var magnitudeB = 0.0

        for (i in vectorA.indices) {

            val a = vectorA[i].toDouble()
            val b = vectorB[i].toDouble()

            dotProduct += a * b
            magnitudeA += a * a
            magnitudeB += b * b
        }

        if (magnitudeA == 0.0 || magnitudeB == 0.0) {
            return 0f
        }

        val denominator =
            sqrt(magnitudeA) * sqrt(magnitudeB)

        return (dotProduct / denominator).toFloat()
    }

    /**
     * Converts the ByteArray stored in Room back into
     * the original FloatArray embedding.
     *
     * The embedding is stored using little-endian
     * 32-bit floating point values.
     */
    private fun byteArrayToFloatArray(
        bytes: ByteArray
    ): FloatArray {

        if (bytes.isEmpty()) {
            return FloatArray(0)
        }

        if (bytes.size % Float.SIZE_BYTES != 0) {
            throw IllegalArgumentException(
                "Invalid embedding byte array size: ${bytes.size}"
            )
        }

        val floatCount =
            bytes.size / Float.SIZE_BYTES

        val floatArray =
            FloatArray(floatCount)

        val byteBuffer =
            ByteBuffer
                .wrap(bytes)
                .order(ByteOrder.LITTLE_ENDIAN)

        for (i in 0 until floatCount) {
            floatArray[i] =
                byteBuffer.getFloat()
        }

        return floatArray
    }

    companion object {

        /**
         * Converts an embedding FloatArray into a ByteArray
         * suitable for storing in Room/SQLite.
         *
         * Uses little-endian 32-bit floating point values.
         */
        fun floatArrayToByteArray(
            floats: FloatArray
        ): ByteArray {

            if (floats.isEmpty()) {
                return ByteArray(0)
            }

            val bytes =
                ByteArray(
                    floats.size * Float.SIZE_BYTES
                )

            val byteBuffer =
                ByteBuffer
                    .wrap(bytes)
                    .order(ByteOrder.LITTLE_ENDIAN)

            for (value in floats) {
                byteBuffer.putFloat(value)
            }

            return bytes
        }
    }
}

/**
 * Represents a retrieved document chunk and
 * its cosine-similarity score.
 */
data class SearchResult(
    val chunk: DocumentChunkEntity,
    val similarity: Float
)