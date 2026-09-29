package com.siddu.offlinedocai.data

import androidx.room.Database
import androidx.room.RoomDatabase
import androidx.room.TypeConverters
import com.siddu.offlinedocai.data.dao.DocumentDao
import com.siddu.offlinedocai.data.dao.ChunkDao
import com.siddu.offlinedocai.data.dao.ConversationDao
import com.siddu.offlinedocai.data.dao.MessageDao
import com.siddu.offlinedocai.data.entities.DocumentEntity
import com.siddu.offlinedocai.data.entities.ChunkEntity
import com.siddu.offlinedocai.data.entities.ConversationEntity
import com.siddu.offlinedocai.data.entities.MessageEntity

import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

@Database(entities = [DocumentEntity::class, ChunkEntity::class, ConversationEntity::class, MessageEntity::class], version = 2, exportSchema = false)
@TypeConverters(Converters::class, FloatArrayConverter::class)
abstract class AppDatabase : RoomDatabase() {
    abstract fun documentDao(): DocumentDao
    abstract fun chunkDao(): ChunkDao
    abstract fun conversationDao(): ConversationDao
    abstract fun messageDao(): MessageDao

    companion object {
        val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                // 1. Add new structural metadata columns
                db.execSQL("ALTER TABLE chunks ADD COLUMN chunkIndex INTEGER NOT NULL DEFAULT 0")
                db.execSQL("ALTER TABLE chunks ADD COLUMN pageSpan TEXT NOT NULL DEFAULT ''")
                db.execSQL("ALTER TABLE chunks ADD COLUMN sectionTitle TEXT DEFAULT NULL")
                db.execSQL("ALTER TABLE chunks ADD COLUMN paragraphIndex INTEGER NOT NULL DEFAULT 0")

                // 2. Safely backfill pageSpan with authoritative pageNumber for existing chunks
                db.execSQL("UPDATE chunks SET pageSpan = CAST(pageNumber AS TEXT) WHERE pageSpan = '' OR pageSpan IS NULL")

                // 3. Safely backfill sequential chunkIndex per document based on (pageNumber, id)
                db.execSQL("""
                    UPDATE chunks SET chunkIndex = (
                        SELECT COUNT(*) FROM chunks c2 
                        WHERE c2.documentId = chunks.documentId 
                          AND (c2.pageNumber < chunks.pageNumber OR (c2.pageNumber = chunks.pageNumber AND c2.id < chunks.id))
                    )
                """.trimIndent())

                // 4. Backfill paragraphIndex to match the sequential chunkIndex
                db.execSQL("UPDATE chunks SET paragraphIndex = chunkIndex")
            }
        }
    }
}
