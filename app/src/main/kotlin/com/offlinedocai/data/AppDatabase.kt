package com.offlinedocai.data

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.TypeConverters
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import com.offlinedocai.data.dao.ChunkDao
import com.offlinedocai.data.dao.ConversationDao
import com.offlinedocai.data.dao.DocumentDao
import com.offlinedocai.data.dao.MessageDao
import com.offlinedocai.data.entities.ChunkEntity
import com.offlinedocai.data.entities.ConversationEntity
import com.offlinedocai.data.entities.DocumentEntity
import com.offlinedocai.data.entities.MessageEntity

@Database(
    entities = [DocumentEntity::class, ChunkEntity::class, ConversationEntity::class, MessageEntity::class],
    version = 4,
    exportSchema = true
)
@TypeConverters(Converters::class, FloatArrayConverter::class)
abstract class AppDatabase : RoomDatabase() {
    abstract fun documentDao(): DocumentDao
    abstract fun chunkDao(): ChunkDao
    abstract fun conversationDao(): ConversationDao
    abstract fun messageDao(): MessageDao

    companion object {
        private const val DB_NAME = "offline_docai.db"

        /** Embedding model id that produced every vector stored before schema v3. */
        private const val LEGACY_EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2-v1"

        @Volatile
        private var instance: AppDatabase? = null

        fun getInstance(context: Context): AppDatabase =
            instance ?: synchronized(this) {
                instance ?: Room.databaseBuilder(context.applicationContext, AppDatabase::class.java, DB_NAME)
                    .addMigrations(MIGRATION_1_2, MIGRATION_2_3, MIGRATION_3_4)
                    .fallbackToDestructiveMigrationOnDowngrade()
                    .build()
                    .also { instance = it }
            }

        val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE chunks ADD COLUMN chunkIndex INTEGER NOT NULL DEFAULT 0")
                db.execSQL("ALTER TABLE chunks ADD COLUMN pageSpan TEXT NOT NULL DEFAULT ''")
                db.execSQL("ALTER TABLE chunks ADD COLUMN sectionTitle TEXT DEFAULT NULL")
                db.execSQL("ALTER TABLE chunks ADD COLUMN paragraphIndex INTEGER NOT NULL DEFAULT 0")
                db.execSQL("UPDATE chunks SET pageSpan = CAST(pageNumber AS TEXT) WHERE pageSpan = '' OR pageSpan IS NULL")
                db.execSQL(
                    """
                    UPDATE chunks SET chunkIndex = (
                        SELECT COUNT(*) FROM chunks c2
                        WHERE c2.documentId = chunks.documentId
                          AND (c2.pageNumber < chunks.pageNumber OR (c2.pageNumber = chunks.pageNumber AND c2.id < chunks.id))
                    )
                    """.trimIndent()
                )
                db.execSQL("UPDATE chunks SET paragraphIndex = chunkIndex")
            }
        }

        /**
         * v2 -> v3:
         *  - chunks: rebuilt with a FOREIGN KEY (ON DELETE CASCADE) to documents, an index on documentId,
         *    and embeddingModelId / embeddingDim columns. Orphaned chunks (document already deleted) and
         *    duplicate chunks (created by the old no-op delete during re-processing) are dropped.
         *  - documents: localPath / indexSignature / errorMessage. indexSignature starts NULL so every
         *    document is re-chunked and re-embedded with the new pipeline on next launch.
         *  - messages: index on conversationId.
         */
        val MIGRATION_2_3 = object : Migration(2, 3) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE documents ADD COLUMN localPath TEXT")
                db.execSQL("ALTER TABLE documents ADD COLUMN indexSignature TEXT")
                db.execSQL("ALTER TABLE documents ADD COLUMN errorMessage TEXT")

                db.execSQL(
                    "CREATE TABLE IF NOT EXISTS `chunks_new` (" +
                        "`id` INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, " +
                        "`documentId` INTEGER NOT NULL, " +
                        "`pageNumber` INTEGER NOT NULL, " +
                        "`text` TEXT NOT NULL, " +
                        "`embedding` BLOB, " +
                        "`chunkIndex` INTEGER NOT NULL, " +
                        "`pageSpan` TEXT NOT NULL, " +
                        "`sectionTitle` TEXT, " +
                        "`paragraphIndex` INTEGER NOT NULL, " +
                        "`embeddingModelId` TEXT, " +
                        "`embeddingDim` INTEGER NOT NULL, " +
                        "FOREIGN KEY(`documentId`) REFERENCES `documents`(`id`) ON UPDATE NO ACTION ON DELETE CASCADE )"
                )
                db.execSQL(
                    """
                    INSERT INTO chunks_new (id, documentId, pageNumber, text, embedding, chunkIndex, pageSpan,
                                            sectionTitle, paragraphIndex, embeddingModelId, embeddingDim)
                    SELECT c.id, c.documentId, c.pageNumber, c.text, c.embedding, c.chunkIndex, c.pageSpan,
                           c.sectionTitle, c.paragraphIndex,
                           CASE WHEN c.embedding IS NULL THEN NULL ELSE '$LEGACY_EMBEDDING_MODEL_ID' END,
                           CASE WHEN c.embedding IS NULL THEN 0 ELSE length(c.embedding) / 4 END
                    FROM chunks c
                    WHERE c.documentId IN (SELECT id FROM documents)
                      AND c.id IN (SELECT MIN(id) FROM chunks GROUP BY documentId, chunkIndex, text)
                    """.trimIndent()
                )
                db.execSQL("DROP TABLE chunks")
                db.execSQL("ALTER TABLE chunks_new RENAME TO chunks")
                db.execSQL("CREATE INDEX IF NOT EXISTS `index_chunks_documentId` ON `chunks` (`documentId`)")

                db.execSQL("DELETE FROM messages WHERE conversationId NOT IN (SELECT id FROM conversations)")
                db.execSQL("CREATE INDEX IF NOT EXISTS `index_messages_conversationId` ON `messages` (`conversationId`)")
            }
        }

        val MIGRATION_3_4 = object : Migration(3, 4) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE messages ADD COLUMN metrics TEXT")
            }
        }
    }
}
