package com.example.offlinedocai.data.database

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import com.example.offlinedocai.data.database.dao.ChatDao
import com.example.offlinedocai.data.database.dao.DocumentDao
import com.example.offlinedocai.data.database.entities.ConversationEntity
import com.example.offlinedocai.data.database.entities.DocumentChunkEntity
import com.example.offlinedocai.data.database.entities.DocumentEntity
import com.example.offlinedocai.data.database.entities.MessageEntity

@Database(
    entities = [
        DocumentEntity::class,
        DocumentChunkEntity::class,
        ConversationEntity::class,
        MessageEntity::class
    ],
    version = 1,
    exportSchema = false
)
abstract class AppDatabase : RoomDatabase() {
    abstract fun documentDao(): DocumentDao
    abstract fun chatDao(): ChatDao

    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null

        fun getDatabase(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "offline_doc_ai_database"
                )
                    .fallbackToDestructiveMigration()
                    .build()
                INSTANCE = instance
                instance
            }
        }
    }
}
