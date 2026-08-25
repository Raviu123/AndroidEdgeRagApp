package com.siddu.offlinedocai.data

import androidx.room.TypeConverter
import java.util.Date

object Converters {
    @TypeConverter
    fun fromTimestamp(value: Long?): Date? = value?.let { Date(it) }

    @TypeConverter
    fun dateToTimestamp(date: Date?): Long? = date?.time
}
