package tkngch.bookmarkScorer.domain

import java.time.Instant
import java.time.LocalDate
import java.time.ZoneOffset

data class Visits(val records: List<VisitInstant>) {
    private val asDailyVisitCounts: DailyVisitCounts by lazy {
        records.map {
            VisitDate(it.bookmarkId, LocalDate.ofInstant(it.instant, ZoneOffset.UTC))
        }.groupingBy { it }.eachCount()
    }

    private val averageDailyCounts: ComputingMethodForDailyVisitCounts by lazy {
        AverageVisitCounts()
    }
    fun inferAverageDailyCounts() = averageDailyCounts.inferToday(asDailyVisitCounts)

    private val inferenceModel: ComputingMethodForDailyVisitCounts by lazy {
        ModelForDailyVisitCounts()
    }
    fun inferDailyCounts() = inferenceModel.inferToday(asDailyVisitCounts)
}

data class VisitInstant(val bookmarkId: BookmarkId, val instant: Instant)
internal data class VisitDate(val bookmarkId: BookmarkId, val date: LocalDate)

typealias BookmarkId = String
typealias InferredCounts = Map<BookmarkId, InferredCount>
typealias InferredCount = Double

internal typealias VisitCount = Int
internal typealias DailyVisitCounts = Map<VisitDate, VisitCount>
