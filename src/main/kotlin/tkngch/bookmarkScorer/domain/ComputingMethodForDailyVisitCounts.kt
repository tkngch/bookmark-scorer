package tkngch.bookmarkScorer.domain

import java.time.LocalDate
import java.time.temporal.ChronoUnit
import kotlin.math.pow

internal interface ComputingMethodForDailyVisitCounts {
    private fun getToday(): LocalDate = LocalDate.now()

    fun inferToday(records: DailyVisitCounts) = inferOnChosenDate(records, getToday())

    fun meanSquaredErrorOnMostRecentDate(records: DailyVisitCounts): Double {
        val mostRecentDate = records.asIterable().map { it.key.date }.maxOrNull() ?: getToday()
        val inferred = inferOnChosenDate(records.filterKeys { it.date < mostRecentDate }, mostRecentDate)
        val error = inferred.asIterable().map {
            records.getOrDefault(VisitDate(bookmarkId = it.key, date = mostRecentDate), 0) - it.value
        }
        return error.map { it.pow(2) }.sum() / error.size
    }

    fun inferOnChosenDate(records: DailyVisitCounts, date: LocalDate): InferredCounts
}

internal class AverageVisitCounts : ComputingMethodForDailyVisitCounts {
    override fun inferOnChosenDate(records: DailyVisitCounts, date: LocalDate): InferredCounts {
        val pastRecords = records.filterKeys { it.date < date }

        val countsByBookmark: Map<BookmarkId, Double> =
            pastRecords.asIterable().groupBy({ it.key.bookmarkId }, { it.value }).asIterable().map {
                Pair(it.key, it.value.sum().toDouble())
            }.toMap()

        val nDatesByBookmark: Map<BookmarkId, Double> =
            pastRecords.keys.groupBy({ it.bookmarkId }, { it.date }).asIterable().map {
                // ChronoUnit.DAYS.between gives the number of days (upper bound exclusive). Plus 1 to make it inclusive.
                Pair(it.key, ChronoUnit.DAYS.between(it.value.minOrNull()!!, it.value.maxOrNull()!!).toDouble() + 1)
            }.toMap()

        val averageCountsByBookmark: Map<BookmarkId, Double> =
            countsByBookmark.asIterable().map { Pair(it.key, it.value / nDatesByBookmark[it.key]!!) }.toMap()

        val bookmarkIds = records.keys.map { it.bookmarkId }.distinct()

        return bookmarkIds.map { bookmarkId: BookmarkId ->
            Pair(bookmarkId, averageCountsByBookmark.getOrDefault(bookmarkId, 0.0))
        }.toMap()
    }
}
