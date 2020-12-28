package tkngch.bookmarkScorer.domain

import java.time.Instant
import java.time.LocalDate
import java.time.ZoneOffset
import java.time.temporal.ChronoUnit

data class Visits(val records: List<Visit>) {
    val inferredAverageDailyVisits = records.map { ComputedScore(bookmarkId = it.bookmarkId, score = it.inferAverageDailyVisits()) }
}

data class Visit(val bookmarkId: String, val visitDates: List<Instant>) {
    fun inferAverageDailyVisits(): Double {
        val now = Instant.now()
        val validVisitDates = visitDates.filter { it < now }
        val sinceDate: Instant = validVisitDates.sorted().firstOrNull() ?: now

        // ChronoUnit.DAYS.between gives the number of days (upper bound exclusive). Plus 1 to make it inclusive.
        val nDatesBetweenSinceDateAndToday = ChronoUnit.DAYS.between(getDate(sinceDate), getDate(now)) + 1

        return validVisitDates.size / nDatesBetweenSinceDateAndToday.toDouble()
    }

    private fun getDate(x: Instant) = LocalDate.ofInstant(x, ZoneOffset.UTC)
}
