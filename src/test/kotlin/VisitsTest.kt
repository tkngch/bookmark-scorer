package tkngch.bookmarkScorer.domain

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.Test
import java.time.Instant

class VisitsTest {

    @Nested
    inner class AverageDailyVisits {

        @Test
        fun `infer zero when a bookmark has no visit records`() {
            val visit = Visit(bookmarkId = "bookmarkId", visitDates = listOf())
            assertEquals(0.0, visit.inferAverageDailyVisits())
        }

        @Test
        fun `infer one when a bookmark has only one record from today`() {
            val visit = Visit(bookmarkId = "bookmarkId", visitDates = listOf(Instant.now()))
            assertEquals(1.0, visit.inferAverageDailyVisits())
        }

        @Test
        fun `ignore future visit dates`() {
            val tomorrow = Instant.now().plusSeconds(24 * 60 * 60)
            val visit = Visit(bookmarkId = "bookmarkId", visitDates = listOf(tomorrow))
            assertEquals(0.0, visit.inferAverageDailyVisits())
        }

        @Test
        fun `infer several scores in one go`() {
            val visits = Visits(
                listOf(
                    Visit(bookmarkId = "id01", visitDates = listOf()),
                    Visit(bookmarkId = "id02", visitDates = listOf(Instant.parse("2020-12-28T12:00:00Z"))),
                    Visit(bookmarkId = "id02", visitDates = listOf(Instant.parse("2020-12-02T10:00:00Z"), Instant.parse("2120-12-28T12:00:00Z")))
                )
            )
            val scores = visits.inferredAverageDailyVisits
            assertEquals(visits.records.size, scores.size)
            assertFalse(scores.map { it.score < 0.0 }.reduce { acc, bool -> acc || bool })
        }
    }
}
