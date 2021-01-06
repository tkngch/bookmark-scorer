package tkngch.bookmarkScorer.domain

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.Test
import java.time.LocalDate

class ComputingMethodForDailyVisitCountsTest {

    @Nested
    inner class AverageVisitCountsTest {
        private val instance = AverageVisitCounts()
        private val today = LocalDate.now()

        @Test
        fun `infer zero when a bookmark has no visit records`() {
            val bookmarkId = "bookmarkId"
            val records = mapOf(VisitDate(bookmarkId, today) to 0)
            assertEquals(0.0, instance.inferToday(records)[bookmarkId])
        }

        @Test
        fun `infer one when a bookmark has only one record from yesterday`() {
            val bookmarkId = "bookmarkId"
            val records = mapOf(VisitDate(bookmarkId, today.plusDays(-1)) to 1)
            assertEquals(1.0, instance.inferToday(records)[bookmarkId])
        }

        @Test
        fun `ignore future visit dates`() {
            val bookmarkId = "bookmarkId"
            val tomorrow = today.plusDays(1)
            val records = mapOf(VisitDate(bookmarkId, tomorrow) to 1)
            assertEquals(0.0, instance.inferToday(records)[bookmarkId])
        }

        @Test
        fun `infer several scores in one go`() {
            val records = mapOf(
                VisitDate("id01", today.plusDays(-1)) to 1,
                VisitDate("id01", today.plusDays(-2)) to 3,
                VisitDate("id01", today.plusDays(-3)) to 2,
                VisitDate("id02", today.plusDays(-4)) to 4,
                VisitDate("id02", today.plusDays(-5)) to 1,
                VisitDate("id03", today.plusDays(-1)) to 2,
                VisitDate("id03", today.plusDays(-2)) to 4,

            )
            val scores = instance.inferToday(records)
            assertEquals(records.keys.map { it.bookmarkId }.distinct().size, scores.keys.size)
            assertFalse(
                scores.asIterable().map { it.value < 0.0 }.reduce { acc, bool -> acc || bool }
            )
        }
    }

    @Nested
    inner class PoissonRegressionTest {
        private val instance = PoissonRegression()
        private val today = LocalDate.now()

        @Test
        fun `infer obvious values`() {
            val ids = listOf(
                "f7fff973-686f-47d0-bc55-1a218e0d9173",
                "65752c72-7c80-448a-b99c-f8c9215c1197",
                "cb85e9c7-505c-4063-8244-70cc1081eac5",
                "2b14fc6b-a3cd-4356-bf53-5fda35a3556f",
                "54e79e4b-3da9-45a0-bf17-d17a8406ab7a",
                "9693c17d-d19d-4933-91aa-501fd0bfe0c0",
                "868976f0-6cf7-4f44-88e9-d2060f664b52"
            )
            val nDatesToConsider = 10
            // Generate the records such that the first BookmarkId is associated with
            // always-smallest visit-counts, and that the last bookmark-id is associated with
            // always-largest visit-counts.
            val records: DailyVisitCounts = ids.mapIndexed { index: Int, id: BookmarkId ->
                LongRange(start = 1, endInclusive = nDatesToConsider.toLong()).map { nBack: Long ->
                    Pair(VisitDate(id, today.plusDays(-nBack)), index + 1)
                }
            }.flatten().toMap()

            val scores = instance.inferToday(records)
            assertTrue(
                ids.zipWithNext().map { pair ->
                    scores[pair.first]!! < scores[pair.second]!!
                }.reduce { first, second -> first && second }
            )
        }

        @Test
        fun `infer known values`() {
            val records: DailyVisitCounts = mapOf(
                VisitDate("id01", today.plusDays(-100)) to 1
            )
            val inferred = instance.inferToday(records)
            assertEquals(0.0, inferred["id01"]!!)
        }
    }
}
