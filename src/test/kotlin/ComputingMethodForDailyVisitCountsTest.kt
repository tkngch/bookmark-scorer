package tkngch.bookmarkScorer.domain

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.Test
import java.time.LocalDate

class ComputingMethodForDailyVisitCountsTest {

    @Nested
    inner class AverageVisitCountsTest {
        private val instance = AverageVisitCounts()

        @Test
        fun `infer zero when a bookmark has no visit records`() {
            val bookmarkId = "bookmarkId"
            val records = mapOf(VisitDate(bookmarkId, LocalDate.now()) to 0)
            assertEquals(0.0, instance.inferToday(records).get(bookmarkId))
        }

        @Test
        fun `infer one when a bookmark has only one record from yesterday`() {
            val bookmarkId = "bookmarkId"
            val records = mapOf(VisitDate(bookmarkId, LocalDate.now().plusDays(-1)) to 1)
            assertEquals(1.0, instance.inferToday(records).get(bookmarkId))
        }

        @Test
        fun `ignore future visit dates`() {
            val bookmarkId = "bookmarkId"
            val tomorrow = LocalDate.now().plusDays(1)
            val records = mapOf(VisitDate(bookmarkId, tomorrow) to 1)
            assertEquals(0.0, instance.inferToday(records).get(bookmarkId))
        }

        @Test
        fun `infer several scores in one go`() {
            val records = mapOf(
                VisitDate("id01", LocalDate.now().plusDays(-1)) to 1,
                VisitDate("id01", LocalDate.now().plusDays(-2)) to 3,
                VisitDate("id01", LocalDate.now().plusDays(-3)) to 2,
                VisitDate("id02", LocalDate.now().plusDays(-4)) to 4,
                VisitDate("id02", LocalDate.now().plusDays(-5)) to 1,
                VisitDate("id03", LocalDate.now().plusDays(-1)) to 2,
                VisitDate("id03", LocalDate.now().plusDays(-2)) to 4,

            )
            val scores = instance.inferToday(records)
            assertEquals(records.keys.map { it.bookmarkId }.distinct().size, scores.keys.size)
            assertFalse(scores.asIterable().map { it.value < 0.0 }.reduce { acc, bool -> acc || bool })
        }
    }
}
