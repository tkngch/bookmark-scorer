package tkngch.bookmarkScorer.domain

import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.time.LocalDate
import java.time.temporal.ChronoUnit
import kotlin.math.exp

internal interface ComputingMethodForDailyVisitCounts {
    fun inferToday(records: DailyVisitCounts): InferredCounts
}

internal class AverageVisitCounts : ComputingMethodForDailyVisitCounts {
    override fun inferToday(records: DailyVisitCounts): InferredCounts {
        val today = LocalDate.now()
        val pastRecords = records.filterKeys { it.date < today }

        val countsByBookmark: Map<BookmarkId, Double> =
            pastRecords.asIterable().groupBy({ it.key.bookmarkId }, { it.value }).asIterable().map {
                Pair(it.key, it.value.sum().toDouble())
            }.toMap()

        val nDatesByBookmark: Map<BookmarkId, Double> =
            pastRecords.keys.groupBy({ it.bookmarkId }, { it.date }).asIterable().map {
                // ChronoUnit.DAYS.between gives the number of days (upper bound exclusive). Plus 1 to make it inclusive.
                Pair(
                    it.key,
                    ChronoUnit.DAYS.between(
                        it.value.minOrNull()!!,
                        it.value.maxOrNull()!!
                    ).toDouble() + 1
                )
            }.toMap()

        val averageCountsByBookmark: Map<BookmarkId, Double> = countsByBookmark.asIterable().map {
            Pair(it.key, it.value / nDatesByBookmark[it.key]!!)
        }.toMap()

        val bookmarkIds = records.keys.map { it.bookmarkId }.distinct()

        return bookmarkIds.map { bookmarkId: BookmarkId ->
            Pair(bookmarkId, averageCountsByBookmark.getOrDefault(bookmarkId, 0.0))
        }.toMap()
    }
}

internal class PoissonRegression : ComputingMethodForDailyVisitCounts {
    private val model: Module by lazy {
        Module.load({}.javaClass.getResource("/poisson_regression.pt").path)
    }

    override fun inferToday(records: DailyVisitCounts): InferredCounts {
        val today = LocalDate.now()
        val nDatesToConsider: Long = 10 // Determined and required by the model.

        val allBookmarkIds = records.keys.map { it.bookmarkId }.distinct()
        val pastCounts: Map<BookmarkId, List<Int>> = allBookmarkIds.map { bookmarkId ->
            Pair(
                bookmarkId,
                LongRange(start = 1, endInclusive = nDatesToConsider).map { nDaysBack ->
                    records.getOrDefault(VisitDate(bookmarkId, today.plusDays(-nDaysBack)), 0)
                }
            )
        }.filter { it.second.sum() > 0 }.toMap()

        val bookmarkIds = pastCounts.keys.toList()
        val counts = bookmarkIds.map { bookmarkId ->
            pastCounts[bookmarkId]!!.map { it.toFloat() }
        }.flatten().toFloatArray()

        val shape: LongArray = listOf(bookmarkIds.size.toLong(), nDatesToConsider).toLongArray()
        val tensor: Tensor = Tensor.fromBlob(counts, shape)

        val output: IValue = model.forward(IValue.from(tensor))
        val inferred: List<Float> = output.toTensor().dataAsFloatArray.toList().map { exp(it) }

        val nonzero = bookmarkIds.mapIndexed { index: Int, bookmarkId: BookmarkId ->
            Pair(bookmarkId, inferred[index].toDouble())
        }.toMap()
        val zero = allBookmarkIds.filterNot { bookmarkIds.contains(it) }.map { bookmarkId ->
            Pair(bookmarkId, 0.0)
        }.toMap()
        return nonzero + zero
    }
}
