package tkngch.bookmarkScorer.domain

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.File.createTempFile
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

internal class ModelForDailyVisitCounts : ComputingMethodForDailyVisitCounts {
    private val model: Module by lazy {
        // libtorch cannot load a file if the file is located inside a jar file. So copy the file to
        // outside the jar before loading.
        val modelFile = File({}.javaClass.getResource("/model.pt").path)
        val tmpFile = createTempFile("bookmark-scorer-model-", ".pt")
        modelFile.copyTo(tmpFile, overwrite = true)
        Module.load(tmpFile.absolutePath.toString())
    }
    private val metadata: JsonElement by lazy {
        Json.parseToJsonElement({}.javaClass.getResource("/model_metadata.json").readText())
    }

    override fun inferToday(records: DailyVisitCounts): InferredCounts {
        val today = LocalDate.now()
        val nDatesToConsider: Long =
            metadata.jsonObject["hyper_parameters"]!!.jsonObject["n_days"]!!.jsonPrimitive.long

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
