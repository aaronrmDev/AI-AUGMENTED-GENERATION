from src.cag.domain.entities import BatchRequest
from src.cag.domain.ports import BatchScheduler


class FIFOBatchScheduler(BatchScheduler):
    # The baseline CAG.md implicitly argues against: treat every request
    # as independent and batch them purely in arrival order. Requests
    # that happen to share a long prefix are only grouped together when
    # they happen to arrive together, so in any realistic interleaved
    # arrival pattern the same prefix gets prefilled once per batch it
    # is scattered across.
    def form_batches(
        self, pending: list[BatchRequest], max_batch_size: int
    ) -> list[list[BatchRequest]]:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        return [
            pending[start : start + max_batch_size]
            for start in range(0, len(pending), max_batch_size)
        ]
