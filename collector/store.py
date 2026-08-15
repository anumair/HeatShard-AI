"""In-memory counters for Component 1 (Metrics Collection).

MetricsStore accumulates raw per-record and co-access counters between
window boundaries. snapshot_and_reset() atomically hands off the current
window's data and starts a fresh one -- callers (WindowWorker) are
responsible for persisting the snapshot.
"""

import threading
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class RecordCounters:
    access_count: int = 0
    write_count: int = 0
    cache_miss_count: int = 0
    latency_sum_ms: float = 0.0
    latency_count: int = 0

    @property
    def avg_latency_ms(self) -> float:
        return self.latency_sum_ms / self.latency_count if self.latency_count else 0.0


class MetricsStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._counters = defaultdict(RecordCounters)
        self._co_access = defaultdict(int)

    def record_access(self, record_id: str, is_write: bool, latency_ms: float, cache_miss: bool):
        with self._lock:
            c = self._counters[record_id]
            c.access_count += 1
            if is_write:
                c.write_count += 1
            if cache_miss:
                c.cache_miss_count += 1
            c.latency_sum_ms += latency_ms
            c.latency_count += 1

    def record_co_access(self, record_ids):
        """Mark a set of records as touched within the same simulated transaction."""
        uniq = sorted(set(record_ids))
        if len(uniq) < 2:
            return
        with self._lock:
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    self._co_access[(uniq[i], uniq[j])] += 1

    def snapshot_and_reset(self):
        with self._lock:
            counters_snapshot = dict(self._counters)
            co_access_snapshot = dict(self._co_access)
            self._counters = defaultdict(RecordCounters)
            self._co_access = defaultdict(int)
        return counters_snapshot, co_access_snapshot
