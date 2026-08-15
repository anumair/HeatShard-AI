"""Thin proxy layer every simulated request passes through before hitting a shard.

Wraps ShardCluster: records per-record counters (Component 1), simulates
an app-level LRU cache to derive cache_miss_count, and samples 1-in-100
requests into the diagnostic query log.
"""

import time
from collections import OrderedDict

from collector import storage
from collector.diagnostics import DiagnosticSampler
from collector.store import MetricsStore
from common.config import CACHE_CAPACITY, DIAG_SAMPLE_RATE
from common.shard_client import ShardCluster


class LocalCache:
    """Per-process LRU used only to simulate app-level cache hit/miss for metrics."""

    def __init__(self, capacity=CACHE_CAPACITY):
        self.capacity = capacity
        self._data = OrderedDict()

    def get(self, key) -> bool:
        """Returns True on hit, False on miss. Touching always counts as a put."""
        hit = key in self._data
        if hit:
            self._data.move_to_end(key)
        return hit

    def put(self, key):
        self._data[key] = True
        self._data.move_to_end(key)
        if len(self._data) > self.capacity:
            self._data.popitem(last=False)


class MetricsCollector:
    def __init__(self, db_path=None, cache_capacity=CACHE_CAPACITY, diag_sample_rate=DIAG_SAMPLE_RATE):
        self.cluster = ShardCluster()
        self.store = MetricsStore()
        self.cache = LocalCache(cache_capacity)
        self.db_path = storage.init_db(db_path)
        self.diagnostics = DiagnosticSampler(self.db_path, diag_sample_rate)
        self._window_start = time.time()

    def get(self, key: str):
        shard = self.cluster.shard_for_key(key)
        cache_hit = self.cache.get(key)

        t0 = time.perf_counter()
        value = self.cluster.client(shard).get(key)
        latency_ms = (time.perf_counter() - t0) * 1000
        self.cache.put(key)

        self.store.record_access(key, is_write=False, latency_ms=latency_ms, cache_miss=not cache_hit)
        self.diagnostics.maybe_log(key, "GET", shard, latency_ms)
        return value

    def set(self, key: str, value):
        shard = self.cluster.shard_for_key(key)

        t0 = time.perf_counter()
        self.cluster.client(shard).set(key, value)
        latency_ms = (time.perf_counter() - t0) * 1000
        self.cache.put(key)

        self.store.record_access(key, is_write=True, latency_ms=latency_ms, cache_miss=False)
        self.diagnostics.maybe_log(key, "SET", shard, latency_ms)

    def transaction(self, keys):
        """Mark a group of keys as touched together, feeding the co-access graph."""
        self.store.record_co_access(keys)

    def snapshot_window(self):
        """Flush the current window's counters to SQLite and start a new window."""
        window_end = time.time()
        counters, co_access = self.store.snapshot_and_reset()
        storage.write_window(self.db_path, self._window_start, window_end, counters)
        storage.write_co_access(self.db_path, self._window_start, window_end, co_access)
        self._window_start = window_end
        return counters, co_access
