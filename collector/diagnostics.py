"""1-in-N sampled full query logging, separate from the always-on counter pipeline."""

import random

from collector import storage


class DiagnosticSampler:
    def __init__(self, db_path, sample_rate: float = 0.01):
        self.db_path = db_path
        self.sample_rate = sample_rate

    def maybe_log(self, record_id: str, op: str, shard: str, latency_ms: float):
        if random.random() < self.sample_rate:
            storage.write_query_log(self.db_path, record_id, op, shard, latency_ms)
