"""Stage 0/1 load generator: uniform-random reads/writes across all shards,
routed through the MetricsCollector so every request feeds Component 1's
per-record counters. A fraction of ops touch a related record group
(product/reviews/inventory for the same index) in one go, to populate the
co-access graph. Zipfian distribution and flash-sale spikes land in Stage 2.
"""

import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collector.middleware import MetricsCollector  # noqa: E402
from collector.window_worker import WindowWorker  # noqa: E402


def related_keys(i: int):
    return [f"product:{i}", f"reviews:{i}", f"inventory:{i}"]


def touch(collector: MetricsCollector, key: str, write_ratio: float, ops: int) -> str:
    shard = collector.cluster.shard_for_key(key)
    if random.random() < write_ratio:
        collector.set(key, f"payload-{ops}")
    else:
        collector.get(key)
    return shard


def run(num_keys: int, duration: float, rate: float, write_ratio: float, group_ratio: float, window_seconds: float):
    collector = MetricsCollector()
    worker = WindowWorker(collector, interval_seconds=window_seconds)
    worker.start()

    hits = Counter()
    latencies = []
    ops = 0
    interval = 1.0 / rate if rate > 0 else 0

    start = time.monotonic()
    end = start + duration
    while time.monotonic() < end:
        t0 = time.perf_counter()

        if random.random() < group_ratio:
            i = random.randrange(num_keys)
            keys = related_keys(i)
            for key in keys:
                shard = touch(collector, key, write_ratio, ops)
                hits[shard] += 1
                ops += 1
            collector.transaction(keys)
        else:
            key = f"product:{random.randrange(num_keys)}"
            shard = touch(collector, key, write_ratio, ops)
            hits[shard] += 1
            ops += 1

        latencies.append((time.perf_counter() - t0) * 1000)
        if interval:
            time.sleep(interval)

    worker.stop()
    counters, co_access = collector.snapshot_window()

    elapsed = time.monotonic() - start
    print(f"\n{ops} ops in {elapsed:.1f}s ({ops / elapsed:.1f} ops/sec)")
    print(f"avg latency: {sum(latencies) / len(latencies):.2f} ms")
    print("requests per shard:")
    for shard in collector.cluster.shard_names():
        print(f"  {shard}: {hits[shard]}")
    print(f"\nfinal partial window: {len(counters)} distinct records, {len(co_access)} co-access pairs")
    print(f"metrics db: {collector.db_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-shard load generator, routed through the metrics collector")
    parser.add_argument("--num-keys", type=int, default=50)
    parser.add_argument("--duration", type=float, default=10.0, help="seconds")
    parser.add_argument("--rate", type=float, default=20.0, help="ops/sec")
    parser.add_argument("--write-ratio", type=float, default=0.3)
    parser.add_argument("--group-ratio", type=float, default=0.15, help="fraction of ticks that touch a related record group")
    parser.add_argument("--window-seconds", type=float, default=5.0)
    args = parser.parse_args()

    run(args.num_keys, args.duration, args.rate, args.write_ratio, args.group_ratio, args.window_seconds)
