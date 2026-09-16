"""General-purpose load generator: reads/writes across all shards, routed
through the MetricsCollector so every request feeds Component 1's
per-record counters. Supports uniform or Zipfian key selection, and either
the synthetic key space (product:i / reviews:i / inventory:i, default) or
real product ids from the Olist dataset (--key-source olist). A fraction
of ops touch a related record group in one go, to populate the co-access
graph.

For a repeatable, scripted flash-sale scenario (Stage 2's actual
deliverable), see flash_sale_scenario.py -- this script is the general
manual/ad-hoc traffic driver.
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
from common.keyspace import KeySpace  # noqa: E402
from simulator.zipf import ZipfSampler  # noqa: E402


def touch(collector: MetricsCollector, key: str, write_ratio: float, ops: int) -> str:
    shard = collector.cluster.shard_for_key(key)
    if random.random() < write_ratio:
        collector.set(key, f"payload-{ops}")
    else:
        collector.get(key)
    return shard


def make_key_picker(num_keys: int, distribution: str, skew: float):
    if distribution == "zipf":
        sampler = ZipfSampler(num_keys, skew=skew)
        return lambda: sampler.sample()
    return lambda: random.randrange(num_keys)


def run(
    num_keys: int,
    duration: float,
    rate: float,
    write_ratio: float,
    group_ratio: float,
    window_seconds: float,
    distribution: str,
    skew: float,
    key_source: str,
):
    key_space = KeySpace.synthetic(num_keys) if key_source == "synthetic" else KeySpace.from_olist(num_keys=num_keys)
    num_keys = key_space.num_keys

    collector = MetricsCollector()
    worker = WindowWorker(collector, interval_seconds=window_seconds)
    worker.start()

    pick_rank = make_key_picker(num_keys, distribution, skew)

    hits = Counter()
    latencies = []
    ops = 0
    interval = 1.0 / rate if rate > 0 else 0

    start = time.monotonic()
    end = start + duration
    while time.monotonic() < end:
        t0 = time.perf_counter()

        if random.random() < group_ratio:
            i = pick_rank()
            keys = key_space.related_keys(i)
            for key in keys:
                shard = touch(collector, key, write_ratio, ops)
                hits[shard] += 1
                ops += 1
            collector.transaction(keys)
        else:
            key = key_space.record_id(pick_rank())
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
    parser.add_argument("--distribution", choices=["uniform", "zipf"], default="uniform")
    parser.add_argument("--skew", type=float, default=1.2, help="zipf skew parameter (higher = more skewed)")
    parser.add_argument("--key-source", choices=["synthetic", "olist"], default="synthetic")
    args = parser.parse_args()

    run(
        args.num_keys,
        args.duration,
        args.rate,
        args.write_ratio,
        args.group_ratio,
        args.window_seconds,
        args.distribution,
        args.skew,
        args.key_source,
    )
