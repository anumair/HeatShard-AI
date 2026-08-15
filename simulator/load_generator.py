"""Stage 0 load generator: uniform-random reads/writes across all shards.

Just enough to prove requests land on different nodes. Zipfian distribution
and flash-sale spike injection are added in Stage 2.
"""

import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.shard_client import ShardCluster  # noqa: E402


def run(num_keys: int, duration: float, rate: float, write_ratio: float):
    cluster = ShardCluster()
    keys = [f"product:{i}" for i in range(num_keys)]

    hits = Counter()
    latencies = []
    ops = 0
    interval = 1.0 / rate if rate > 0 else 0

    start = time.monotonic()
    end = start + duration
    while time.monotonic() < end:
        key = random.choice(keys)
        shard = cluster.shard_for_key(key)
        client = cluster.client(shard)

        t0 = time.perf_counter()
        if random.random() < write_ratio:
            client.set(key, f"payload-{ops}")
        else:
            client.get(key)
        latencies.append((time.perf_counter() - t0) * 1000)

        hits[shard] += 1
        ops += 1
        if interval:
            time.sleep(interval)

    elapsed = time.monotonic() - start
    print(f"\n{ops} ops in {elapsed:.1f}s ({ops / elapsed:.1f} ops/sec)")
    print(f"avg latency: {sum(latencies) / len(latencies):.2f} ms")
    print("requests per shard:")
    for shard in cluster.shard_names():
        print(f"  {shard}: {hits[shard]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trivial cross-shard load generator")
    parser.add_argument("--num-keys", type=int, default=50)
    parser.add_argument("--duration", type=float, default=10.0, help="seconds")
    parser.add_argument("--rate", type=float, default=20.0, help="ops/sec")
    parser.add_argument("--write-ratio", type=float, default=0.3)
    args = parser.parse_args()

    run(args.num_keys, args.duration, args.rate, args.write_ratio)
