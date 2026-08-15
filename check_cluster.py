"""Stage 0 sanity check: confirm read/write works on every shard and report latency."""

import time

from common.shard_client import ShardCluster


def main():
    cluster = ShardCluster()
    print(f"{'shard':<10} {'status':<8} {'write ms':<10} {'read ms':<10}")

    all_ok = True
    for name in cluster.shard_names():
        client = cluster.client(name)
        try:
            t0 = time.perf_counter()
            client.set("healthcheck", "ok")
            write_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            value = client.get("healthcheck")
            read_ms = (time.perf_counter() - t0) * 1000

            status = "OK" if value == "ok" else "BAD VALUE"
            print(f"{name:<10} {status:<8} {write_ms:<10.2f} {read_ms:<10.2f}")
        except Exception as exc:
            all_ok = False
            print(f"{name:<10} {'FAIL':<8} {exc}")

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
