"""CLI to inspect the time-windowed metrics table produced by Stage 1."""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collector.storage import DEFAULT_DB_PATH  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Inspect the windowed metrics table")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--record", help="filter to one record_id")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    if args.record:
        rows = conn.execute(
            "SELECT * FROM metric_windows WHERE record_id = ? ORDER BY window_start DESC LIMIT ?",
            (args.record, args.limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM metric_windows ORDER BY window_start DESC, access_count DESC LIMIT ?",
            (args.limit,),
        ).fetchall()

    header = f"{'window_start':<20} {'record_id':<16} {'access':<7} {'write':<7} {'miss':<6} {'avg_ms':<8}"
    print(header)
    for r in rows:
        ts = time.strftime("%H:%M:%S", time.localtime(r["window_start"]))
        print(
            f"{ts:<20} {r['record_id']:<16} {r['access_count']:<7} {r['write_count']:<7} "
            f"{r['cache_miss_count']:<6} {r['avg_latency_ms']:<8.2f}"
        )

    co_rows = conn.execute(
        "SELECT record_a, record_b, SUM(count) as total FROM co_access_windows "
        "GROUP BY record_a, record_b ORDER BY total DESC LIMIT 10"
    ).fetchall()
    if co_rows:
        print("\ntop co-access pairs:")
        for r in co_rows:
            print(f"  {r['record_a']} <-> {r['record_b']}: {r['total']}")


if __name__ == "__main__":
    main()
