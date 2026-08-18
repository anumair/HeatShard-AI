"""CLI to inspect the per-window heat score history for a record."""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collector.storage import DEFAULT_DB_PATH  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Inspect the heat score history for a record")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--record", required=True)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM heat_windows WHERE record_id = ? ORDER BY window_start ASC",
        (args.record,),
    ).fetchall()
    conn.close()

    if not rows:
        print(f"no heat data for {args.record} -- run predictor/compute_heat.py first")
        return

    scores = [r["heat_score"] for r in rows]
    scale = max(max(scores, default=1.0), 1e-6)

    print(f"{args.record} heat score per window:")
    for r in rows:
        ts = time.strftime("%H:%M:%S", time.localtime(r["window_start"]))
        bar_len = max(0, int((r["heat_score"] / scale) * 40))
        bar = "#" * bar_len
        print(f"  {ts}  {r['heat_score']:>7.3f}  {bar}")


if __name__ == "__main__":
    main()
