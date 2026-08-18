"""Render a PNG plot of heat score over time for one or more records --
the Stage 3 checkpoint test artifact: a clear rise leading into a flash
sale (thanks to the event signal) and decay afterward.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from collector.storage import DEFAULT_DB_PATH  # noqa: E402

DEFAULT_OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "heat_plot.png"


def fetch_series(db_path, record_id):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT window_start, heat_score FROM heat_windows WHERE record_id = ? ORDER BY window_start ASC",
        (record_id,),
    ).fetchall()
    conn.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Plot heat score over time for one or more records")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--record", action="append", required=True, help="repeatable")
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(10, 5))
    t_min = None
    plotted = 0
    for record_id in args.record:
        rows = fetch_series(args.db, record_id)
        if not rows:
            print(f"no heat data for {record_id}, skipping")
            continue
        if t_min is None:
            t_min = rows[0]["window_start"]
        xs = [r["window_start"] - t_min for r in rows]
        ys = [r["heat_score"] for r in rows]
        ax.plot(xs, ys, marker="o", label=record_id)
        plotted += 1

    if plotted == 0:
        print("nothing to plot -- run predictor/compute_heat.py first")
        return

    ax.set_xlabel("seconds since first window")
    ax.set_ylabel("heat score")
    ax.set_title("HeatShard: Adaptive Heat Index over time")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
