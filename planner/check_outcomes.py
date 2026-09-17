"""Stage 6 outcome tracking: after a relocation plan would have been
"executed," check whether each relocated record actually became hot in
the window that followed -- feeding that signal back into Stage 4's
adaptive confidence threshold exactly like a flagged prediction's
outcome would be, so a plan built on unreliable predictions makes the
system more conservative going forward.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collector.storage import DEFAULT_DB_PATH  # noqa: E402
from planner import storage as planner_storage  # noqa: E402
from predictor.confidence import AdaptiveThreshold  # noqa: E402
from predictor.heat_index import SPIKE_HISTORY_WINDOWS, SPIKE_MULTIPLIER  # noqa: E402


def was_actually_hot(db_path, record_id: str, window_start: float) -> bool:
    """Same rule as HeatIndex.is_spike, applied directly to recorded
    history: did access_count exceed SPIKE_MULTIPLIER times the rolling
    baseline in the window right after `window_start`?"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT window_start, access_count FROM metric_windows WHERE record_id = ? ORDER BY window_start ASC",
        (record_id,),
    ).fetchall()
    conn.close()

    series = [(r["window_start"], r["access_count"]) for r in rows]
    idx = next((i for i, (ws, _) in enumerate(series) if ws == window_start), None)
    if idx is None or idx + 1 >= len(series):
        return False  # plan's window not found, or no later window recorded yet

    history = [count for _, count in series[max(0, idx - SPIKE_HISTORY_WINDOWS + 1): idx + 1]]
    if len(history) < 2:
        return False
    baseline = sum(history) / len(history)
    next_access = series[idx + 1][1]
    return next_access > SPIKE_MULTIPLIER * max(baseline, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Check relocation-plan outcomes and update the adaptive threshold")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--initial-threshold", type=float, default=None)
    args = parser.parse_args()

    planner_storage.init_db(args.db)
    pending = planner_storage.unchecked_plans(args.db)
    if not pending:
        print("no unchecked relocation plans -- run planner/run_relocation.py first")
        return

    threshold = AdaptiveThreshold(initial=args.initial_threshold) if args.initial_threshold is not None else AdaptiveThreshold()

    print(f"{'record_id':<45} {'predicted':<10} {'actual':<8}")
    for row in pending:
        was_hot = was_actually_hot(args.db, row["record_id"], row["window_start"])
        planner_storage.mark_outcome(args.db, row["plan_id"], was_hot)
        threshold.record_outcome(was_hot)
        print(f"{row['record_id']:<45} {'hot':<10} {'hot' if was_hot else 'not hot':<8}")

    print(f"\nchecked {len(pending)} relocation outcome(s)")
    print(f"adaptive threshold after these outcomes: {threshold.threshold:.2f}")
    for n, t in threshold.history:
        print(f"  after {n} outcomes -> {t:.2f}")


if __name__ == "__main__":
    main()
