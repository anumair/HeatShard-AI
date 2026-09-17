"""Stage 7: baselines & benchmarking.

Runs all three systems against the same recorded scenario and computes
the methodology's defined metrics (Section 4.3): data movement volume,
relocation precision, relocation recall, false-positive rate, and
post-relocation load variance.

  static     - no rebalancing, ever
  reactive   - reactive threshold-based whole-shard migration
               (planner/reactive_baseline.py), triggered only once
               OBSERVED shard load actually crosses a fixed threshold --
               no predictions, so it can only ever act after a shard is
               already overloaded
  heatshard  - Stage 6's targeted, predictive relocation plan
               (planner/run_relocation.py's compute_plan()), which acts
               as soon as a prediction crosses the probability floor,
               typically before the spike has fully landed

Each system is evaluated at ITS OWN natural decision point rather than
one shared snapshot: HeatShard's earliest actionable prediction window,
and the reactive baseline's earliest window where a shard's rolling
load actually crosses the threshold (scanned chronologically). The gap
between those two timestamps is the headline "how much earlier did
prediction beat reaction" number. Static uses HeatShard's decision-time
snapshot as its unmoved reference point, so its "before" state is a
fair apples-to-apples baseline for what doing nothing would have looked
like at that same moment.

Ground truth ("did a record actually become hot") reuses HeatIndex's
own is_spike rule, scanned across the whole recorded scenario.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from collector.storage import DEFAULT_DB_PATH  # noqa: E402
from common.shard_client import ShardCluster  # noqa: E402
from planner.reactive_baseline import DEFAULT_THRESHOLD_MULTIPLIER, reactive_plan  # noqa: E402
from planner.run_relocation import MIN_CANDIDATE_PROBABILITY, compute_plan  # noqa: E402
from predictor.heat_index import SPIKE_HISTORY_WINDOWS, SPIKE_MULTIPLIER  # noqa: E402

DEFAULT_OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "evaluation.png"


def ever_hot_records(db_path) -> set:
    """Every record that actually spiked (is_spike ground-truth rule) at
    any point in the recorded scenario."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT record_id, window_start, access_count FROM metric_windows ORDER BY record_id, window_start"
    ).fetchall()
    conn.close()

    by_record = {}
    for r in rows:
        by_record.setdefault(r["record_id"], []).append(r["access_count"])

    hot = set()
    for record_id, series in by_record.items():
        for i in range(1, len(series)):
            history = series[max(0, i - SPIKE_HISTORY_WINDOWS): i]
            if len(history) < 2:
                continue
            baseline = sum(history) / len(history)
            if series[i] > SPIKE_MULTIPLIER * max(baseline, 1.0):
                hot.add(record_id)
                break
    return hot


def load_at_window(db_path, up_to_window_start, cluster, window_lookback=5):
    """Per-shard/per-record load rolled up over the `window_lookback`
    windows ending at (and including) `up_to_window_start`."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    windows = [
        r["window_start"]
        for r in conn.execute(
            "SELECT DISTINCT window_start FROM metric_windows WHERE window_start <= ? ORDER BY window_start DESC LIMIT ?",
            (up_to_window_start, window_lookback),
        ).fetchall()
    ]
    if not windows:
        conn.close()
        return {}, {}
    placeholders = ",".join("?" * len(windows))
    rows = conn.execute(
        f"SELECT record_id, SUM(access_count) as total FROM metric_windows WHERE window_start IN ({placeholders}) GROUP BY record_id",
        windows,
    ).fetchall()
    conn.close()

    record_load = {r["record_id"]: float(r["total"]) for r in rows}
    shard_load = {name: 0.0 for name in cluster.shard_names()}
    for record_id, load in record_load.items():
        shard_load[cluster.shard_for_key(record_id)] += load
    return shard_load, record_load


def find_reactive_trigger(db_path, cluster, threshold_multiplier, window_lookback=5):
    """Scans the scenario chronologically for the first window where a
    shard's rolling load crosses the threshold. Returns
    (window_start, shard_load, record_load), or (None, {}, {}) if it
    never triggers in this scenario."""
    conn = sqlite3.connect(db_path)
    windows = [r[0] for r in conn.execute("SELECT DISTINCT window_start FROM metric_windows ORDER BY window_start ASC").fetchall()]
    conn.close()

    for window_start in windows:
        shard_load, record_load = load_at_window(db_path, window_start, cluster, window_lookback)
        if not shard_load:
            continue
        avg_load = sum(shard_load.values()) / len(shard_load)
        if any(load > avg_load * threshold_multiplier for load in shard_load.values()):
            return window_start, shard_load, record_load
    return None, {}, {}


def variance(loads: dict) -> float:
    return float(np.var(list(loads.values()))) if loads else 0.0


def hypothetical_loads(shard_load: dict, moves: list, record_load: dict) -> dict:
    hypothetical = dict(shard_load)
    for record_id, source, destination in moves:
        load = record_load.get(record_id, 0.0)
        hypothetical[source] = hypothetical.get(source, 0.0) - load
        hypothetical[destination] = hypothetical.get(destination, 0.0) + load
    return hypothetical


def evaluate_system(name, moves, shard_load, record_load, hot_records):
    moved = {m[0] for m in moves}
    tp = moved & hot_records
    fp = moved - hot_records

    return {
        "name": name,
        "data_movement": len(moved),
        "precision": len(tp) / len(moved) if moved else 0.0,
        "recall": len(tp) / len(hot_records) if hot_records else 0.0,
        "false_positive_rate": len(fp) / len(moved) if moved else 0.0,
        "variance_before": variance(shard_load),
        "variance_after": variance(hypothetical_loads(shard_load, moves, record_load)),
    }


def plot_comparison(results: list, out_path):
    names = [r["name"] for r in results]
    colors = ["#888888", "#DD8452", "#4C72B0"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    bar_metrics = [
        ("data_movement", "Data movement (records)", axes[0][0]),
        ("precision", "Relocation precision", axes[0][1]),
        ("recall", "Relocation recall", axes[0][2]),
        ("false_positive_rate", "False-positive rate", axes[1][0]),
    ]
    for key, title, ax in bar_metrics:
        ax.bar(names, [r[key] for r in results], color=colors[: len(results)])
        ax.set_title(title)

    ax = axes[1][1]
    width = 0.35
    x = np.arange(len(names))
    ax.bar(x - width / 2, [r["variance_before"] for r in results], width, label="before")
    ax.bar(x + width / 2, [r["variance_after"] for r in results], width, label="after")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("Load variance (before vs after each system's own move)")
    ax.legend()

    axes[1][2].axis("off")

    fig.suptitle("HeatShard vs. baselines (Stage 7)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description="Stage 7: evaluate HeatShard against static and reactive baselines")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--min-probability", type=float, default=MIN_CANDIDATE_PROBABILITY)
    parser.add_argument("--threshold-multiplier", type=float, default=DEFAULT_THRESHOLD_MULTIPLIER)
    parser.add_argument("--window-lookback", type=int, default=5)
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    args = parser.parse_args()

    cluster = ShardCluster()
    hot_records = ever_hot_records(args.db)
    print(f"ground truth: {len(hot_records)} record(s) actually became hot at some point in this scenario")

    heatshard_result = compute_plan(args.db, min_probability=args.min_probability, cluster=cluster, window_lookback=args.window_lookback)
    if heatshard_result is None:
        print("no HeatShard predictions found -- run predictor/run_prediction.py first")
        return

    heatshard_window = heatshard_result["window_start"]
    heatshard_moves = [(m.record_id, m.source_shard, m.destination_shard) for m in heatshard_result["moves"]]

    reactive_window, reactive_shard_load, reactive_record_load = find_reactive_trigger(
        args.db, cluster, args.threshold_multiplier, window_lookback=args.window_lookback
    )
    reactive_result = (
        reactive_plan(reactive_shard_load, reactive_record_load, cluster, args.threshold_multiplier)
        if reactive_window is not None
        else {"moves": []}
    )

    print(f"\nHeatShard decided at window_start={heatshard_window}")
    if reactive_window is not None:
        lead_time = reactive_window - heatshard_window
        print(f"reactive threshold first crossed at window_start={reactive_window}")
        print(f"HeatShard acted {lead_time:.0f}s earlier than the reactive baseline would have noticed anything")
    else:
        print("reactive threshold never crossed in this scenario -- it would have done nothing at all")

    systems = [
        ("static", [], heatshard_result["loads"], heatshard_result["record_load"]),
        ("reactive", reactive_result["moves"], reactive_shard_load or heatshard_result["loads"], reactive_record_load or heatshard_result["record_load"]),
        ("heatshard", heatshard_moves, heatshard_result["loads"], heatshard_result["record_load"]),
    ]

    results = [evaluate_system(name, moves, shard_load, record_load, hot_records) for name, moves, shard_load, record_load in systems]

    print(f"\n{'system':<12} {'moved':>7} {'precision':>10} {'recall':>8} {'FP rate':>9} {'var before':>11} {'var after':>10}")
    for r in results:
        print(
            f"{r['name']:<12} {r['data_movement']:>7} {r['precision']:>10.2f} {r['recall']:>8.2f} "
            f"{r['false_positive_rate']:>9.2f} {r['variance_before']:>11.0f} {r['variance_after']:>10.0f}"
        )

    plot_comparison(results, args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
