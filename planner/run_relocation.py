"""Stage 6 CLI: turn Stage 4's predictions + Stage 5's dependency graph +
current shard load into a concrete relocation plan, and compare its data
movement volume against a naive whole-shard-migration baseline -- the
headline Stage 6 checkpoint result.

Plans from the EARLIEST window that has any candidate above the
probability floor, rather than the latest recorded window -- the whole
point is to act as soon as there's a usable signal, before the spike
fully lands, and it leaves later windows available for check_outcomes.py
(Stage 6) and evaluate.py (Stage 7) to judge the plan against.

compute_plan() is the reusable core; Stage 7's evaluate.py calls it
directly to get HeatShard's side of the three-way baseline comparison
without duplicating this logic.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collector.storage import DEFAULT_DB_PATH  # noqa: E402
from common.shard_client import ShardCluster  # noqa: E402
from planner import storage as planner_storage  # noqa: E402
from planner.baseline import whole_shard_migration_volume  # noqa: E402
from planner.dependency_graph import DependencyGraph  # noqa: E402
from planner.relocation_planner import RelocationCandidate, RelocationPlanner  # noqa: E402
from planner.shard_load import record_recent_load, shard_loads  # noqa: E402

MIN_CANDIDATE_PROBABILITY = 0.3  # floor for even considering a record; ExpectedValue does the real filtering


def pick_target_window(db_path, min_probability: float):
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT MIN(window_start) as ws FROM predictions WHERE p_ensemble >= ?", (min_probability,)
    ).fetchone()
    conn.close()
    return row[0] if row else None


def compute_plan(
    db_path,
    window_start=None,
    min_probability: float = MIN_CANDIDATE_PROBABILITY,
    min_co_access: int = 3,
    window_lookback: int = 5,
    cluster: ShardCluster = None,
):
    """Returns a dict: window_start, candidates, moves, loads, record_load,
    cluster, graph -- or None if no window has an actionable prediction."""
    cluster = cluster or ShardCluster()
    graph = DependencyGraph.build(db_path, min_co_access=min_co_access)
    record_load = record_recent_load(db_path, window_lookback=window_lookback)
    loads = shard_loads(record_load, cluster)

    window_start = window_start or pick_target_window(db_path, min_probability)
    if window_start is None:
        return None

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    prediction_rows = conn.execute(
        "SELECT * FROM predictions WHERE window_start = ? AND p_ensemble >= ?",
        (window_start, min_probability),
    ).fetchall()
    conn.close()

    candidates = [
        RelocationCandidate(
            record_id=row["record_id"],
            source_shard=cluster.shard_for_key(row["record_id"]),
            p_hotspot=row["p_ensemble"],
            confidence=row["confidence"],
            predicted_load=record_load.get(row["record_id"], 1.0),
        )
        for row in prediction_rows
    ]

    planner = RelocationPlanner(cluster=cluster, graph=graph)
    moves = planner.plan(candidates, loads)

    return {
        "window_start": window_start,
        "candidates": candidates,
        "moves": moves,
        "loads": loads,
        "record_load": record_load,
        "cluster": cluster,
        "graph": graph,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute a relocation plan from Stage 4's predictions")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--window-start", type=float, default=None, help="plan from this exact window (default: earliest actionable one)")
    parser.add_argument("--min-probability", type=float, default=MIN_CANDIDATE_PROBABILITY)
    parser.add_argument("--min-co-access", type=int, default=3)
    parser.add_argument("--window-lookback", type=int, default=5)
    args = parser.parse_args()

    planner_storage.init_db(args.db)

    result = compute_plan(
        args.db,
        window_start=args.window_start,
        min_probability=args.min_probability,
        min_co_access=args.min_co_access,
        window_lookback=args.window_lookback,
    )

    if result is None:
        print("no predictions above the probability floor -- run predictor/run_prediction.py first")
        return

    print("current shard load:")
    for shard, load in result["loads"].items():
        print(f"  {shard}: {load:.0f}")

    candidates, moves = result["candidates"], result["moves"]
    print(f"\nplanning from window_start={result['window_start']} ({len(candidates)} candidate(s) above p >= {args.min_probability})")

    planner_storage.write_plan(args.db, result["window_start"], moves)

    print(f"\n{len(candidates)} candidate(s) considered, {len(moves)} move(s) in the plan\n")
    if moves:
        print(f"{'record_id':<45} {'from':<10} {'to':<10} {'EV':>7} {'benefit':>8} {'cost':>7} {'conf':>6}")
        for m in moves:
            print(
                f"{m.record_id:<45} {m.source_shard:<10} {m.destination_shard:<10} "
                f"{m.expected_value:>7.2f} {m.predicted_benefit:>8.2f} {m.predicted_cost:>7.2f} {m.confidence:>6.2f}"
            )
    else:
        print("no candidate cleared a positive expected value -- no relocation recommended")

    hot_record_ids = {c.record_id for c in candidates}
    baseline = whole_shard_migration_volume(result["record_load"], hot_record_ids, result["cluster"])

    print("\n--- data movement comparison (Stage 6 checkpoint) ---")
    print(f"HeatShard (targeted):  {len(moves)} record(s) relocated")
    print(
        f"Naive (whole-shard):   {baseline['records_moved']} record(s) relocated "
        f"({len(baseline['affected_shards'])} shard(s) fully migrated: {baseline['affected_shards']})"
    )
    if baseline["records_moved"] > 0:
        reduction = 100 * (1 - len(moves) / baseline["records_moved"])
        print(f"reduction: {reduction:.1f}% less data movement than the naive baseline")


if __name__ == "__main__":
    main()
