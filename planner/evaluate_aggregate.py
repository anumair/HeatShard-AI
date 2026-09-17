"""Stage 7, done properly: aggregate the static/reactive/HeatShard
comparison across many independent scenarios instead of reporting one
run's numbers.

A single scenario's precision/recall/lead-time vary a lot seed to seed
-- confirmed empirically during development, where HeatShard's
precision ranged from 0.0 to 1.0 and its lead time from 0s to 24s
across individual runs with nothing changed but the random seed.
Reporting "a representative run" is cherry-picking. This script runs N
independent, freshly-seeded scenarios of the same shape (never used to
train the model), evaluates all three systems on each exactly as
evaluate.py does for one run, and reports mean +/- std per metric so
the headline numbers are actually defensible.
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.shard_client import ShardCluster  # noqa: E402
from planner.evaluate import evaluate_system, ever_hot_records, find_reactive_trigger  # noqa: E402
from planner.reactive_baseline import DEFAULT_THRESHOLD_MULTIPLIER, reactive_plan  # noqa: E402
from planner.run_relocation import MIN_CANDIDATE_PROBABILITY, compute_plan  # noqa: E402
from simulator.flash_sale_scenario import ScenarioConfig  # noqa: E402
from simulator.flash_sale_scenario import run as run_scenario  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "data" / "eval_runs"
DEFAULT_OUT_PATH = PROJECT_ROOT / "data" / "evaluation_aggregate.png"

METRICS = ["data_movement", "precision", "recall", "false_positive_rate", "variance_before", "variance_after"]
SYSTEMS = ["static", "reactive", "heatshard"]
METRIC_TITLES = {
    "data_movement": "Data movement (records)",
    "precision": "Relocation precision",
    "recall": "Relocation recall",
    "false_positive_rate": "False-positive rate",
    "variance_before": "Variance before",
    "variance_after": "Variance after",
}


def compute_pipeline(db_path, events_path):
    """Run Stage 3 + Stage 4 against one recorded scenario via the exact
    same tested CLI scripts the rest of the project uses, so this
    evaluation exercises real code paths, not a reimplementation."""
    for script in ("predictor/compute_heat.py", "predictor/run_prediction.py"):
        subprocess.run(
            [sys.executable, script, "--db", str(db_path), "--events", str(events_path)],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=90, check=True,
        )


def evaluate_one(db_path, cluster, min_probability, threshold_multiplier, window_lookback=5):
    hot_records = ever_hot_records(str(db_path))
    heatshard_result = compute_plan(str(db_path), min_probability=min_probability, cluster=cluster, window_lookback=window_lookback)
    if heatshard_result is None:
        return None

    heatshard_window = heatshard_result["window_start"]
    heatshard_moves = [(m.record_id, m.source_shard, m.destination_shard) for m in heatshard_result["moves"]]

    reactive_window, reactive_shard_load, reactive_record_load = find_reactive_trigger(
        str(db_path), cluster, threshold_multiplier, window_lookback=window_lookback
    )
    reactive_result = (
        reactive_plan(reactive_shard_load, reactive_record_load, cluster, threshold_multiplier)
        if reactive_window is not None
        else {"moves": []}
    )

    systems = {
        "static": ([], heatshard_result["loads"], heatshard_result["record_load"]),
        "reactive": (
            reactive_result["moves"],
            reactive_shard_load or heatshard_result["loads"],
            reactive_record_load or heatshard_result["record_load"],
        ),
        "heatshard": (heatshard_moves, heatshard_result["loads"], heatshard_result["record_load"]),
    }

    results = {name: evaluate_system(name, moves, sl, rl, hot_records) for name, (moves, sl, rl) in systems.items()}
    lead_time = (reactive_window - heatshard_window) if reactive_window is not None else None
    return results, lead_time


def plot_summary(summary, out_path, n):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    colors = {"static": "#888888", "reactive": "#f87171", "heatshard": "#22d3ee"}

    for ax, metric in zip(axes.flat, METRICS):
        names = list(summary.keys())
        means = [summary[name][metric]["mean"] for name in names]
        stds = [summary[name][metric]["std"] for name in names]
        ax.bar(names, means, yerr=stds, capsize=6, color=[colors[n] for n in names])
        ax.set_title(METRIC_TITLES[metric])

    fig.suptitle(f"HeatShard vs. baselines, aggregated over {n} independent scenarios (mean ± std)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description="Aggregate the Stage 7 baseline comparison across many independent scenarios")
    parser.add_argument("--num-runs", type=int, default=15)
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--baseline-seconds", type=float, default=9.0)
    parser.add_argument("--lead-seconds", type=float, default=6.0)
    parser.add_argument("--spike-seconds", type=float, default=12.0)
    parser.add_argument("--cooldown-seconds", type=float, default=9.0)
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--spike-num-records", type=int, default=3)
    parser.add_argument("--min-probability", type=float, default=MIN_CANDIDATE_PROBABILITY)
    parser.add_argument("--threshold-multiplier", type=float, default=DEFAULT_THRESHOLD_MULTIPLIER)
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    cluster = ShardCluster()

    cfg_kwargs = dict(
        rate=args.rate,
        window_seconds=args.window_seconds,
        baseline_seconds=args.baseline_seconds,
        lead_seconds=args.lead_seconds,
        spike_seconds=args.spike_seconds,
        cooldown_seconds=args.cooldown_seconds,
        spike_num_records=args.spike_num_records,
    )

    per_system_metrics = {name: {m: [] for m in METRICS} for name in SYSTEMS}
    lead_times = []
    completed = 0

    for i in range(args.num_runs):
        seed = args.seed_start + i
        db_path = EVAL_DIR / f"eval_{seed}.db"
        events_path = EVAL_DIR / f"eval_{seed}_events.json"

        print(f"\n=== eval run {i + 1}/{args.num_runs} (seed={seed}) ===", flush=True)
        try:
            cfg = ScenarioConfig(seed=seed, **cfg_kwargs)
            run_scenario(cfg, db_path=str(db_path), events_path=str(events_path))
            compute_pipeline(db_path, events_path)
            outcome = evaluate_one(db_path, cluster, args.min_probability, args.threshold_multiplier)
        except Exception as exc:
            print(f"  run failed, skipping: {exc}")
            continue

        if outcome is None:
            print("  no actionable predictions this run -- skipped")
            continue

        results, lead_time = outcome
        completed += 1
        for name in SYSTEMS:
            for metric in METRICS:
                per_system_metrics[name][metric].append(results[name][metric])
        if lead_time is not None:
            lead_times.append(lead_time)

        print(
            f"  heatshard: moved={results['heatshard']['data_movement']} "
            f"precision={results['heatshard']['precision']:.2f} recall={results['heatshard']['recall']:.2f} "
            f"var {results['heatshard']['variance_before']:.0f}->{results['heatshard']['variance_after']:.0f}"
        )

    print(f"\n{completed}/{args.num_runs} runs produced an actionable HeatShard plan")
    if completed == 0:
        print("no usable runs -- nothing to aggregate")
        return

    print(f"\n{'system':<12} {'metric':<20} {'mean':>10} {'std':>10} {'min':>8} {'max':>8}  n")
    summary = {}
    for name in SYSTEMS:
        summary[name] = {}
        for metric in METRICS:
            values = np.array(per_system_metrics[name][metric])
            summary[name][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "n": len(values),
            }
            print(f"{name:<12} {metric:<20} {values.mean():>10.2f} {values.std():>10.2f} {values.min():>8.2f} {values.max():>8.2f}  {len(values)}")

    if lead_times:
        lt = np.array(lead_times)
        print(
            f"\nHeatShard lead time over reactive: mean={lt.mean():.1f}s std={lt.std():.1f}s "
            f"min={lt.min():.1f}s max={lt.max():.1f}s (n={len(lt)}/{completed} runs where reactive ever triggered)"
        )
    else:
        print("\nreactive never triggered in any completed run -- no lead-time distribution to report")

    plot_summary(summary, args.out, completed)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
