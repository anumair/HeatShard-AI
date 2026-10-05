"""Pick the Relocation Planner's candidate-probability floor on a
VALIDATION set of scenarios (never the final evaluation set).

The floor only decides which records are even considered; ExpectedValue
does the real filtering. A lower floor lets the planner act earlier (on
less certain, calibrated probabilities) but with less observed load to
balance; a higher floor acts later with more evidence. This script shows
that trade-off per floor so the default can be chosen on data, not
by hand, and reports it for transparency.
"""

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.shard_client import ShardCluster  # noqa: E402
from planner.evaluate_aggregate import compute_pipeline, evaluate_one  # noqa: E402
from planner.reactive_baseline import DEFAULT_THRESHOLD_MULTIPLIER  # noqa: E402

DEFAULT_FLOORS = [0.3, 0.2, 0.15, 0.1, 0.05]


def main():
    parser = argparse.ArgumentParser(description="Sweep the planner's candidate-probability floor on validation scenarios")
    parser.add_argument("--db", action="append", required=True, help="validation scenario db (repeatable; globs ok); events expected at <db>_events.json")
    parser.add_argument("--floors", type=float, nargs="+", default=DEFAULT_FLOORS)
    args = parser.parse_args()

    dbs = []
    for pattern in args.db:
        dbs.extend(sorted(glob.glob(pattern)) or [pattern])
    cluster = ShardCluster()

    for db in dbs:
        compute_pipeline(db, db[:-3] + "_events.json")

    print(f"{len(dbs)} validation scenarios\n")
    print(f"{'floor':>6} {'moved':>6} {'precision':>10} {'recall':>8} {'FP':>5} {'peak var before':>16} {'peak var after':>15} {'reduction':>10} {'acts before 1st hot':>20}")
    for floor in args.floors:
        moved, precision, recall, fp, pb, pa, lead = [], [], [], [], [], [], []
        for db in dbs:
            outcome = evaluate_one(db, cluster, floor, DEFAULT_THRESHOLD_MULTIPLIER)
            if outcome is None:
                continue
            results, _, lead_vs_hot = outcome
            h = results["heatshard"]
            moved.append(h["data_movement"]); precision.append(h["precision"]); recall.append(h["recall"])
            fp.append(h["false_positive_rate"]); pb.append(h["peak_variance_before"]); pa.append(h["peak_variance_after"])
            if lead_vs_hot["heatshard"] is not None:
                lead.append(lead_vs_hot["heatshard"] > 0)
        print(
            f"{floor:>6.2f} {np.mean(moved):>6.1f} {np.mean(precision):>10.2f} {np.mean(recall):>8.2f} {np.mean(fp):>5.2f} "
            f"{np.mean(pb):>16.0f} {np.mean(pa):>15.0f} {100 * (1 - np.mean(pa) / np.mean(pb)):>9.0f}% {sum(lead):>14}/{len(lead)}"
        )


if __name__ == "__main__":
    main()
