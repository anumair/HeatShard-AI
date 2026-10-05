"""Generate several flash-sale scenarios as Stage 4 training data.

Each run gets its own metrics db + event feed under data/training_runs/,
so one run's registered events don't overwrite another's -- training
needs to replay each scenario against the event context it actually ran
with. A single scenario alone rarely has enough positive (spike) windows
to train on; this gives the XGBoost sub-model a decent sample across
several independent spike scenarios.

--vary-params randomizes spike magnitude/bias/num-records/skew per run
(deterministically, seeded) instead of holding them fixed across every
run. It also draws the realism knobs (unannounced surprise spikes, decoy
events that fizzle, ramped spikes, window length, request rate) so the
event channel is not a perfect oracle for the model. Varying only the seed changes *which* records spike but not the
*shape* of the spike, which caps how much a classifier can generalize
from more runs of the same shape; --vary-params is the recommended way
to get more out of additional training data.
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.flash_sale_scenario import ScenarioConfig, run  # noqa: E402

TRAINING_DIR = Path(__file__).resolve().parent.parent / "data" / "training_runs"


def varied_config(seed: int, key_source: str = "olist") -> ScenarioConfig:
    """Deterministic, seed-derived scenario shape -- shared by training-data
    generation and the held-out evaluation sets so both come from the same
    distribution (but different seeds)."""
    rng = random.Random(seed)
    window = rng.choice([3.0, 3.0, 4.0, 5.0])
    return ScenarioConfig(
        seed=seed,
        rate=rng.uniform(25.0, 45.0),
        window_seconds=window,
        baseline_seconds=max(12.0, 4 * window),
        lead_seconds=max(9.0, 2 * window),
        spike_seconds=max(15.0, 4 * window),
        cooldown_seconds=max(12.0, 3 * window),
        spike_num_records=rng.randint(2, 5),
        spike_magnitude=rng.uniform(4.0, 12.0),
        spike_bias=rng.uniform(0.65, 0.95),
        skew=rng.uniform(0.9, 1.6),
        surprise_num_records=rng.choice([0, 1, 1, 2]),
        decoy_num_records=rng.choice([0, 1, 2, 3]),
        ramp_seconds=rng.choice([0.0, 0.0, 3.0, 6.0]),
        key_source=key_source,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate multiple flash-sale scenarios for XGBoost training")
    parser.add_argument("--num-runs", type=int, default=6)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--baseline-seconds", type=float, default=12.0)
    parser.add_argument("--lead-seconds", type=float, default=9.0)
    parser.add_argument("--spike-seconds", type=float, default=15.0)
    parser.add_argument("--cooldown-seconds", type=float, default=12.0)
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--spike-num-records", type=int, default=3)
    parser.add_argument("--skew", type=float, default=ScenarioConfig.skew)
    parser.add_argument("--key-source", choices=["synthetic", "olist"], default="olist")
    parser.add_argument("--vary-params", action="store_true", help="randomize spike shape, realism knobs, window length and rate per run (see varied_config)")
    parser.add_argument("--out-dir", default=str(TRAINING_DIR), help="where to write run_<seed>.db / run_<seed>_events.json")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_paths, events_paths = [], []

    for i in range(args.num_runs):
        seed = args.seed_start + i
        if args.vary_params:
            cfg = varied_config(seed, key_source=args.key_source)
        else:
            cfg = ScenarioConfig(
                seed=seed,
                rate=args.rate,
                window_seconds=args.window_seconds,
                baseline_seconds=args.baseline_seconds,
                lead_seconds=args.lead_seconds,
                spike_seconds=args.spike_seconds,
                cooldown_seconds=args.cooldown_seconds,
                spike_num_records=args.spike_num_records,
                skew=args.skew,
                key_source=args.key_source,
            )
        db_path = out_dir / f"run_{seed}.db"
        events_path = out_dir / f"run_{seed}_events.json"
        for stale in (db_path, events_path):
            stale.unlink(missing_ok=True)  # the collector APPENDS to an existing db -- never reuse one
        print(f"\n=== training run seed={seed} ===")
        run(cfg, db_path=str(db_path), events_path=str(events_path))
        db_paths.append(str(db_path))
        events_paths.append(str(events_path))

    print("\ngenerated training runs:")
    for db_path, events_path in zip(db_paths, events_paths):
        print(f"  --db {db_path} --events {events_path}")


if __name__ == "__main__":
    main()
