"""Generate several flash-sale scenarios as Stage 4 training data.

Each run gets its own metrics db + event feed under data/training_runs/,
so one run's registered events don't overwrite another's -- training
needs to replay each scenario against the event context it actually ran
with. A single scenario alone rarely has enough positive (spike) windows
to train on; this gives the XGBoost sub-model a decent sample across
several independent spike scenarios.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.flash_sale_scenario import ScenarioConfig, run  # noqa: E402

TRAINING_DIR = Path(__file__).resolve().parent.parent / "data" / "training_runs"


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
    parser.add_argument("--key-source", choices=["synthetic", "olist"], default="olist")
    args = parser.parse_args()

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    db_paths, events_paths = [], []

    for i in range(args.num_runs):
        seed = args.seed_start + i
        cfg = ScenarioConfig(
            seed=seed,
            rate=args.rate,
            window_seconds=args.window_seconds,
            baseline_seconds=args.baseline_seconds,
            lead_seconds=args.lead_seconds,
            spike_seconds=args.spike_seconds,
            cooldown_seconds=args.cooldown_seconds,
            spike_num_records=args.spike_num_records,
            key_source=args.key_source,
        )
        db_path = TRAINING_DIR / f"run_{seed}.db"
        events_path = TRAINING_DIR / f"run_{seed}_events.json"
        print(f"\n=== training run seed={seed} ===")
        run(cfg, db_path=str(db_path), events_path=str(events_path))
        db_paths.append(str(db_path))
        events_paths.append(str(events_path))

    print("\ngenerated training runs:")
    for db_path, events_path in zip(db_paths, events_paths):
        print(f"  --db {db_path} --events {events_path}")


if __name__ == "__main__":
    main()
