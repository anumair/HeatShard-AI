"""Stage 2 deliverable: a repeatable, scriptable flash-sale scenario.

Runs four phases against the live cluster, all routed through the metrics
collector:

  baseline   - normal Zipfian traffic, nothing scheduled yet
  pre_spike  - event metadata for the flash sale is registered in the
               event channel right at the start of this phase, but actual
               traffic hasn't moved yet (this is what the predictor is
               meant to key off of before the spike is visible in raw
               counters)
  spike      - traffic is heavily biased toward a small set of "flash
               sale" records, at the same overall rate as baseline
  cooldown   - back to normal Zipfian traffic

A manifest recording exactly which records spiked and the wall-clock
boundaries of each phase is written to data/scenarios/ (ground-truth
labels for Stage 4's predictor training) and mirrored to
data/last_scenario.json for convenience.
"""

import argparse
import json
import random
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collector.events import write_events  # noqa: E402
from collector.middleware import MetricsCollector  # noqa: E402
from collector.window_worker import WindowWorker  # noqa: E402
from common.keyspace import KeySpace  # noqa: E402
from simulator.load_generator import touch  # noqa: E402
from simulator.zipf import ZipfSampler  # noqa: E402

SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "data" / "scenarios"
LAST_SCENARIO_PATH = Path(__file__).resolve().parent.parent / "data" / "last_scenario.json"


@dataclass
class ScenarioConfig:
    num_keys: int = 200
    skew: float = 1.2
    seed: int = 42
    rate: float = 40.0
    write_ratio: float = 0.3
    group_ratio: float = 0.15
    window_seconds: float = 5.0
    baseline_seconds: float = 30.0
    lead_seconds: float = 20.0
    spike_seconds: float = 20.0
    cooldown_seconds: float = 20.0
    spike_num_records: int = 4
    spike_magnitude: float = 8.0
    spike_bias: float = 0.85  # P(traffic goes to a spike record) during the spike phase
    key_source: str = "olist"  # or "synthetic" for the original product:i / reviews:i / inventory:i scheme


def run_phase(collector, key_space, pick_rank, label, duration, rate, write_ratio, group_ratio, rng):
    print(f"-- {label} ({duration:.0f}s, ~{rate:.0f} ops/sec) --")
    interval = 1.0 / rate if rate > 0 else 0
    end = time.monotonic() + duration
    ops = 0
    while time.monotonic() < end:
        if rng.random() < group_ratio:
            keys = key_space.related_keys(pick_rank())
            for key in keys:
                touch(collector, key_space, key, write_ratio, ops)
                ops += 1
            collector.transaction(keys)
        else:
            touch(collector, key_space, key_space.record_id(pick_rank()), write_ratio, ops)
            ops += 1
        if interval:
            time.sleep(interval)
    return ops


def print_spike_record_history(db_path, spike_records):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    print("\nper-window access_count for spike records (baseline -> pre_spike -> spike -> cooldown):")
    for record_id in spike_records:
        rows = conn.execute(
            "SELECT access_count FROM metric_windows WHERE record_id = ? ORDER BY window_start ASC",
            (record_id,),
        ).fetchall()
        series = " ".join(f"{r['access_count']:>3}" for r in rows)
        print(f"  {record_id:<14} {series}")
    conn.close()


def run(cfg: ScenarioConfig, db_path=None, events_path=None):
    rng = random.Random(cfg.seed)
    key_space = (
        KeySpace.synthetic(cfg.num_keys)
        if cfg.key_source == "synthetic"
        else KeySpace.from_olist(num_keys=cfg.num_keys)
    )
    num_keys = key_space.num_keys

    collector = MetricsCollector(db_path=db_path)
    worker = WindowWorker(collector, interval_seconds=cfg.window_seconds)
    worker.start()

    baseline_sampler = ZipfSampler(num_keys, skew=cfg.skew, seed=cfg.seed)
    spike_ranks = rng.sample(range(num_keys), cfg.spike_num_records)
    spike_records = [key_space.record_id(i) for i in spike_ranks]

    def normal_pick():
        return baseline_sampler.sample()

    def spike_pick():
        if rng.random() < cfg.spike_bias:
            return rng.choice(spike_ranks)
        return baseline_sampler.sample()

    phases = []

    t0 = time.time()
    ops_baseline = run_phase(collector, key_space, normal_pick, "baseline", cfg.baseline_seconds, cfg.rate, cfg.write_ratio, cfg.group_ratio, rng)
    t1 = time.time()
    phases.append({"name": "baseline", "start": t0, "end": t1, "ops": ops_baseline})

    now = time.time()
    events = [
        {
            "record_id": rid,
            "event_type": "flash_sale",
            "scheduled_time": now + cfg.lead_seconds,
            "expected_magnitude": cfg.spike_magnitude,
        }
        for rid in spike_records
    ]
    write_events(events, path=events_path)
    event_registered_at = now
    print(f"\nregistered event metadata for {spike_records} -> spike scheduled at +{cfg.lead_seconds:.0f}s\n")

    ops_pre = run_phase(
        collector, key_space, normal_pick, "pre_spike (event known, traffic still normal)",
        cfg.lead_seconds, cfg.rate, cfg.write_ratio, cfg.group_ratio, rng,
    )
    t2 = time.time()
    phases.append({"name": "pre_spike", "start": t1, "end": t2, "ops": ops_pre})

    ops_spike = run_phase(
        collector, key_space, spike_pick, f"SPIKE on {spike_records}",
        cfg.spike_seconds, cfg.rate, cfg.write_ratio, cfg.group_ratio, rng,
    )
    t3 = time.time()
    phases.append({"name": "spike", "start": t2, "end": t3, "ops": ops_spike})

    ops_cool = run_phase(
        collector, key_space, normal_pick, "cooldown",
        cfg.cooldown_seconds, cfg.rate, cfg.write_ratio, cfg.group_ratio, rng,
    )
    t4 = time.time()
    phases.append({"name": "cooldown", "start": t3, "end": t4, "ops": ops_cool})

    worker.stop()
    collector.snapshot_window()
    time.sleep(0.2)

    manifest = {
        "config": asdict(cfg),
        "spike_records": spike_records,
        "event_registered_at": event_registered_at,
        "phases": phases,
        "db_path": str(collector.db_path),
    }

    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = SCENARIOS_DIR / f"scenario_{int(t0)}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    with open(LAST_SCENARIO_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    total_ops = ops_baseline + ops_pre + ops_spike + ops_cool
    print(f"total ops: {total_ops}")
    print(f"spike records: {spike_records}")
    print(f"manifest: {manifest_path}")
    print(f"metrics db: {collector.db_path}")

    print_spike_record_history(collector.db_path, spike_records)

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repeatable flash-sale traffic scenario")
    parser.add_argument("--num-keys", type=int, default=ScenarioConfig.num_keys)
    parser.add_argument("--skew", type=float, default=ScenarioConfig.skew)
    parser.add_argument("--seed", type=int, default=ScenarioConfig.seed)
    parser.add_argument("--rate", type=float, default=ScenarioConfig.rate)
    parser.add_argument("--write-ratio", type=float, default=ScenarioConfig.write_ratio)
    parser.add_argument("--group-ratio", type=float, default=ScenarioConfig.group_ratio)
    parser.add_argument("--window-seconds", type=float, default=ScenarioConfig.window_seconds)
    parser.add_argument("--baseline-seconds", type=float, default=ScenarioConfig.baseline_seconds)
    parser.add_argument("--lead-seconds", type=float, default=ScenarioConfig.lead_seconds)
    parser.add_argument("--spike-seconds", type=float, default=ScenarioConfig.spike_seconds)
    parser.add_argument("--cooldown-seconds", type=float, default=ScenarioConfig.cooldown_seconds)
    parser.add_argument("--spike-num-records", type=int, default=ScenarioConfig.spike_num_records)
    parser.add_argument("--spike-magnitude", type=float, default=ScenarioConfig.spike_magnitude)
    parser.add_argument("--spike-bias", type=float, default=ScenarioConfig.spike_bias)
    parser.add_argument("--db", default=None, help="metrics db path (default: data/metrics.db)")
    parser.add_argument("--events-out", default=None, help="event feed path (default: data/events.json)")
    parser.add_argument("--key-source", choices=["synthetic", "olist"], default=ScenarioConfig.key_source)
    args = parser.parse_args()

    cfg = ScenarioConfig(
        num_keys=args.num_keys,
        skew=args.skew,
        seed=args.seed,
        rate=args.rate,
        write_ratio=args.write_ratio,
        group_ratio=args.group_ratio,
        window_seconds=args.window_seconds,
        baseline_seconds=args.baseline_seconds,
        lead_seconds=args.lead_seconds,
        spike_seconds=args.spike_seconds,
        key_source=args.key_source,
        cooldown_seconds=args.cooldown_seconds,
        spike_num_records=args.spike_num_records,
        spike_magnitude=args.spike_magnitude,
        spike_bias=args.spike_bias,
    )
    run(cfg, db_path=args.db, events_path=args.events_out)
