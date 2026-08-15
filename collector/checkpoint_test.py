"""Stage 1 checkpoint test: manually spike traffic on one record and confirm
its counters visibly jump in the next window snapshot.

Runs baseline -> spike -> cooldown against a dedicated database, then prints
the spiked record's per-window history so the jump (and decay afterward) is
directly visible.
"""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collector.middleware import MetricsCollector  # noqa: E402
from collector.window_worker import WindowWorker  # noqa: E402

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "checkpoint_test.db"
NUM_KEYS = 50
SPIKE_KEY = "product:7"
WINDOW_SECONDS = 3
PHASE_SECONDS = 9  # 3 windows per phase
BASELINE_RATE = 15  # ops/sec
SPIKE_RATE = 80  # ops/sec


def run_phase(collector, label, duration, rate, spiking):
    print(f"-- {label} ({duration}s, ~{rate} ops/sec) --")
    interval = 1.0 / rate
    end = time.monotonic() + duration
    ops = 0
    while time.monotonic() < end:
        key = SPIKE_KEY if spiking else f"product:{random.randrange(NUM_KEYS)}"
        if random.random() < 0.3:
            collector.set(key, f"payload-{ops}")
        else:
            collector.get(key)
        ops += 1
        time.sleep(interval)
    return ops


def main():
    if DB_PATH.exists():
        DB_PATH.unlink()

    collector = MetricsCollector(db_path=DB_PATH)
    worker = WindowWorker(collector, interval_seconds=WINDOW_SECONDS)
    worker.start()

    run_phase(collector, "baseline", PHASE_SECONDS, BASELINE_RATE, spiking=False)
    run_phase(collector, f"SPIKE on {SPIKE_KEY}", PHASE_SECONDS, SPIKE_RATE, spiking=True)
    run_phase(collector, "cooldown", PHASE_SECONDS, BASELINE_RATE, spiking=False)

    worker.stop()
    collector.snapshot_window()
    time.sleep(0.2)

    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM metric_windows WHERE record_id = ? ORDER BY window_start ASC",
        (SPIKE_KEY,),
    ).fetchall()

    print(f"\n{SPIKE_KEY} access_count per window:")
    for i, r in enumerate(rows):
        bar = "#" * r["access_count"]
        print(f"  window {i:<2} access={r['access_count']:<4} write={r['write_count']:<4} {bar}")


if __name__ == "__main__":
    main()
