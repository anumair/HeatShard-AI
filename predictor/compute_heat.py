"""Stage 3 CLI: replay a recorded scenario's windows through the Adaptive
Heat Index and persist per-window heat scores (+ weight-refit history).

Run this right after a scenario (checkpoint_test.py, flash_sale_scenario.py,
or a plain load_generator.py run) while data/events.json still reflects
that same run's registered events.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collector.events import EventStore  # noqa: E402
from collector.storage import DEFAULT_DB_PATH  # noqa: E402
from predictor import storage as heat_storage  # noqa: E402
from predictor.heat_index import HeatIndex  # noqa: E402
from predictor.window_reader import read_windows  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Compute the adaptive heat index over a recorded scenario")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--events", default=None, help="event feed path (default: data/events.json)")
    parser.add_argument("--refit-every", type=int, default=10, help="windows between weight refits (methodology default: 20-50)")
    parser.add_argument("--min-refit-samples", type=int, default=20)
    args = parser.parse_args()

    heat_storage.init_db(args.db)
    heat_storage.clear_heat_tables(args.db)

    events = EventStore(path=args.events)
    heat_index = HeatIndex(events=events, refit_every=args.refit_every, min_refit_samples=args.min_refit_samples)

    num_windows = 0
    for window_start, window_end, counters in read_windows(args.db):
        heat_scores = heat_index.process_window(counters, window_start, window_end)
        heat_storage.write_heat_window(args.db, window_start, window_end, heat_scores)
        num_windows += 1

    for window_index, weights in heat_index.weight_history:
        heat_storage.write_weight_history(args.db, window_index, weights)
        print(f"[refit @ window {window_index}] weights -> " + ", ".join(f"{k}={v:.2f}" for k, v in weights.items()))

    print(f"\nprocessed {num_windows} windows")
    print("final weights: " + ", ".join(f"{k}={v:.2f}" for k, v in heat_index.weights.items()))


if __name__ == "__main__":
    main()
