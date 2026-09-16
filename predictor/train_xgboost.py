"""Train the XGBoost sub-model of the Hybrid Prediction Engine on one or
more recorded scenarios (see simulator/generate_training_runs.py).

Reuses HeatIndex's own "did this record spike in the window that
followed" labeling -- the same signal its weight refitting already uses
-- via the sample_callback hook, so training labels stay consistent with
the rest of Component 2/3 instead of a separately-defined notion of
"spike".
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import xgboost as xgb  # noqa: E402

from collector.events import EventStore  # noqa: E402
from predictor.features import FEATURE_NAMES  # noqa: E402
from predictor.heat_index import HeatIndex  # noqa: E402
from predictor.window_reader import read_windows  # noqa: E402
from predictor.xgb_model import XGBHotspotModel  # noqa: E402

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "xgb_model.json"


def collect_samples(db_paths, events_paths):
    X, y = [], []

    def sink(record_id, features, label):
        X.append([features[name] for name in FEATURE_NAMES])
        y.append(label)

    for db_path, events_path in zip(db_paths, events_paths):
        events = EventStore(path=events_path) if events_path else EventStore()
        heat_index = HeatIndex(events=events, sample_callback=sink)
        for window_start, window_end, counters in read_windows(db_path):
            heat_index.process_window(counters, window_start, window_end)

    return np.array(X), np.array(y)


def main():
    parser = argparse.ArgumentParser(description="Train the XGBoost hotspot classifier")
    parser.add_argument("--db", action="append", required=True, help="metrics db path (repeatable)")
    parser.add_argument("--events", action="append", default=None, help="matching event feed path per --db (repeatable, same order)")
    parser.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    events_paths = args.events or [None] * len(args.db)
    if len(events_paths) != len(args.db):
        parser.error("--events must be given once per --db (same order), or omitted entirely")

    X, y = collect_samples(args.db, events_paths)
    n_pos = int(y.sum()) if len(y) else 0
    print(f"training samples: {len(y)} ({n_pos} positive, {len(y) - n_pos} negative)")
    if len(y) == 0 or n_pos == 0:
        print("not enough positive examples to train -- generate more scenarios (simulator/generate_training_runs.py)")
        return

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        eval_metric="logloss",
        scale_pos_weight=(len(y) - n_pos) / n_pos,
    )
    model.fit(X, y)

    print(f"training accuracy: {model.score(X, y):.3f}")

    XGBHotspotModel(booster=model).save(args.out)
    print(f"saved model to {args.out}")


if __name__ == "__main__":
    main()
