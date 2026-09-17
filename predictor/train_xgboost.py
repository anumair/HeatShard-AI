"""Train the XGBoost sub-model of the Hybrid Prediction Engine on one or
more recorded scenarios (see simulator/generate_training_runs.py).

Reuses HeatIndex's own "did this record spike in the window that
followed" labeling -- the same signal its weight refitting already uses
-- via the sample_callback hook, so training labels stay consistent with
the rest of Component 2/3 instead of a separately-defined notion of
"spike".

Positive examples are rare (~1% of windows), so plain accuracy is
meaningless -- a model that never predicts "hot" would still score
~99%. Pass --test-db/--test-events (entirely separate scenarios the
model never trains on) to get a real precision/recall/F1 report instead.
The split is scenario-level, not a random row split, because adjacent
windows within one scenario are correlated (decayed features carry over
window to window) -- a random split would leak information between
train and test.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

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


DEFAULT_MAX_SCALE_POS_WEIGHT = 20.0  # empirically the best precision/recall tradeoff on held-out data;
# the full imbalance ratio (~100x) pushes recall up but collapses precision (see README)


def fit_model(X, y, scale_pos_weight=None):
    n_pos = int(y.sum())
    full_ratio = (len(y) - n_pos) / n_pos
    weight = scale_pos_weight if scale_pos_weight is not None else min(full_ratio, DEFAULT_MAX_SCALE_POS_WEIGHT)
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        eval_metric="logloss",
        scale_pos_weight=weight,
    )
    model.fit(X, y)
    return model


def report(name, y_true, y_pred):
    n_pos = int(y_true.sum())
    print(f"\n{name}: {len(y_true)} samples ({n_pos} positive, {len(y_true) - n_pos} negative)")
    print(classification_report(y_true, y_pred, target_names=["not hot", "hot"], zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"confusion matrix [[TN FP] [FN TP]]:\n{cm}")


def parse_pairs(db_list, events_list, label):
    events_list = events_list or [None] * len(db_list or [])
    if db_list and len(events_list) != len(db_list):
        raise SystemExit(f"--{label}-events must be given once per --{label}-db (same order), or omitted entirely")
    return db_list or [], events_list


def main():
    parser = argparse.ArgumentParser(description="Train (and optionally evaluate) the XGBoost hotspot classifier")
    parser.add_argument("--db", action="append", required=True, help="training metrics db path (repeatable)")
    parser.add_argument("--events", action="append", default=None, help="matching event feed path per --db (repeatable, same order)")
    parser.add_argument("--test-db", action="append", default=None, help="held-out scenario db path, never trained on (repeatable)")
    parser.add_argument("--test-events", action="append", default=None, help="matching event feed path per --test-db (repeatable, same order)")
    parser.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--refit-on-all", action="store_true", help="after evaluating, refit the saved model on train+test combined")
    parser.add_argument("--scale-pos-weight", type=float, default=None, help=f"default: min(full imbalance ratio, {DEFAULT_MAX_SCALE_POS_WEIGHT}) -- the full ratio alone collapses precision")
    args = parser.parse_args()

    train_dbs, train_events = parse_pairs(args.db, args.events, "")
    test_dbs, test_events = parse_pairs(args.test_db, args.test_events, "test")

    X_train, y_train = collect_samples(train_dbs, train_events)
    if len(y_train) == 0 or y_train.sum() == 0:
        print("not enough positive examples to train -- generate more scenarios (simulator/generate_training_runs.py)")
        return

    model = fit_model(X_train, y_train, scale_pos_weight=args.scale_pos_weight)
    report("train (in-sample, not a generalization measure)", y_train, model.predict(X_train))

    if test_dbs:
        X_test, y_test = collect_samples(test_dbs, test_events)
        report("held-out test (scenarios never trained on)", y_test, model.predict(X_test))

        if args.refit_on_all:
            X_all = np.concatenate([X_train, X_test])
            y_all = np.concatenate([y_train, y_test])
            model = fit_model(X_all, y_all, scale_pos_weight=args.scale_pos_weight)
            print(f"\nrefit final model on train+test combined: {len(y_all)} samples ({int(y_all.sum())} positive)")
    else:
        print(
            "\nno --test-db given -- the numbers above are in-sample and will look better than real "
            "generalization. Pass --test-db/--test-events with scenarios not in --db for an honest read."
        )

    XGBHotspotModel(booster=model).save(args.out)
    print(f"\nsaved model to {args.out}")


if __name__ == "__main__":
    main()
