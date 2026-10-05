"""Train the XGBoost sub-model of the Hybrid Prediction Engine.

Target: P(record is hot in the next window), with "hot" defined once, in
predictor/labels.py, and shared with every other stage.

What training produces (xgb_model.json + xgb_model.meta.json):
  1. The booster, fit on all supplied scenarios.
  2. Isotonic calibration of both sub-models, fit on OUT-OF-FOLD scores
     (5-fold, folds are whole scenarios) so the probabilities the
     Relocation Planner multiplies by a benefit are honest.
  3. The ensemble blend weight (trend vs XGBoost) that maximises
     out-of-fold average precision.
  4. The operating threshold that maximises out-of-fold F1, and the
     precision it achieved -- the adaptive threshold's starting point
     and the precision band it then tries to hold.

Pass --test-db/--test-events (scenarios never trained on) for a
held-out report with a sub-model ablation; the full evaluation lives in
predictor/evaluate_prediction.py.
"""

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from predictor.calibration import apply_calibration, fit_calibration  # noqa: E402
from predictor.dataset import collect_samples  # noqa: E402
from predictor.eval_utils import best_threshold, summarize  # noqa: E402
from predictor.features import MODEL_FEATURE_NAMES  # noqa: E402
from predictor.xgb_model import XGBHotspotModel  # noqa: E402

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "xgb_model.json"
DEFAULT_MAX_SCALE_POS_WEIGHT = 10.0
TREND_COLUMN = MODEL_FEATURE_NAMES.index("p_trend")
XGB_PARAMS = dict(
    n_estimators=250,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    eval_metric="logloss",
    n_jobs=4,
)


def fit_model(X, y, scale_pos_weight=None):
    n_pos = int(y.sum())
    full_ratio = (len(y) - n_pos) / max(n_pos, 1)
    weight = scale_pos_weight if scale_pos_weight is not None else min(full_ratio, DEFAULT_MAX_SCALE_POS_WEIGHT)
    model = xgb.XGBClassifier(scale_pos_weight=weight, **XGB_PARAMS)
    model.fit(X, y)
    return model


def out_of_fold_scores(X, y, groups, scale_pos_weight, folds=5):
    scores = np.zeros(len(y))
    for train_idx, test_idx in GroupKFold(n_splits=min(folds, len(set(groups)))).split(X, y, groups):
        model = fit_model(X[train_idx], y[train_idx], scale_pos_weight)
        scores[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return scores


def choose_blend(y, xgb_cal, trend_cal):
    """Blend weight on XGBoost in [0, 1] maximising average precision."""
    from sklearn.metrics import average_precision_score

    best_w, best_ap = 1.0, -1.0
    for w in np.linspace(0.0, 1.0, 11):
        ap = average_precision_score(y, w * xgb_cal + (1 - w) * trend_cal)
        if ap > best_ap:
            best_w, best_ap = float(w), float(ap)
    return best_w, best_ap


def fit_meta(X, y, groups, scale_pos_weight):
    """Everything except the final booster: calibration, blend, threshold."""
    oof = out_of_fold_scores(X, y, groups, scale_pos_weight)
    xgb_calibration = fit_calibration(oof, y)
    trend_calibration = fit_calibration(X[:, TREND_COLUMN], y)
    xgb_cal = apply_calibration(xgb_calibration, oof)
    trend_cal = apply_calibration(trend_calibration, X[:, TREND_COLUMN])
    blend_weight, blend_ap = choose_blend(y, xgb_cal, trend_cal)
    blended = blend_weight * xgb_cal + (1 - blend_weight) * trend_cal
    threshold, precision, recall, f1 = best_threshold(y, blended)
    meta = {
        "feature_names": MODEL_FEATURE_NAMES,
        "xgb_calibration": xgb_calibration,
        "trend_calibration": trend_calibration,
        "blend_weight": blend_weight,
        "threshold": threshold,
        "precision_at_threshold": precision,
        "cv": {
            "folds": "5-fold, grouped by scenario",
            "samples": int(len(y)),
            "positives": int(y.sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "average_precision": blend_ap,
        },
    }
    return meta, {"xgb": xgb_cal, "trend": trend_cal, "blend": blended}


def parse_pairs(db_list, events_list, label):
    events_list = events_list or [None] * len(db_list or [])
    if db_list and len(events_list) != len(db_list):
        raise SystemExit(f"--{label}events must be given once per --{label}db (same order), or omitted entirely")
    return db_list or [], events_list


def expand(patterns):
    out = []
    for pattern in patterns or []:
        out.extend(sorted(glob.glob(pattern)) or [pattern])
    return out


def main():
    parser = argparse.ArgumentParser(description="Train the XGBoost hotspot classifier + calibration/blend/threshold meta")
    parser.add_argument("--db", action="append", required=True, help="training metrics db (repeatable; globs ok). Events are expected at <db minus .db>_events.json unless --events is given")
    parser.add_argument("--events", action="append", default=None, help="event feed per --db, same order")
    parser.add_argument("--test-db", action="append", default=None, help="held-out scenario db (repeatable; globs ok)")
    parser.add_argument("--test-events", action="append", default=None)
    parser.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--scale-pos-weight", type=float, default=None, help=f"default: min(imbalance ratio, {DEFAULT_MAX_SCALE_POS_WEIGHT:.0f}); calibration corrects the resulting probability inflation")
    args = parser.parse_args()

    train_dbs = expand(args.db)
    train_events = args.events or [p[:-3] + "_events.json" for p in train_dbs]
    test_dbs = expand(args.test_db)
    test_events = args.test_events or [p[:-3] + "_events.json" for p in test_dbs]

    samples = collect_samples(train_dbs, train_events)
    X, y, groups = samples.X, samples.y, samples.groups
    if len(y) == 0 or y.sum() == 0:
        print("not enough positive examples to train -- generate more scenarios (simulator/generate_training_runs.py --vary-params)")
        return
    print(f"training set: {len(train_dbs)} scenarios, {len(y)} samples, {int(y.sum())} hot ({100 * y.mean():.2f}%)")

    meta, oof = fit_meta(X, y, groups, args.scale_pos_weight)
    cv = meta["cv"]
    print(
        f"\nout-of-fold (scenario-grouped 5-fold) at validated threshold {meta['threshold']:.2f}: "
        f"precision={cv['precision']:.3f} recall={cv['recall']:.3f} F1={cv['f1']:.3f} AP={cv['average_precision']:.3f}"
    )
    print(f"blend weight on XGBoost: {meta['blend_weight']:.1f} (trend gets {1 - meta['blend_weight']:.1f})")
    for name in ("trend", "xgb", "blend"):
        t, p, r, f = best_threshold(y, oof[name])
        print(f"  ablation  {name:<6} out-of-fold best-F1={f:.3f} (P={p:.2f} R={r:.2f} @ {t:.2f})")

    booster = fit_model(X, y, args.scale_pos_weight)
    model = XGBHotspotModel(booster=booster, meta=meta)

    if test_dbs:
        test = collect_samples(test_dbs, test_events)
        probs = model.predict_proba_matrix(test.X)
        blended = meta["blend_weight"] * probs + (1 - meta["blend_weight"]) * apply_calibration(meta["trend_calibration"], test.X[:, TREND_COLUMN])
        res = summarize(test.y, blended, meta["threshold"])
        print(
            f"\nheld-out test ({len(test_dbs)} scenarios, {len(test.y)} samples, {int(test.y.sum())} hot) at the validated threshold: "
            f"precision={res['precision']:.3f} recall={res['recall']:.3f} F1={res['f1']:.3f} AP={res['average_precision']:.3f}"
        )

    model.save(args.out)
    print(f"\nsaved model + meta to {args.out}")


if __name__ == "__main__":
    main()
