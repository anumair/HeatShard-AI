"""Held-out evaluation of the Hybrid Prediction Engine.

Scores every (record, window) pair of scenarios the model never trained
on against the shared "hot next window" label (predictor/labels.py) and
reports, for each sub-model and the blend:

  - average precision (threshold-free ranking quality)
  - precision / recall / F1 at the model's validated threshold
    (the honest operating-point number: the threshold was fixed at
    training time, not tuned on this data)
  - best-possible F1 on this data (oracle threshold; an upper bound,
    shown only to separate "bad ranking" from "bad threshold")

plus early-warning recall (hot records that were still quiet in the
current window -- the cases a purely reactive system cannot catch) and a
breakdown by record kind using each scenario's manifest: announced flash
sales, unannounced surprise spikes, decoy events that fizzled, and
ordinary records.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from predictor.calibration import apply_calibration  # noqa: E402
from predictor.dataset import collect_samples  # noqa: E402
from predictor.eval_utils import best_threshold, summarize  # noqa: E402
from predictor.features import MODEL_FEATURE_NAMES  # noqa: E402
from predictor.labels import HOT_MIN_QPS  # noqa: E402
from predictor.xgb_model import XGBHotspotModel  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COL = {name: i for i, name in enumerate(MODEL_FEATURE_NAMES)}


def load_manifests():
    by_db = {}
    for path in glob.glob(str(PROJECT_ROOT / "data" / "scenarios" / "*.json")):
        with open(path) as f:
            m = json.load(f)
        if m.get("db_path"):
            by_db[str(Path(m["db_path"]).resolve())] = m
    return by_db


def record_kinds(samples, manifests):
    kinds = []
    for record_id, db_path in zip(samples.record_ids, [samples.run_ids[g] for g in samples.groups]):
        m = manifests.get(str(Path(db_path).resolve()))
        if m is None:
            kinds.append("unknown")
        elif record_id in m.get("event_spike_records", m.get("spike_records", [])):
            kinds.append("announced_flash_sale")
        elif record_id in m.get("surprise_records", []):
            kinds.append("surprise_spike")
        elif record_id in m.get("decoy_records", []):
            kinds.append("decoy_event")
        else:
            kinds.append("ordinary")
    return np.array(kinds)


def main():
    parser = argparse.ArgumentParser(description="Held-out evaluation of the prediction engine")
    parser.add_argument("--db", action="append", required=True, help="held-out scenario db (repeatable; globs ok)")
    parser.add_argument("--model", default=str(PROJECT_ROOT / "data" / "xgb_model.json"))
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    dbs = []
    for pattern in args.db:
        dbs.extend(sorted(glob.glob(pattern)) or [pattern])
    events = [p[:-3] + "_events.json" for p in dbs]

    model = XGBHotspotModel.load(args.model)
    meta = model.meta
    samples = collect_samples(dbs, events)
    X, y = samples.X, samples.y

    p_xgb = model.predict_proba_matrix(X)
    p_trend = apply_calibration(meta["trend_calibration"], X[:, COL["p_trend"]])
    w = meta["blend_weight"]
    p_blend = w * p_xgb + (1 - w) * p_trend
    threshold = meta["threshold"]

    print(f"{len(dbs)} held-out scenarios, {len(y)} (record, window) samples, {int(y.sum())} hot ({100 * y.mean():.2f}%)")
    print(f"model: blend weight on XGBoost={w:.1f}, validated threshold={threshold:.2f}\n")

    variants = {
        "heat score alone": X[:, COL["heat"]],
        "current rate alone (hot now => hot next)": X[:, COL["log_qps"]],
        "trend sub-model alone": p_trend,
        "XGBoost sub-model alone": p_xgb,
        "blend (the engine)": p_blend,
    }
    results = {}
    print(f"{'variant':<44} {'AP':>6} {'F1@val-thr':>11} {'P':>6} {'R':>6} {'best F1*':>9}")
    for name, scores in variants.items():
        ap = summarize(y, scores, threshold)["average_precision"]
        if name in ("blend (the engine)", "XGBoost sub-model alone", "trend sub-model alone"):
            op = summarize(y, scores, threshold)
            op_txt = f"{op['f1']:>11.3f} {op['precision']:>6.2f} {op['recall']:>6.2f}"
        else:
            op = None
            op_txt = f"{'-':>11} {'-':>6} {'-':>6}"
        _, _, _, best_f1 = best_threshold(y, scores, grid=np.unique(np.quantile(scores, np.linspace(0.5, 0.9995, 150))))
        print(f"{name:<44} {ap:>6.3f} {op_txt} {best_f1:>9.3f}")
        results[name] = {"average_precision": ap, "operating_point": op, "best_f1_oracle_threshold": best_f1}
    print("* oracle threshold chosen on this same data -- an upper bound, not a result")

    # early warning: hot next window but not already hot this window
    qps = np.expm1(X[:, COL["log_qps"]])
    hot_now = (X[:, COL["ratio_to_label_base"]] >= 3.0) & (qps >= HOT_MIN_QPS)
    onset = (y == 1) & ~hot_now
    flagged = p_blend >= threshold
    kinds = record_kinds(samples, load_manifests())
    print(f"\nearly warning: {int(onset.sum())} hot-next-window records were still quiet this window (a reactive system flags 0% of these)")
    results["early_warning"] = {}
    for kind in ("announced_flash_sale", "surprise_spike", "ordinary"):
        m = onset & (kinds == kind)
        if m.any():
            print(f"  {kind:<22} {int(m.sum()):>3} onset cases, engine flags {100 * flagged[m].mean():.0f}%")
            results["early_warning"][kind] = {"cases": int(m.sum()), "recall": float(flagged[m].mean())}
    results["early_warning_recall_all"] = float(flagged[onset].mean()) if onset.any() else None

    print(f"\n{'record kind':<22} {'rows':>7} {'hot rows':>9} {'flagged':>8} {'recall':>7} {'precision':>10}")
    results["by_kind"] = {}
    for kind in ("announced_flash_sale", "surprise_spike", "decoy_event", "ordinary"):
        m = kinds == kind
        if not m.any():
            continue
        hot, fl = y[m] == 1, flagged[m]
        recall = fl[hot].mean() if hot.any() else float("nan")
        precision = hot[fl].mean() if fl.any() else float("nan")
        print(f"{kind:<22} {int(m.sum()):>7} {int(hot.sum()):>9} {int(fl.sum()):>8} {recall:>7.2f} {precision:>10.2f}")
        results["by_kind"][kind] = {"rows": int(m.sum()), "hot": int(hot.sum()), "flagged": int(fl.sum()), "recall": float(recall), "precision": float(precision)}

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
