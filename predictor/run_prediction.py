"""Stage 4 CLI: run the Hybrid Prediction Engine over a recorded scenario.

Replays the scenario's windows chronologically through HeatIndex (for
heat scores + normalized features) and PredictionEngine (trend + XGBoost
ensemble), persists per-window predictions, feeds the adaptive
confidence threshold real outcomes as they resolve one window later, and
-- if a scenario manifest is available -- prints each spike record's
P(hotspot) trajectory against the baseline/pre_spike/spike/cooldown
phase boundaries. That trajectory is the Stage 4 checkpoint test:
probability should rise during pre_spike/early spike, not just after the
spike has already landed.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collector.events import EventStore  # noqa: E402
from collector.storage import DEFAULT_DB_PATH  # noqa: E402
from predictor import storage as heat_storage  # noqa: E402
from predictor.confidence import AdaptiveThreshold  # noqa: E402
from predictor.heat_index import HeatIndex  # noqa: E402
from predictor.prediction_engine import PredictionEngine  # noqa: E402
from predictor.window_reader import read_windows  # noqa: E402
from predictor.xgb_model import XGBHotspotModel  # noqa: E402

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "xgb_model.json"
DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parent.parent / "data" / "last_scenario.json"


def phase_for(window_start: float, window_end: float, phases: list) -> str:
    mid = (window_start + window_end) / 2
    for phase in phases:
        if phase["start"] <= mid <= phase["end"]:
            return phase["name"]
    return "?"


def main():
    parser = argparse.ArgumentParser(description="Run the Hybrid Prediction Engine over a recorded scenario")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--events", default=None)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="trained XGBoost model (falls back to trend-only if missing)")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--initial-threshold", type=float, default=None)
    args = parser.parse_args()

    heat_storage.init_db(args.db)
    heat_storage.clear_prediction_tables(args.db)

    events = EventStore(path=args.events)
    heat_index = HeatIndex(events=events)

    xgb_model = None
    model_path = Path(args.model)
    if model_path.exists():
        xgb_model = XGBHotspotModel.load(model_path)
        print(f"loaded XGBoost model from {model_path}")
    else:
        print(f"no XGBoost model at {model_path} -- running trend-only (train one with predictor/train_xgboost.py)")

    threshold = AdaptiveThreshold(initial=args.initial_threshold) if args.initial_threshold is not None else AdaptiveThreshold()
    engine = PredictionEngine(xgb_model=xgb_model, threshold=threshold)

    pending_flags = {}  # record_id -> True, carried one window to check against the actual outcome
    window_index = 0

    for window_start, window_end, counters in read_windows(args.db):
        window_index += 1

        for record_id in pending_flags:
            actual = counters[record_id].access_count if record_id in counters else 0
            threshold.record_outcome(heat_index.is_spike(record_id, actual))
        pending_flags = {}

        heat_scores = heat_index.process_window(counters, window_start, window_end)
        predictions = engine.predict_window(heat_scores, heat_index.last_normalized_features)
        heat_storage.write_predictions(args.db, window_start, window_end, predictions)
        heat_storage.write_threshold_history(args.db, window_index, threshold.threshold)

        for record_id, p in predictions.items():
            if p["flagged"]:
                pending_flags[record_id] = True

    print(f"\nprocessed {window_index} windows, final threshold: {threshold.threshold:.2f}")
    if threshold.history:
        print("threshold adjustments:")
        for n, t in threshold.history:
            print(f"  after {n} flagged outcomes -> {t:.2f}")

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"\nno manifest at {manifest_path} -- skipping spike-record trajectory report")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)
    spike_records = manifest.get("spike_records", [])
    phases = manifest.get("phases", [])
    if not spike_records:
        return

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    print("\nP(hotspot) trajectory for spike records:")
    for record_id in spike_records:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE record_id = ? ORDER BY window_start ASC", (record_id,)
        ).fetchall()
        print(f"\n  {record_id}")
        for r in rows:
            phase = phase_for(r["window_start"], r["window_end"], phases)
            flag = "  <- FLAGGED" if r["flagged"] else ""
            print(f"    p_ensemble={r['p_ensemble']:.2f}  confidence={r['confidence']:.2f}  [{phase:<9}]{flag}")
    conn.close()


if __name__ == "__main__":
    main()
