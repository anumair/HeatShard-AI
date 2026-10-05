"""XGBoost sub-model: predicts P(record is hot next window) from the full
per-record feature vector (predictor/features.MODEL_FEATURE_NAMES).

Trained offline (train_xgboost.py). Saved as two files:
  xgb_model.json       - the booster
  xgb_model.meta.json  - feature order, isotonic calibration maps for both
                         sub-models, the learned ensemble blend weight and
                         the validated operating threshold/precision
"""

import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from predictor.calibration import apply_calibration
from predictor.features import MODEL_FEATURE_NAMES


def meta_path_for(model_path) -> Path:
    model_path = Path(model_path)
    return model_path.with_name(model_path.stem + ".meta.json")


class XGBHotspotModel:
    def __init__(self, booster: xgb.XGBClassifier = None, meta: dict = None):
        self.booster = booster
        self.meta = meta or {}

    @classmethod
    def load(cls, path) -> "XGBHotspotModel":
        booster = xgb.XGBClassifier()
        booster.load_model(str(path))
        meta_path = meta_path_for(path)
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} is missing -- this model predates prediction engine v2; retrain with predictor/train_xgboost.py"
            )
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("feature_names") != MODEL_FEATURE_NAMES:
            raise ValueError("model was trained on a different feature set -- retrain with predictor/train_xgboost.py")
        return cls(booster=booster, meta=meta)

    def save(self, path):
        self.booster.save_model(str(path))
        with open(meta_path_for(path), "w") as f:
            json.dump(self.meta, f, indent=2)

    def predict_proba_matrix(self, X) -> np.ndarray:
        """Calibrated P(hot) for a (n, len(MODEL_FEATURE_NAMES)) matrix."""
        raw = self.booster.predict_proba(X)[:, 1]
        return apply_calibration(self.meta.get("xgb_calibration"), raw)

    def predict_proba(self, features: dict) -> dict:
        """features: {record_id: {feature_name: value}}. Returns calibrated {record_id: p_xgb}."""
        if not features or self.booster is None:
            return {}
        record_ids = list(features.keys())
        X = np.array([[features[rid][name] for name in MODEL_FEATURE_NAMES] for rid in record_ids])
        probs = self.predict_proba_matrix(X)
        return {rid: float(p) for rid, p in zip(record_ids, probs)}
