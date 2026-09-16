"""XGBoost sub-model: predicts P(hotspot in next window) from a record's
normalized heat-index features for the current window. Trained offline
(see train_xgboost.py) on labeled samples produced by HeatIndex's own
"did this record spike in the window that followed" signal, so its
training labels are consistent with the rest of Component 2/3.
"""

import numpy as np
import xgboost as xgb

from predictor.features import FEATURE_NAMES


class XGBHotspotModel:
    def __init__(self, booster: xgb.XGBClassifier = None):
        self.booster = booster

    @classmethod
    def load(cls, path) -> "XGBHotspotModel":
        booster = xgb.XGBClassifier()
        booster.load_model(str(path))
        return cls(booster=booster)

    def save(self, path):
        self.booster.save_model(str(path))

    def predict_proba(self, normalized_features: dict) -> dict:
        """normalized_features: {record_id: {feature_name: z-score}}. Returns {record_id: p_xgb}."""
        if not normalized_features or self.booster is None:
            return {}
        record_ids = list(normalized_features.keys())
        X = np.array([[normalized_features[rid][name] for name in FEATURE_NAMES] for rid in record_ids])
        probs = self.booster.predict_proba(X)[:, 1]
        return {rid: float(p) for rid, p in zip(record_ids, probs)}
