"""Component 3: Hybrid Prediction Engine.

Combines the trend sub-model (damped-trend count forecast) and the
XGBoost sub-model into a single calibrated P(hot next window) per record,
plus a confidence score, and decides whether to "flag" a record by gating
the blended probability through an adaptive confidence threshold
(predictor/confidence.py).

The blend is learned, not a fixed 50/50: train_xgboost.py picks the
weight that maximises out-of-fold average precision and stores it, with
both sub-models' calibration maps, in the model's meta file. Without a
trained XGBoost model the engine degrades gracefully to the trend model
alone. Confidence is the agreement of the two calibrated sub-models.
"""

from predictor.calibration import apply_calibration
from predictor.confidence import AdaptiveThreshold


class PredictionEngine:
    def __init__(self, xgb_model=None, threshold: AdaptiveThreshold = None):
        self.xgb_model = xgb_model
        self.threshold = threshold or AdaptiveThreshold()
        meta = xgb_model.meta if xgb_model else {}
        self.blend_weight = meta.get("blend_weight", 0.5)  # weight on XGBoost
        self.trend_calibration = meta.get("trend_calibration")

    def predict_window(self, features: dict) -> dict:
        """features: this window's output from ModelFeatureBuilder.
        Returns {record_id: {p_trend, p_xgb, p_ensemble, confidence, flagged, threshold}}."""
        p_xgb = self.xgb_model.predict_proba(features) if self.xgb_model else {}

        results = {}
        for record_id, row in features.items():
            pt = float(apply_calibration(self.trend_calibration, row["p_trend"]))
            px = p_xgb.get(record_id)

            if px is None:
                p_ensemble, confidence = pt, 0.5
            else:
                p_ensemble = self.blend_weight * px + (1.0 - self.blend_weight) * pt
                confidence = 1.0 - abs(pt - px)

            results[record_id] = {
                "p_trend": pt,
                "p_xgb": px,
                "p_ensemble": p_ensemble,
                "confidence": confidence,
                "flagged": p_ensemble >= self.threshold.threshold,
                "threshold": self.threshold.threshold,
            }
        return results
