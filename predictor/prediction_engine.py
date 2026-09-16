"""Component 3: Hybrid Prediction Engine.

Combines the trend sub-model and the XGBoost sub-model into a single
P(hotspot in next window) + confidence score per record, and decides
whether to "flag" a record by gating the ensemble probability through an
adaptive confidence threshold (predictor/confidence.py). Runs trend-only
if no XGBoost model is supplied (e.g. before one has been trained) --
the ensemble degrades gracefully to whichever sub-models are active,
matching the methodology's "confidence = agreement across active
sub-models" framing.
"""

from predictor.confidence import AdaptiveThreshold
from predictor.trend_model import TrendModel


class PredictionEngine:
    def __init__(self, xgb_model=None, trend_window: int = None, threshold: AdaptiveThreshold = None):
        self.trend_model = TrendModel(window=trend_window) if trend_window else TrendModel()
        self.xgb_model = xgb_model
        self.threshold = threshold or AdaptiveThreshold()

    def predict_window(self, heat_scores: dict, normalized_features: dict) -> dict:
        """heat_scores/normalized_features: this window's output from HeatIndex.
        Returns {record_id: {p_trend, p_xgb, p_ensemble, confidence, flagged, threshold}}."""
        p_trend = self.trend_model.update_and_score(heat_scores)
        p_xgb = self.xgb_model.predict_proba(normalized_features) if self.xgb_model else {}

        results = {}
        for record_id in heat_scores:
            pt = p_trend.get(record_id)
            px = p_xgb.get(record_id)

            active = [p for p in (pt, px) if p is not None]
            if not active:
                continue
            p_ensemble = sum(active) / len(active)
            confidence = 1.0 - abs(pt - px) if (pt is not None and px is not None) else 0.5

            results[record_id] = {
                "p_trend": pt,
                "p_xgb": px,
                "p_ensemble": p_ensemble,
                "confidence": confidence,
                "flagged": p_ensemble >= self.threshold.threshold,
                "threshold": self.threshold.threshold,
            }
        return results
