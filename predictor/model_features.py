"""Builds the full per-record feature vector the prediction models use.

Sits on top of a HeatIndex that has just processed a window: reuses its
z-scored signals, its window-aligned count history and its (leak-safe)
event context, and adds the engineered history / trend / event-timing
features listed in features.ENGINEERED_FEATURE_NAMES.
"""

import math

from predictor.features import FEATURE_NAMES
from predictor.labels import HOT_BASELINE_FLOOR, lagged_baseline
from predictor.trend_model import TrendModel


class ModelFeatureBuilder:
    def __init__(self, heat_index, trend_model: TrendModel = None):
        self.heat_index = heat_index
        self.trend_model = trend_model or TrendModel()

    def build(self, counters: dict, heat_scores: dict, window_end: float, window_seconds: float) -> dict:
        """Call right after heat_index.process_window() for the same window.
        Returns {record_id: {feature_name: value}} covering FEATURE_NAMES + ENGINEERED_FEATURE_NAMES."""
        total = sum(c.access_count for c in counters.values()) or 1
        normalized = self.heat_index.last_normalized_features
        features = {}
        for record_id, counter in counters.items():
            history = self.heat_index.count_history(record_id)  # includes the current window, newest last
            current = history[-1]
            prior = history[:-1]
            recent_mean = sum(prior[-5:]) / len(prior[-5:]) if prior else 0.0
            label_base = max(lagged_baseline(history), HOT_BASELINE_FLOOR)
            trend = self.trend_model.score(history, window_seconds)

            row = dict(normalized[record_id])
            row.update(
                log_qps=math.log1p(current / window_seconds),
                qps_ratio_recent=current / (recent_mean + 1.0),
                qps_delta=(current - recent_mean) / window_seconds,
                qps_lag1=(prior[-1] if prior else 0) / window_seconds,
                qps_lag2=(prior[-2] if len(prior) > 1 else 0) / window_seconds,
                ratio_to_label_base=current / label_base,
                share=current / total,
                heat=heat_scores[record_id],
                trend_forecast_qps=trend["trend_forecast"] / window_seconds,
                trend_ratio=trend["trend_ratio"],
                p_trend=trend["p_trend"],
            )
            row.update(self.heat_index.event_features(record_id, window_end))
            features[record_id] = row
        return features
