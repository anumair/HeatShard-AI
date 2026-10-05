"""Statistical trend sub-model: the cheap, always-on half of the Hybrid
Prediction Engine.

Forecasts a record's next-window access count with Holt's damped linear
exponential smoothing over its recent per-window counts, then asks the
same question the label asks: will that forecast clear the "hot" bar
(predictor/labels.py -- HOT_MULTIPLIER x the record's lagged baseline, and
above the absolute noise floor)? The log-ratio of forecast to that bar is
squashed through a sigmoid onto a [0, 1] P(hotspot) scale.

It is stateless -- everything it needs is the record's window-aligned
count history -- so it can be applied to any record at any window, and its
raw forecast is also handed to the XGBoost sub-model as an input feature.
"""

import math

from predictor.labels import HOT_BASELINE_FLOOR, HOT_MIN_QPS, HOT_MULTIPLIER, lagged_baseline

ALPHA = 0.6  # level smoothing
BETA = 0.4  # trend smoothing
PHI = 0.8  # trend damping: surges are short, don't extrapolate them forever
SHARPNESS = 3.0  # sigmoid steepness on the log(forecast / hot bar) scale


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def forecast_next(counts) -> float:
    """Damped-trend Holt forecast of the next window's count."""
    counts = list(counts)
    if not counts:
        return 0.0
    level = float(counts[0])
    trend = float(counts[1] - counts[0]) if len(counts) > 1 else 0.0
    for c in counts[1:]:
        previous_level = level
        level = ALPHA * c + (1 - ALPHA) * (previous_level + PHI * trend)
        trend = BETA * (level - previous_level) + (1 - BETA) * PHI * trend
    return max(level + PHI * trend, 0.0)


def hot_bar(counts, window_seconds: float) -> float:
    """Access count the *next* window must reach to be labelled hot."""
    return max(HOT_MULTIPLIER * max(lagged_baseline(counts), HOT_BASELINE_FLOOR), HOT_MIN_QPS * window_seconds)


class TrendModel:
    def score(self, counts, window_seconds: float) -> dict:
        """counts: the record's window-aligned history, newest last (current window included)."""
        forecast = forecast_next(counts)
        bar = hot_bar(counts, window_seconds)
        return {
            "trend_forecast": forecast,
            "trend_ratio": forecast / bar,
            "p_trend": _sigmoid(SHARPNESS * math.log((forecast + 1.0) / (bar + 1.0))),
        }
