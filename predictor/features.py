"""Shared feature schema for the heat index and its weight fitter.

Split out into its own module so heat_index.py and weight_fitter.py can
both depend on it without importing each other.
"""

FEATURE_NAMES = [
    "qps",
    "growth_rate",
    "avg_latency",
    "rw_ratio",
    "cache_miss_rate",
    "popularity",
    "event_signal",
]

# Initial fixed weights (w1..w7 in the methodology). event_signal starts
# weighted almost as heavily as qps so a record can visibly heat up before
# a scheduled flash sale actually shifts its raw traffic.
DEFAULT_WEIGHTS = {
    "qps": 0.25,
    "growth_rate": 0.15,
    "avg_latency": 0.10,
    "rw_ratio": 0.05,
    "cache_miss_rate": 0.10,
    "popularity": 0.10,
    "event_signal": 0.25,
}

# Everything the XGBoost sub-model sees. The first 7 are the heat index's
# z-scored signals; the rest are per-record context the z-scoring discards
# (recent history, the label's own baseline, trend forecast, event timing).
# Rates are per-second so a model trained on one window length transfers.
ENGINEERED_FEATURE_NAMES = [
    "log_qps",
    "qps_ratio_recent",   # current rate vs mean of the last 5 windows
    "qps_delta",
    "qps_lag1",
    "qps_lag2",
    "ratio_to_label_base",  # current count vs the lagged baseline the label itself will use
    "share",              # fraction of this window's total traffic
    "heat",
    "trend_forecast_qps",
    "trend_ratio",
    "p_trend",
    "event_announced",
    "event_tte",
    "event_magnitude",
]
MODEL_FEATURE_NAMES = FEATURE_NAMES + ENGINEERED_FEATURE_NAMES
