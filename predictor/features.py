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
