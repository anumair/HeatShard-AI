"""Component 2: Adaptive Heat Index.

Turns each window's raw per-record counters into a single hotness score:

    Heat_i(t) = w1*QPS + w2*GrowthRate + w3*AvgLatency + w4*RWRatio
              + w5*CacheMissRate + w6*Popularity + w7*EventSignal

All seven contributing metrics are exponentially decayed between windows
(so a record's heat cools gradually instead of resetting to zero the
moment it stops seeing traffic) and z-score normalized across the
population of records active that window, before being weighted. Weights
start fixed (features.DEFAULT_WEIGHTS) and are periodically refit by
WeightFitter using observed outcomes as the training signal.
"""

from collections import defaultdict, deque

import numpy as np

from collector.events import EventStore
from predictor.features import DEFAULT_WEIGHTS, FEATURE_NAMES
from predictor.labels import HISTORY_LEN, is_hot
from predictor.weight_fitter import WeightFitter

DECAY_LAMBDA = 0.6  # metric(t) = lambda*raw(t) + (1-lambda)*metric(t-1)
EVENT_HORIZON_SECONDS = 120.0  # how far ahead a scheduled event starts contributing signal
NO_EVENT_TTE = 150.0  # "seconds until event" sentinel when no announced event is near


class RecordState:
    __slots__ = ("decayed", "access_history")

    def __init__(self):
        self.decayed = {name: 0.0 for name in FEATURE_NAMES}
        # Window-aligned access counts (a 0 for every window the record was
        # idle), newest last -- what predictor/labels.py judges "hot" against.
        self.access_history = deque(maxlen=HISTORY_LEN)


class HeatIndex:
    def __init__(
        self,
        weights=None,
        events: EventStore = None,
        decay_lambda: float = DECAY_LAMBDA,
        refit_every: int = 25,
        min_refit_samples: int = 30,
        sample_callback=None,
    ):
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.events = events or EventStore()
        self.decay_lambda = decay_lambda
        self._state = defaultdict(RecordState)
        self._fitter = WeightFitter(refit_every=refit_every, min_samples=min_refit_samples)
        self._pending_features = {}  # record_id -> normalized features from the previous window
        self.weight_history = []  # (window_index, weights) appended on every refit
        self._window_index = 0
        # Called as sample_callback(record_id, normalized_features, label) every time a
        # label is resolved -- lets Stage 4's XGBoost training reuse this exact same
        # "did it spike in the window that followed" signal instead of recomputing it.
        self._sample_callback = sample_callback
        # Normalized (z-scored) features from the most recently processed window --
        # Stage 4's live PredictionEngine reads this right after process_window().
        self.last_normalized_features = {}

    def _nearest_event(self, record_id: str, now: float):
        """(signal, seconds_until_start, magnitude) of the strongest
        already-announced event near `now`; (0.0, NO_EVENT_TTE, 0.0) if none."""
        best = (0.0, NO_EVENT_TTE, 0.0)
        for e in self.events.visible_for_record(record_id, now):
            dt = e["scheduled_time"] - now
            if -30.0 <= dt <= EVENT_HORIZON_SECONDS:
                magnitude = e.get("expected_magnitude", 1.0)
                signal = (1.0 - max(dt, 0.0) / EVENT_HORIZON_SECONDS) * magnitude
                if signal > best[0]:
                    best = (signal, dt, magnitude)
        return best

    def _event_signal(self, record_id: str, now: float) -> float:
        return self._nearest_event(record_id, now)[0]

    def event_features(self, record_id: str, now: float) -> dict:
        """Raw (un-normalized) event context for the prediction models:
        whether an announced event is near, how many seconds until it
        starts (negative once started), and its announced magnitude."""
        _, tte, magnitude = self._nearest_event(record_id, now)
        return {"event_announced": 1.0 if magnitude > 0 else 0.0, "event_tte": tte, "event_magnitude": magnitude}

    def _raw_features(self, record_id: str, counters, window_seconds: float, window_end: float) -> dict:
        access = counters.access_count
        history = self._state[record_id].access_history
        prev_access = history[-1] if history else 0
        return {
            "qps": access / window_seconds,
            "growth_rate": (access - prev_access) / (prev_access + 1.0),
            "avg_latency": counters.avg_latency_ms,
            "rw_ratio": (counters.write_count / access) if access else 0.0,
            "cache_miss_rate": (counters.cache_miss_count / access) if access else 0.0,
            "event_signal": self._event_signal(record_id, window_end),
        }

    @staticmethod
    def _normalize(decayed_by_record: dict) -> dict:
        if not decayed_by_record:
            return {}
        record_ids = list(decayed_by_record.keys())
        matrix = np.array([[decayed_by_record[rid][name] for name in FEATURE_NAMES] for rid in record_ids])
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std[std < 1e-9] = 1.0
        z = (matrix - mean) / std
        return {rid: dict(zip(FEATURE_NAMES, z[i])) for i, rid in enumerate(record_ids)}

    def is_hot(self, record_id: str, access_count: int, window_seconds: float) -> bool:
        """Would `access_count` in the window being processed count as hot?
        Judged against history up to (excluding) that window."""
        return is_hot(access_count, self._state[record_id].access_history, window_seconds)

    def count_history(self, record_id: str) -> list:
        return list(self._state[record_id].access_history)

    def process_window(self, counters: dict, window_start: float, window_end: float) -> dict:
        """counters: {record_id: obj with .access_count/.write_count/.cache_miss_count/.avg_latency_ms}.
        Returns {record_id: heat_score} for this window."""
        self._window_index += 1
        window_seconds = max(window_end - window_start, 1e-6)
        total_access = sum(c.access_count for c in counters.values()) or 1

        # Resolve labels for features carried over from the previous window,
        # now that we know what this record actually did this window.
        for record_id, features in self._pending_features.items():
            actual = counters[record_id].access_count if record_id in counters else 0
            label = 1 if self.is_hot(record_id, actual, window_seconds) else 0
            self._fitter.add_sample(features, label)
            if self._sample_callback:
                self._sample_callback(record_id, features, label)

        raw = {}
        for record_id, c in counters.items():
            feats = self._raw_features(record_id, c, window_seconds, window_end)
            feats["popularity"] = c.access_count / total_access
            raw[record_id] = feats

        decayed = {}
        for record_id, feats in raw.items():
            state = self._state[record_id]
            state.decayed = {
                name: self.decay_lambda * feats[name] + (1 - self.decay_lambda) * state.decayed[name]
                for name in FEATURE_NAMES
            }
            decayed[record_id] = state.decayed

        normalized = self._normalize(decayed)
        heat_scores = {
            record_id: sum(self.weights[name] * feats[name] for name in FEATURE_NAMES)
            for record_id, feats in normalized.items()
        }

        for record_id, state in self._state.items():
            if record_id in counters:
                state.access_history.append(counters[record_id].access_count)
            else:  # idle this window: cool its decayed metrics and log a zero
                state.access_history.append(0)
                state.decayed = {name: (1 - self.decay_lambda) * v for name, v in state.decayed.items()}

        self._pending_features = normalized
        self.last_normalized_features = normalized

        new_weights, refit_happened = self._fitter.maybe_refit(self.weights)
        if refit_happened:
            self.weights = new_weights
            self.weight_history.append((self._window_index, dict(self.weights)))

        return heat_scores
