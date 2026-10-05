"""One window-at-a-time feature pipeline used identically by training,
live prediction replay and evaluation, so the three can never disagree
about how a feature or a label is computed.

Records with an announced upcoming event are scored every window even
when they saw no traffic (an all-zero placeholder row), so a flash sale
on a cold record is predictable before its first request -- without
this, a record with no history has no feature row at all until after
the spike has already started.

step() takes one window of per-record counters and returns:
  heat_scores - this window's heat index per record
  features    - the full model feature vector per record (this window)
  resolved    - [(record_id, features_from_previous_window, label)]: now
                that this window's counts are known, did each record
                from the PREVIOUS window turn out hot? Training collects
                these as labelled samples; the adaptive threshold uses
                them as the outcome of earlier predictions.
"""

from collections import namedtuple

from predictor.heat_index import EVENT_HORIZON_SECONDS, HeatIndex
from predictor.model_features import ModelFeatureBuilder
from predictor.window_reader import WindowRow

StepResult = namedtuple("StepResult", ["heat_scores", "features", "resolved"])


class FeaturePipeline:
    def __init__(self, events=None, **heat_kwargs):
        self.heat_index = HeatIndex(events=events, **heat_kwargs)
        self.builder = ModelFeatureBuilder(self.heat_index)
        self._pending = {}

    def step(self, counters: dict, window_start: float, window_end: float) -> StepResult:
        window_seconds = max(window_end - window_start, 1e-6)

        idle_watched = self.heat_index.events.watched_records(window_end, EVENT_HORIZON_SECONDS) - counters.keys()
        if idle_watched:
            counters = {**counters, **{rid: WindowRow(0, 0, 0, 0.0) for rid in idle_watched}}

        # Labels first: must be judged against history that excludes this window.
        resolved = []
        for record_id, features in self._pending.items():
            actual = counters[record_id].access_count if record_id in counters else 0
            resolved.append((record_id, features, 1 if self.heat_index.is_hot(record_id, actual, window_seconds) else 0))

        heat_scores = self.heat_index.process_window(counters, window_start, window_end)
        features = self.builder.build(counters, heat_scores, window_end, window_seconds)
        self._pending = features
        return StepResult(heat_scores, features, resolved)
