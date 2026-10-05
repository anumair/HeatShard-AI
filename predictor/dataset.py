"""Turns recorded scenarios into labelled (features, hot-next-window) samples."""

from collections import namedtuple

import numpy as np

from collector.events import EventStore
from predictor.features import MODEL_FEATURE_NAMES
from predictor.pipeline import FeaturePipeline
from predictor.window_reader import read_windows

Samples = namedtuple("Samples", ["X", "y", "groups", "record_ids", "run_ids"])


def collect_samples(db_paths, events_paths) -> Samples:
    """One labelled row per (record active in window t) -> did it turn out
    hot in window t+1. `groups` indexes the source run, so cross-validation
    can hold out whole scenarios (adjacent windows of one run are strongly
    correlated; a random row split would leak)."""
    X, y, groups, record_ids = [], [], [], []
    for run_index, (db_path, events_path) in enumerate(zip(db_paths, events_paths)):
        pipeline = FeaturePipeline(events=EventStore(path=events_path) if events_path else EventStore())
        for window_start, window_end, counters in read_windows(db_path):
            for record_id, features, label in pipeline.step(counters, window_start, window_end).resolved:
                X.append([features[name] for name in MODEL_FEATURE_NAMES])
                y.append(label)
                groups.append(run_index)
                record_ids.append(record_id)
    return Samples(np.array(X, dtype=float), np.array(y, dtype=int), np.array(groups), record_ids, list(db_paths))
