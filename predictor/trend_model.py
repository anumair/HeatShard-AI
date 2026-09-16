"""Statistical trend analysis: the cheap, always-on sub-model of the
Hybrid Prediction Engine. Fits a linear slope over each record's last K
heat scores, z-scores that slope across the window's active population
(same normalization pattern as the heat index itself), and squashes it
through a sigmoid so it lands on the same [0, 1] "P(hotspot)" scale as
the XGBoost sub-model.
"""

import math
from collections import defaultdict, deque

import numpy as np

TREND_WINDOW = 5  # how many past heat scores to fit the slope over


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class TrendModel:
    def __init__(self, window: int = TREND_WINDOW):
        self.window = window
        self._history = defaultdict(lambda: deque(maxlen=window))

    def update_and_score(self, heat_scores: dict) -> dict:
        """heat_scores: {record_id: heat score this window}. Returns {record_id: p_trend}."""
        for record_id, score in heat_scores.items():
            self._history[record_id].append(score)

        slopes = {}
        for record_id in heat_scores:
            hist = self._history[record_id]
            if len(hist) < 2:
                slopes[record_id] = 0.0
                continue
            xs = np.arange(len(hist))
            ys = np.array(hist)
            slopes[record_id] = float(np.polyfit(xs, ys, 1)[0])

        if not slopes:
            return {}

        values = np.array(list(slopes.values()))
        mean, std = values.mean(), values.std()
        std = std if std > 1e-9 else 1.0
        return {rid: _sigmoid((s - mean) / std) for rid, s in slopes.items()}
