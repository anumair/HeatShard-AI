"""Periodic weight refitting for the heat index.

Every `refit_every` windows, fit weights via ordinary least squares against
observed outcomes -- did the record actually spike in the window that
followed the one these features came from? -- so the index learns which
signals are currently most predictive instead of relying on the fixed
initial weights forever.
"""

import numpy as np

from predictor.features import FEATURE_NAMES


class WeightFitter:
    def __init__(self, refit_every: int = 25, min_samples: int = 30):
        self.refit_every = refit_every
        self.min_samples = min_samples
        self._buffer = []  # list of (feature_vector, label)
        self._windows_since_fit = 0

    def add_sample(self, normalized_features: dict, label: int):
        self._buffer.append(([normalized_features[name] for name in FEATURE_NAMES], label))

    def maybe_refit(self, current_weights: dict):
        """Returns (weights, refit_happened)."""
        self._windows_since_fit += 1
        if self._windows_since_fit < self.refit_every or len(self._buffer) < self.min_samples:
            return current_weights, False

        X = np.array([v for v, _ in self._buffer])
        y = np.array([label for _, label in self._buffer], dtype=float)

        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        # z-scored features can pull the fit negative; clip and renormalize
        # so the weighted sum stays a well-behaved, comparable heat score.
        coeffs = np.clip(coeffs, 0.0, None)
        total = coeffs.sum()
        new_weights = (
            current_weights
            if total < 1e-9
            else {name: float(c / total) for name, c in zip(FEATURE_NAMES, coeffs)}
        )

        self._buffer.clear()
        self._windows_since_fit = 0
        return new_weights, True
