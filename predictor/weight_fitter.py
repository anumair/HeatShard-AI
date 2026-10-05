"""Periodic weight refitting for the heat index.

Every `refit_every` windows, fit weights via (ridge-regularised) least squares against
observed outcomes -- did the record actually spike in the window that
followed the one these features came from? -- so the index learns which
signals are currently most predictive instead of relying on the fixed
initial weights forever.
"""

import numpy as np

from predictor.features import FEATURE_NAMES

RIDGE = 5.0  # shrinks the fit toward zero coefficients so a few noisy samples can't dominate
SMOOTHING = 0.5  # new weights = SMOOTHING * fitted + (1 - SMOOTHING) * previous
MIN_WEIGHT = 0.02  # no signal is ever switched off completely


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

        # Ridge-regularised least squares, then clip / renormalise / smooth:
        # plain OLS on a rare, noisy label repeatedly collapsed the whole
        # index onto a single feature (e.g. popularity = 1.0) after one refit.
        n_features = X.shape[1]
        coeffs = np.linalg.solve(X.T @ X + RIDGE * np.eye(n_features), X.T @ y)
        coeffs = np.clip(coeffs, 0.0, None)
        total = coeffs.sum()
        if total < 1e-9:
            new_weights = current_weights
        else:
            fitted = coeffs / total
            blended = np.array([SMOOTHING * f + (1 - SMOOTHING) * current_weights[name] for f, name in zip(fitted, FEATURE_NAMES)])
            blended = np.maximum(blended, MIN_WEIGHT)
            blended = blended / blended.sum()
            new_weights = {name: float(w) for name, w in zip(FEATURE_NAMES, blended)}

        self._buffer.clear()
        self._windows_since_fit = 0
        return new_weights, True
