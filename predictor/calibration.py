"""Isotonic probability calibration, stored as plain arrays so a saved
model needs nothing but numpy to apply it.

Both sub-models emit scores on a [0, 1] scale that is only loosely a
probability (XGBoost is trained with a heavy positive-class weight; the
trend model's sigmoid is a heuristic squash). The Relocation Planner's
ExpectedValue multiplies P(hotspot) by a benefit, so P has to *mean*
"this fraction of records scored here really turn out hot" -- calibration
makes that true.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_calibration(scores, labels) -> dict:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(scores, dtype=float), np.asarray(labels, dtype=float))
    return {"x": [float(v) for v in iso.X_thresholds_], "y": [float(v) for v in iso.y_thresholds_]}


def apply_calibration(calibration, scores):
    """Works on a scalar, list or array; with calibration=None returns scores unchanged."""
    if not calibration:
        return scores
    return np.interp(scores, calibration["x"], calibration["y"])
