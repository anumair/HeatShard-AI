"""Adaptive confidence threshold: the P(hotspot) level required to "flag"
a record as a predicted hotspot. Tracks whether recently flagged
predictions were later validated (the record actually spiked) and nudges
the threshold up when flags have been low-value (raising the bar so the
system stops crying wolf), down when they've been reliably correct
(lets it act sooner) -- driven purely by the system's own recent track
record, not a fixed setting.

The starting threshold and the precision band to hold come from the
trained model's validation run (see train_xgboost.py): the threshold that
maximised held-out F1, and the precision it achieved. The adaptive rule
then only has to correct drift away from that validated operating point,
instead of searching for one from scratch.
"""

from collections import deque

INITIAL_THRESHOLD = 0.6
MIN_THRESHOLD = 0.15
MAX_THRESHOLD = 0.9
STEP = 0.05
OUTCOME_WINDOW = 10  # how many recent flagged predictions to judge precision over
TARGET_PRECISION = 0.5  # below this, raise the bar; comfortably above it, lower the bar
LOWER_MARGIN = 0.2  # precision must exceed target + margin before the bar is lowered


class AdaptiveThreshold:
    def __init__(
        self,
        initial: float = INITIAL_THRESHOLD,
        min_threshold: float = MIN_THRESHOLD,
        max_threshold: float = MAX_THRESHOLD,
        step: float = STEP,
        outcome_window: int = OUTCOME_WINDOW,
        target_precision: float = TARGET_PRECISION,
        lower_margin: float = LOWER_MARGIN,
    ):
        self.threshold = initial
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.step = step
        self.target_precision = target_precision
        self.lower_margin = lower_margin
        self._outcomes = deque(maxlen=outcome_window)  # 1 = flagged prediction was later validated, 0 = wasn't
        self._seen = 0
        self.history = []  # (n_outcomes_seen, threshold) logged every time the threshold moves

    def record_outcome(self, was_correct: bool):
        self._outcomes.append(1 if was_correct else 0)
        self._seen += 1
        if len(self._outcomes) < self._outcomes.maxlen:
            return  # not enough recent flags yet to judge precision

        precision = sum(self._outcomes) / len(self._outcomes)
        moved = False
        if precision < self.target_precision:
            new_threshold = min(self.max_threshold, self.threshold + self.step)
            moved = new_threshold != self.threshold
            self.threshold = new_threshold
        elif precision > self.target_precision + self.lower_margin:
            new_threshold = max(self.min_threshold, self.threshold - self.step)
            moved = new_threshold != self.threshold
            self.threshold = new_threshold

        if moved:
            # Judge the new threshold on fresh outcomes only. Without this the
            # rolling window slides one outcome at a time and the bar can move
            # on every single new outcome (0.38 -> 0.33 -> 0.28 -> 0.33 ...).
            self._outcomes.clear()
            self.history.append((self._seen, self.threshold))
