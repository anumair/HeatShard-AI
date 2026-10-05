"""The single definition of "this record is hot" shared by every stage.

Training labels, the weight fitter's outcome signal, the adaptive
threshold's feedback, relocation-outcome checking and the Stage 7
ground truth all call is_hot(), so they can never drift apart.

A record is hot in a window when its load is both
  - material:  at least HOT_MIN_QPS requests/second (an absolute noise
               floor -- a record going from 0 to 3 hits is not a hotspot), and
  - surging:   at least HOT_MULTIPLIER x its own LAGGED baseline.

The baseline averages the HOT_BASELINE_SPAN windows that ended
HOT_LAG_WINDOWS ago, so a surge cannot absorb itself into its own
baseline after a window or two. That is what makes this a "sustained
surge" label rather than the earlier one-window "rising edge" label,
which fired only on the first window of a spike and mostly on
low-traffic Zipf-tail noise (see README, "Prediction engine v2").
"""

HOT_LAG_WINDOWS = 3
HOT_BASELINE_SPAN = 6
HOT_MULTIPLIER = 3.0
HOT_BASELINE_FLOOR = 2.0  # counts; keeps a near-zero baseline from making everything "3x"
HOT_MIN_QPS = 2.5
HISTORY_LEN = HOT_LAG_WINDOWS + HOT_BASELINE_SPAN  # windows of per-record history the rule needs
HOT_MIN_HISTORY = HOT_LAG_WINDOWS + 2  # not enough history => never labelled hot


def lagged_baseline(history) -> float:
    """Mean access count of the span ending HOT_LAG_WINDOWS ago.
    `history`: window-aligned counts (zeros included), oldest first,
    ending with the window just before the one being judged."""
    recent = list(history)[-HISTORY_LEN:]
    if len(recent) <= HOT_LAG_WINDOWS:
        return 0.0
    base = recent[:-HOT_LAG_WINDOWS]
    return sum(base) / len(base)


def is_hot(access_count: float, history, window_seconds: float) -> bool:
    """Is a window with `access_count` hits hot, given the record's preceding history?"""
    if len(history) < HOT_MIN_HISTORY:
        return False
    if access_count < HOT_MIN_QPS * window_seconds:
        return False
    return access_count >= HOT_MULTIPLIER * max(lagged_baseline(history), HOT_BASELINE_FLOOR)


def hot_windows(counts, window_seconds: float) -> list:
    """Indices into `counts` (a record's window-aligned series, zeros
    included) of every hot window."""
    return [i for i in range(len(counts)) if is_hot(counts[i], counts[max(0, i - HISTORY_LEN):i], window_seconds)]
