"""Zipfian key sampler: rank i (0-indexed) is sampled with weight 1/(i+1)^skew,
so a small number of records get most of the baseline traffic. Cumulative
weights are precomputed once so each sample() call is an O(log n) bisect
instead of an O(n) rescan.
"""

import itertools
import random


class ZipfSampler:
    def __init__(self, num_keys: int, skew: float = 1.2, seed: int = None):
        self.num_keys = num_keys
        self.skew = skew
        self._rng = random.Random(seed)
        weights = [1.0 / ((i + 1) ** skew) for i in range(num_keys)]
        self._ranks = list(range(num_keys))
        self._cum_weights = list(itertools.accumulate(weights))

    def sample(self) -> int:
        """Returns a record rank in [0, num_keys)."""
        return self._rng.choices(self._ranks, cum_weights=self._cum_weights, k=1)[0]
