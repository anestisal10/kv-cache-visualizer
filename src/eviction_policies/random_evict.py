"""
Random eviction policy — worst-case baseline.

Randomly selects which tokens to evict when the cache is full.
Useful as a lower-bound reference: any real policy should beat this.
"""

from __future__ import annotations

import random

from .base import EvictionPolicy


class RandomEvictionPolicy(EvictionPolicy):
    """Randomly keep tokens — worst-case baseline for comparison."""

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def select_tokens_to_keep(
        self,
        cache_size: int,
        max_cache_size: int,
        attention_scores=None,
        token_positions=None,
    ) -> list[int]:
        if cache_size <= max_cache_size:
            return list(range(cache_size))

        # Randomly sample max_cache_size indices to keep
        indices = list(range(cache_size))
        keep = sorted(self._rng.sample(indices, max_cache_size))
        return keep

    @property
    def name(self) -> str:
        return "random"

    @property
    def description(self) -> str:
        return "Random Eviction — randomly drop tokens (worst-case baseline)"
