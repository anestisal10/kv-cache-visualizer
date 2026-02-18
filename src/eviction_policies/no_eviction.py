"""
No-eviction policy — full cache, upper-bound reference.

Keeps everything in the KV-Cache. Will OOM for long sequences,
so only suitable for short comparisons to establish quality upper bound.
"""

from __future__ import annotations

from .base import EvictionPolicy


class NoEvictionPolicy(EvictionPolicy):
    """Keep all tokens — quality upper bound (no eviction at all)."""

    def select_tokens_to_keep(
        self,
        cache_size: int,
        max_cache_size: int,
        attention_scores=None,
        token_positions=None,
    ) -> list[int]:
        # Always keep everything
        return list(range(cache_size))

    @property
    def name(self) -> str:
        return "no_eviction"

    @property
    def description(self) -> str:
        return "No Eviction — full cache (quality upper bound)"
