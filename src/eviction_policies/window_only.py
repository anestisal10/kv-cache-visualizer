"""
Window-Only eviction policy — pure sliding window, NO attention sinks.

This is the key baseline for the research question: if Window-Only works
as well as StreamingLLM, then attention sinks are not necessary.
"""

from __future__ import annotations

from .base import EvictionPolicy


class WindowOnlyPolicy(EvictionPolicy):
    """
    Pure sliding window — keep only the N most recent tokens.
    No special treatment for the first token or any other position.
    """

    def __init__(self, window_size: int = 256):
        self.window_size = window_size

    def select_tokens_to_keep(
        self,
        cache_size: int,
        max_cache_size: int,
        attention_scores=None,
        token_positions=None,
    ) -> list[int]:
        budget = min(max_cache_size, self.window_size)
        if cache_size <= budget:
            return list(range(cache_size))

        # Keep only the last `budget` tokens
        return list(range(cache_size - budget, cache_size))

    @property
    def name(self) -> str:
        return "window_only"

    @property
    def description(self) -> str:
        return f"Window-Only — keep last {self.window_size} tokens (no sink)"
