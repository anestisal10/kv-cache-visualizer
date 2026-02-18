"""
Base class for KV-Cache eviction policies.

Every eviction policy must subclass `EvictionPolicy` and implement
`select_tokens_to_keep()`. The cache manager calls this method whenever
the KV-Cache exceeds its budget.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EvictionPolicy(ABC):
    """Abstract base class for all KV-Cache eviction policies."""

    @abstractmethod
    def select_tokens_to_keep(
        self,
        cache_size: int,
        max_cache_size: int,
        attention_scores: list | None = None,
        token_positions: list[int] | None = None,
    ) -> list[int]:
        """
        Decide which token positions to KEEP in the cache.

        Args:
            cache_size: Current number of tokens in the cache.
            max_cache_size: Maximum allowed tokens in the cache.
            attention_scores: Cumulative attention scores per token (for H2O).
                              Shape: (cache_size,) — sum across all layers/heads.
            token_positions: Original position indices of tokens currently in cache.

        Returns:
            Sorted list of *indices into the current cache* (0-based) to keep.
            Tokens at indices NOT in this list will be evicted.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this policy (e.g., 'streaming_llm')."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for the UI."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
