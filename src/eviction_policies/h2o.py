"""
H2O (Heavy-Hitter Oracle) eviction policy (Zhang et al., NeurIPS 2023).

Keep tokens that have accumulated the highest attention scores ("heavy hitters")
plus a window of the most recent tokens.
"""

from __future__ import annotations

from .base import EvictionPolicy


class H2OPolicy(EvictionPolicy):
    """
    Heavy-Hitter Oracle — dynamic eviction based on cumulative attention.

    Attention in transformers is highly sparse: ~5% of tokens accumulate
    >90% of the attention mass. H2O exploits this by tracking cumulative
    attention scores and keeping only the "heavy hitters" + recent tokens.
    """

    def __init__(self, heavy_hitter_budget: int = 128, recent_window: int = 128):
        """
        Args:
            heavy_hitter_budget: Number of highest-attention tokens to preserve.
            recent_window: Number of most-recent tokens to always keep.
        """
        self.heavy_hitter_budget = heavy_hitter_budget
        self.recent_window = recent_window

    def select_tokens_to_keep(
        self,
        cache_size: int,
        max_cache_size: int,
        attention_scores=None,
        token_positions=None,
    ) -> list[int]:
        budget = min(max_cache_size, self.heavy_hitter_budget + self.recent_window)

        if cache_size <= budget:
            return list(range(cache_size))

        # Recent window — always keep
        window_start = max(cache_size - self.recent_window, 0)
        recent_indices = set(range(window_start, cache_size))

        # Heavy hitters from the non-recent portion
        if attention_scores is not None and len(attention_scores) == cache_size:
            # Candidate indices: everything NOT in the recent window
            candidates = [i for i in range(cache_size) if i not in recent_indices]

            if candidates:
                # Sort candidates by attention score (descending), pick top-k
                scored = [(i, attention_scores[i]) for i in candidates]
                scored.sort(key=lambda x: x[1], reverse=True)
                hh_count = min(self.heavy_hitter_budget, len(scored))
                heavy_hitter_indices = {idx for idx, _ in scored[:hh_count]}
            else:
                heavy_hitter_indices = set()
        else:
            # Fallback: if no scores available, keep earliest tokens as proxy
            fallback_count = min(self.heavy_hitter_budget, window_start)
            heavy_hitter_indices = set(range(fallback_count))

        keep = sorted(recent_indices | heavy_hitter_indices)
        return keep

    @property
    def name(self) -> str:
        return "h2o"

    @property
    def description(self) -> str:
        return (
            f"H2O — keep top {self.heavy_hitter_budget} heavy-hitter tokens "
            f"+ last {self.recent_window} recent tokens"
        )
