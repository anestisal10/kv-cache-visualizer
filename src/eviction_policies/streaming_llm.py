"""
StreamingLLM eviction policy (Xiao et al., ICLR 2024).

Keep the first `n_sink` tokens (attention sinks) plus the most recent
`window_size` tokens. Evict everything in between.
"""

from __future__ import annotations

from .base import EvictionPolicy


class StreamingLLMPolicy(EvictionPolicy):
    """
    Attention-sink + sliding-window policy.

    The key insight from the StreamingLLM paper is that the first few tokens
    act as "attention sinks" — they absorb excess attention mass that the
    softmax needs to dump somewhere. Removing them collapses generation quality.
    """

    def __init__(self, n_sink: int = 4, window_size: int = 252):
        """
        Args:
            n_sink: Number of initial tokens to always keep (attention sinks).
            window_size: Number of most-recent tokens to keep.
        """
        self.n_sink = n_sink
        self.window_size = window_size

    def select_tokens_to_keep(
        self,
        cache_size: int,
        max_cache_size: int,
        attention_scores=None,
        token_positions=None,
    ) -> list[int]:
        budget = min(max_cache_size, self.n_sink + self.window_size)

        if cache_size <= budget:
            return list(range(cache_size))

        # Keep first n_sink indices + last window_size indices
        sink_indices = list(range(min(self.n_sink, cache_size)))
        window_start = max(cache_size - self.window_size, self.n_sink)
        window_indices = list(range(window_start, cache_size))

        keep = sorted(set(sink_indices + window_indices))
        return keep

    @property
    def name(self) -> str:
        return "streaming_llm"

    @property
    def description(self) -> str:
        return (
            f"StreamingLLM — keep first {self.n_sink} sink token(s) "
            f"+ last {self.window_size} recent tokens"
        )
