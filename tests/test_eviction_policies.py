"""
Unit tests for eviction policies.

Tests verify that each policy returns the correct set of tokens to keep
under various cache sizes and scenarios.
"""

import pytest
from src.eviction_policies import (
    get_policy,
    list_policies,
    StreamingLLMPolicy,
    H2OPolicy,
    WindowOnlyPolicy,
    RandomEvictionPolicy,
    NoEvictionPolicy,
)


class TestPolicyRegistry:
    """Test the policy registry functions."""

    def test_list_policies(self):
        names = list_policies()
        assert "streaming_llm" in names
        assert "h2o" in names
        assert "window_only" in names
        assert "random" in names
        assert "no_eviction" in names

    def test_get_policy_streaming_llm(self):
        p = get_policy("streaming_llm", n_sink=2, window_size=10)
        assert isinstance(p, StreamingLLMPolicy)
        assert p.n_sink == 2
        assert p.window_size == 10

    def test_get_policy_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown policy"):
            get_policy("nonexistent_policy")


class TestStreamingLLMPolicy:
    """Test StreamingLLM eviction policy."""

    def test_no_eviction_when_under_budget(self):
        p = StreamingLLMPolicy(n_sink=4, window_size=10)
        keep = p.select_tokens_to_keep(cache_size=8, max_cache_size=14)
        assert keep == list(range(8))

    def test_eviction_keeps_sink_and_window(self):
        p = StreamingLLMPolicy(n_sink=2, window_size=4)
        keep = p.select_tokens_to_keep(cache_size=10, max_cache_size=6)
        # Should keep indices 0, 1 (sinks) and 6, 7, 8, 9 (window)
        assert 0 in keep
        assert 1 in keep
        assert 6 in keep
        assert 9 in keep
        assert len(keep) == 6

    def test_zero_sinks(self):
        p = StreamingLLMPolicy(n_sink=0, window_size=5)
        keep = p.select_tokens_to_keep(cache_size=10, max_cache_size=5)
        # Should only keep last 5
        assert keep == [5, 6, 7, 8, 9]

    def test_name(self):
        p = StreamingLLMPolicy()
        assert p.name == "streaming_llm"


class TestH2OPolicy:
    """Test H2O (Heavy-Hitter Oracle) eviction policy."""

    def test_no_eviction_when_under_budget(self):
        p = H2OPolicy(heavy_hitter_budget=3, recent_window=3)
        keep = p.select_tokens_to_keep(cache_size=5, max_cache_size=6)
        assert keep == list(range(5))

    def test_keeps_heavy_hitters_and_recent(self):
        p = H2OPolicy(heavy_hitter_budget=2, recent_window=3)
        # Attention scores: token 1 and 3 have highest scores
        scores = [0.1, 0.9, 0.2, 0.8, 0.3, 0.1, 0.05, 0.04, 0.02, 0.01]
        keep = p.select_tokens_to_keep(
            cache_size=10, max_cache_size=5,
            attention_scores=scores,
        )
        # Should keep: indices 1, 3 (heavy hitters) + 7, 8, 9 (recent)
        assert 1 in keep
        assert 3 in keep
        assert 7 in keep
        assert 8 in keep
        assert 9 in keep

    def test_fallback_without_scores(self):
        p = H2OPolicy(heavy_hitter_budget=3, recent_window=3)
        keep = p.select_tokens_to_keep(cache_size=10, max_cache_size=6)
        # Without scores, should keep earliest 3 + latest 3
        assert len(keep) == 6
        assert 0 in keep  # fallback keeps earliest
        assert 9 in keep  # recent window

    def test_name(self):
        p = H2OPolicy()
        assert p.name == "h2o"


class TestWindowOnlyPolicy:
    """Test Window-Only eviction policy."""

    def test_no_eviction_when_under_budget(self):
        p = WindowOnlyPolicy(window_size=10)
        keep = p.select_tokens_to_keep(cache_size=5, max_cache_size=10)
        assert keep == list(range(5))

    def test_eviction_keeps_only_recent(self):
        p = WindowOnlyPolicy(window_size=4)
        keep = p.select_tokens_to_keep(cache_size=10, max_cache_size=4)
        # Only last 4 tokens
        assert keep == [6, 7, 8, 9]
        # Token 0 (the sink) should NOT be kept
        assert 0 not in keep

    def test_name(self):
        p = WindowOnlyPolicy()
        assert p.name == "window_only"


class TestRandomEvictionPolicy:
    """Test Random eviction policy."""

    def test_no_eviction_when_under_budget(self):
        p = RandomEvictionPolicy(seed=42)
        keep = p.select_tokens_to_keep(cache_size=5, max_cache_size=10)
        assert keep == list(range(5))

    def test_keeps_correct_count(self):
        p = RandomEvictionPolicy(seed=42)
        keep = p.select_tokens_to_keep(cache_size=10, max_cache_size=5)
        assert len(keep) == 5
        assert all(0 <= i < 10 for i in keep)

    def test_deterministic_with_same_seed(self):
        p1 = RandomEvictionPolicy(seed=42)
        p2 = RandomEvictionPolicy(seed=42)
        k1 = p1.select_tokens_to_keep(cache_size=20, max_cache_size=10)
        k2 = p2.select_tokens_to_keep(cache_size=20, max_cache_size=10)
        assert k1 == k2

    def test_name(self):
        p = RandomEvictionPolicy()
        assert p.name == "random"


class TestNoEvictionPolicy:
    """Test No-Eviction policy."""

    def test_always_keeps_everything(self):
        p = NoEvictionPolicy()
        keep = p.select_tokens_to_keep(cache_size=100, max_cache_size=50)
        assert keep == list(range(100))

    def test_name(self):
        p = NoEvictionPolicy()
        assert p.name == "no_eviction"
