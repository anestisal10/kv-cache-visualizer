"""
Unit tests for the KV-Cache Manager.

Tests use mock KV-Cache tensors (no model required) to verify
eviction logic, state tracking, and snapshot history.
"""

import pytest
import torch

from src.cache_manager import KVCacheManager
from src.eviction_policies import StreamingLLMPolicy, WindowOnlyPolicy


def _make_mock_cache(batch=1, n_layers=2, n_heads=4, seq_len=10, head_dim=8):
    """Create a mock KV-Cache as a tuple of (key, value) per layer."""
    cache = []
    for _ in range(n_layers):
        k = torch.randn(batch, n_heads, seq_len, head_dim)
        v = torch.randn(batch, n_heads, seq_len, head_dim)
        cache.append((k, v))
    return tuple(cache)


def _make_mock_attention(n_layers=2, n_heads=4, q_len=1, kv_len=10):
    """Create mock attention weights."""
    attn = []
    for _ in range(n_layers):
        # Random attention that sums to 1 over kv_len
        a = torch.rand(n_heads, q_len, kv_len)
        a = a / a.sum(dim=-1, keepdim=True)
        attn.append(a)
    return attn


class TestKVCacheManager:
    """Test KVCacheManager core functionality."""

    def test_init(self):
        policy = StreamingLLMPolicy(n_sink=2, window_size=5)
        mgr = KVCacheManager(policy, max_cache_size=7)
        assert mgr.max_cache_size == 7
        assert mgr.policy is policy
        assert len(mgr.snapshots) == 0

    def test_update_creates_snapshot(self):
        policy = StreamingLLMPolicy(n_sink=2, window_size=5)
        mgr = KVCacheManager(policy, max_cache_size=20)  # Big enough = no eviction

        cache = _make_mock_cache(seq_len=5)
        attn = _make_mock_attention(kv_len=5)

        mgr.update(cache, attn)
        assert len(mgr.snapshots) == 1
        assert mgr.snapshots[0].step == 0

    def test_no_eviction_when_under_budget(self):
        policy = StreamingLLMPolicy(n_sink=2, window_size=8)
        mgr = KVCacheManager(policy, max_cache_size=10)

        cache = _make_mock_cache(seq_len=5)
        attn = _make_mock_attention(kv_len=5)

        mgr.update(cache, attn)
        assert mgr.snapshots[0].cache_size == 5
        assert mgr.snapshots[0].evicted_indices == []

    def test_eviction_when_over_budget(self):
        policy = WindowOnlyPolicy(window_size=3)
        mgr = KVCacheManager(policy, max_cache_size=3)

        # Initial cache with 5 tokens (over budget of 3)
        cache = _make_mock_cache(seq_len=5)
        attn = _make_mock_attention(kv_len=5, q_len=5)

        result_cache = mgr.update(cache, attn)

        # Should have evicted some tokens
        snap = mgr.snapshots[0]
        assert snap.cache_size == 3
        assert len(snap.evicted_indices) > 0

    def test_token_map_tracks_positions(self):
        policy = StreamingLLMPolicy(n_sink=2, window_size=3)
        mgr = KVCacheManager(policy, max_cache_size=5)

        cache = _make_mock_cache(seq_len=4)
        attn = _make_mock_attention(kv_len=4, q_len=4)
        mgr.update(cache, attn)

        assert mgr.token_map == [0, 1, 2, 3]

    def test_reset_clears_state(self):
        policy = StreamingLLMPolicy()
        mgr = KVCacheManager(policy, max_cache_size=10)

        cache = _make_mock_cache(seq_len=5)
        attn = _make_mock_attention(kv_len=5)
        mgr.update(cache, attn)
        assert len(mgr.snapshots) == 1

        mgr.reset()
        assert len(mgr.snapshots) == 0
        assert len(mgr.token_map) == 0

    def test_visualization_state(self):
        policy = StreamingLLMPolicy(n_sink=2, window_size=5)
        mgr = KVCacheManager(policy, max_cache_size=20)

        cache = _make_mock_cache(seq_len=5)
        attn = _make_mock_attention(kv_len=5)
        mgr.update(cache, attn)

        state = mgr.get_visualization_state()
        assert "alive_tokens" in state
        assert "step" in state
        assert "cache_utilization" in state
        assert "attention_on_token0" in state

    def test_evict_from_legacy_cache(self):
        """Test physical eviction on legacy tuple-format cache."""
        policy = WindowOnlyPolicy(window_size=3)
        mgr = KVCacheManager(policy, max_cache_size=3)

        cache = _make_mock_cache(seq_len=5, n_layers=2, n_heads=4, head_dim=8)
        attn = _make_mock_attention(kv_len=5, q_len=5, n_layers=2, n_heads=4)
        result = mgr.update(cache, attn)

        # Verify resulting cache has only 3 tokens
        if isinstance(result, tuple):
            for layer_kv in result:
                k, v = layer_kv
                assert k.shape[2] == 3
                assert v.shape[2] == 3
