"""
KV-Cache Manager — wraps the raw KV-Cache tensors, applies eviction policies,
and tracks the full history of alive/evicted tokens for visualization.

This is the central component that connects the model backend to the eviction
policies and provides the data the UI needs to render token grids and heatmaps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

from .eviction_policies.base import EvictionPolicy

logger = logging.getLogger(__name__)


@dataclass
class CacheSnapshot:
    """State of the cache at a single generation step."""
    step: int
    alive_indices: list[int]          # Indices (into original token list) that are alive
    evicted_indices: list[int]        # Indices evicted AT this step (newly evicted)
    cumulative_evicted: set[int]      # All indices evicted so far (cumulative)
    cache_size: int                   # Number of tokens currently in cache
    max_cache_size: int               # Budget
    attention_on_token0: float        # Average attention on token 0 (for sink analysis)


class KVCacheManager:
    """
    Manages the KV-Cache lifecycle during autoregressive generation.

    Responsibilities:
      1. Accept new KV entries and attention weights each step.
      2. Decide eviction via the configured policy.
      3. Physically evict tokens from the KV tensors.
      4. Record full history for the UI replay slider.
    """

    def __init__(self, policy: EvictionPolicy, max_cache_size: int):
        self.policy = policy
        self.max_cache_size = max_cache_size

        # ── Live state ───────────────────────────────────────────
        self.current_cache: Optional[tuple] = None  # HF past_key_values
        self.token_map: list[int] = []               # Maps cache-index → original token pos
        self.cumulative_attention: list[float] = []  # Cumulative attention per cache slot

        # ── History for visualization ────────────────────────────
        self.snapshots: list[CacheSnapshot] = []
        self.attention_snapshots: list[torch.Tensor | None] = []  # Selected attn per step
        self._cumulative_evicted: set[int] = set()
        self._step = 0

    def reset(self):
        """Reset all state for a new generation run."""
        self.current_cache = None
        self.token_map = []
        self.cumulative_attention = []
        self.snapshots = []
        self.attention_snapshots = []
        self._cumulative_evicted = set()
        self._step = 0

    def update(
        self,
        past_key_values,
        attention_weights: list[torch.Tensor],
        num_prompt_tokens: int = 0,
    ):
        """
        Called after each generation step (or after prefill).

        Args:
            past_key_values: The KV-Cache from the model (HF DynamicCache/tuple).
            attention_weights: List[n_layers] of (n_heads, q_len, kv_len) tensors.
            num_prompt_tokens: If this is the prefill call, pass the prompt length.
        """
        # Get the current cache sequence length (from the first layer's keys)
        cache_seq_len = self._get_cache_seq_len(past_key_values)

        # ── First call (prefill) — initialize token map ──────────
        if not self.token_map:
            self.token_map = list(range(cache_seq_len))
            self.cumulative_attention = [0.0] * cache_seq_len
        else:
            # A new token was appended
            new_pos = max(self.token_map) + 1 if self.token_map else 0
            self.token_map.append(new_pos)
            self.cumulative_attention.append(0.0)

        # ── Update cumulative attention scores ───────────────────
        self._update_attention_scores(attention_weights, cache_seq_len)

        # ── Compute attention on token 0 (for sink analysis) ─────
        attn_on_token0 = self._compute_attention_on_token0(attention_weights)

        # ── Store the current cache reference ────────────────────
        self.current_cache = past_key_values

        # ── Check if eviction is needed ──────────────────────────
        newly_evicted = []
        if cache_seq_len > self.max_cache_size:
            keep_indices = self.policy.select_tokens_to_keep(
                cache_size=cache_seq_len,
                max_cache_size=self.max_cache_size,
                attention_scores=self.cumulative_attention,
                token_positions=self.token_map,
            )

            # Determine which tokens were evicted
            all_indices = set(range(cache_seq_len))
            evict_indices = sorted(all_indices - set(keep_indices))
            newly_evicted_positions = [self.token_map[i] for i in evict_indices]
            newly_evicted = newly_evicted_positions
            self._cumulative_evicted.update(newly_evicted_positions)

            # Physically evict from cache
            self.current_cache = self._evict_from_cache(
                past_key_values, keep_indices
            )

            # Update token map and attention scores
            self.token_map = [self.token_map[i] for i in keep_indices]
            self.cumulative_attention = [self.cumulative_attention[i] for i in keep_indices]

            cache_seq_len = len(self.token_map)

        # ── Record snapshot ──────────────────────────────────────
        snapshot = CacheSnapshot(
            step=self._step,
            alive_indices=list(self.token_map),
            evicted_indices=newly_evicted,
            cumulative_evicted=set(self._cumulative_evicted),
            cache_size=cache_seq_len,
            max_cache_size=self.max_cache_size,
            attention_on_token0=attn_on_token0,
        )
        self.snapshots.append(snapshot)

        # Store a lightweight attention snapshot (last layer, mean over heads)
        if attention_weights:
            # Last layer → (n_heads, q_len, kv_len) → mean over heads → (q_len, kv_len)
            last_layer_attn = attention_weights[-1].mean(dim=0).cpu()
            self.attention_snapshots.append(last_layer_attn)
        else:
            self.attention_snapshots.append(None)

        self._step += 1
        return self.current_cache

    def get_visualization_state(self, step: int = -1) -> dict:
        """
        Return visualization data for a given step.

        Args:
            step: Step index (-1 for the latest).

        Returns:
            Dict with keys: alive_tokens, evicted_this_step,
            cumulative_evicted, cache_utilization, attention_heatmap,
            attention_on_token0.
        """
        if not self.snapshots:
            return {}

        snap = self.snapshots[step]
        attn = self.attention_snapshots[step] if step < len(self.attention_snapshots) else None

        return {
            "step": snap.step,
            "alive_tokens": snap.alive_indices,
            "evicted_this_step": snap.evicted_indices,
            "cumulative_evicted": list(snap.cumulative_evicted),
            "cache_size": snap.cache_size,
            "cache_utilization": snap.cache_size / snap.max_cache_size,
            "attention_heatmap": attn,
            "attention_on_token0": snap.attention_on_token0,
        }

    def get_full_attention_weights(
        self, attention_weights: list[torch.Tensor], layer: int = -1
    ) -> torch.Tensor | None:
        """
        Return full attention matrix for a specific layer.
        Used for detailed heatmap rendering.

        Args:
            attention_weights: Raw attention from model step output.
            layer: Layer index (-1 for last).

        Returns:
            Tensor of shape (n_heads, q_len, kv_len) or None.
        """
        if not attention_weights:
            return None
        return attention_weights[layer].cpu()

    # ── Private helpers ─────────────────────────────────────────────

    def _get_cache_seq_len(self, past_key_values) -> int:
        """Extract the sequence length from HF's cache format."""
        if past_key_values is None:
            return 0
        # DynamicCache (transformers >= 4.36)
        if hasattr(past_key_values, "get_seq_length"):
            return past_key_values.get_seq_length()
        # Legacy tuple format: tuple of (key, value) per layer
        # key shape: (batch, n_heads, seq_len, head_dim)
        if isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
            first_layer = past_key_values[0]
            if isinstance(first_layer, (tuple, list)):
                return first_layer[0].shape[2]
        return 0

    def _evict_from_cache(self, past_key_values, keep_indices: list[int]):
        """
        Create a new KV-Cache with only the tokens at `keep_indices`.
        Handles both DynamicCache and legacy tuple formats.
        """
        keep_tensor = torch.tensor(keep_indices, dtype=torch.long)

        # DynamicCache
        if hasattr(past_key_values, "key_cache"):
            from transformers.cache_utils import DynamicCache
            new_cache = DynamicCache()
            for layer_idx in range(len(past_key_values.key_cache)):
                k = past_key_values.key_cache[layer_idx]
                v = past_key_values.value_cache[layer_idx]
                device = k.device
                idx = keep_tensor.to(device)
                # k, v shape: (batch, n_kv_heads, seq_len, head_dim)
                new_k = k[:, :, idx, :]
                new_v = v[:, :, idx, :]
                new_cache.update(new_k, new_v, layer_idx)
            return new_cache

        # Legacy tuple format
        new_kv = []
        for layer_kv in past_key_values:
            k, v = layer_kv[0], layer_kv[1]
            device = k.device
            idx = keep_tensor.to(device)
            new_k = k[:, :, idx, :]
            new_v = v[:, :, idx, :]
            new_kv.append((new_k, new_v))
        return tuple(new_kv)

    def _update_attention_scores(
        self, attention_weights: list[torch.Tensor], cache_seq_len: int
    ):
        """Update cumulative attention scores for H2O policy."""
        if not attention_weights:
            return
        # Sum attention received by each token across all layers and heads
        # Each layer: (n_heads, q_len, kv_len) — sum over heads and query positions
        for layer_attn in attention_weights:
            # layer_attn: (n_heads, q_len, kv_len)
            # Attention received = sum over query positions for each key position
            # → (kv_len,) after summing over heads and q_len
            attn_received = layer_attn.sum(dim=(0, 1))  # (kv_len,)
            kv_len = attn_received.shape[0]
            # The kv_len may be less than cache_seq_len (if cache was just evicted)
            # Align to the END of our cumulative_attention list
            offset = len(self.cumulative_attention) - kv_len
            for i in range(kv_len):
                if 0 <= offset + i < len(self.cumulative_attention):
                    self.cumulative_attention[offset + i] += attn_received[i].item()

    def _compute_attention_on_token0(
        self, attention_weights: list[torch.Tensor]
    ) -> float:
        """Compute average attention mass on the first token (position 0)."""
        if not attention_weights or 0 not in self.token_map:
            return 0.0

        # Find which cache index holds original token 0
        try:
            cache_idx_of_token0 = self.token_map.index(0)
        except ValueError:
            return 0.0

        total = 0.0
        count = 0
        for layer_attn in attention_weights:
            # layer_attn: (n_heads, q_len, kv_len)
            if cache_idx_of_token0 < layer_attn.shape[-1]:
                # Attention on token 0 from the last query position
                attn_on_0 = layer_attn[:, -1, cache_idx_of_token0].mean().item()
                total += attn_on_0
                count += 1

        return total / count if count > 0 else 0.0
