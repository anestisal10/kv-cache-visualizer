"""
Generation Orchestrator — ties together the model backend, KV-Cache manager,
and eviction policies to run the full token-by-token generation loop.

Each step yields a visualization state dict that the UI can render.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Generator

import torch

from .model_backend import ModelBackend
from .cache_manager import KVCacheManager

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Complete result of a generation run."""
    prompt: str
    prompt_tokens: list[str]
    generated_tokens: list[str]
    generated_text: str
    steps: list[dict]           # Visualization states per step
    total_time: float           # Wall-clock seconds
    perplexity_per_step: list[float]


class GenerationOrchestrator:
    """
    Coordinates model inference, KV-Cache management, and eviction
    to produce step-by-step visualization data.

    Usage:
        backend = ModelBackend("Qwen/Qwen2-0.5B-Instruct")
        cache_mgr = KVCacheManager(policy, max_cache_size=256)
        orch = GenerationOrchestrator(backend, cache_mgr)

        # Streaming — yields per step
        for state in orch.generate_stream("Hello", max_new_tokens=50):
            update_ui(state)

        # Blocking — returns full result
        result = orch.generate("Hello", max_new_tokens=50)
    """

    def __init__(self, model_backend: ModelBackend, cache_manager: KVCacheManager):
        self.model = model_backend
        self.cache = cache_manager

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> Generator[dict, None, None]:
        """
        Token-by-token generation with eviction, yielding visualization
        state at each step.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (1.0 = greedy via argmax).
            top_k: Top-k sampling (0 = disabled, argmax).

        Yields:
            Dict with: step, token, text_so_far, visualization, perplexity.
        """
        self.cache.reset()
        self.model._ensure_loaded()

        # ── Tokenize ─────────────────────────────────────────────
        token_ids, token_strings = self.model.tokenize(prompt)
        all_token_strings = list(token_strings)
        generated_tokens = []

        logger.info(
            "Starting generation | prompt_len=%d  max_new=%d  policy=%s",
            len(token_ids), max_new_tokens, self.cache.policy.name,
        )

        # ── Prefill — process the entire prompt at once ──────────
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attn_weights, past_kv = self.model.prefill(input_ids)

        # Update cache manager with prefill state
        past_kv = self.cache.update(
            past_kv, attn_weights, num_prompt_tokens=len(token_ids)
        )

        # Yield prefill state
        yield {
            "step": 0,
            "phase": "prefill",
            "token": None,
            "text_so_far": prompt,
            "prompt_tokens": token_strings,
            "generated_tokens": [],
            "visualization": self.cache.get_visualization_state(),
            "perplexity": 0.0,
        }

        # ── Decode loop — generate one token at a time ───────────
        running_log_prob_sum = 0.0
        current_ids = list(token_ids)

        for step in range(1, max_new_tokens + 1):
            # Forward pass with the KV-Cache
            step_output = self.model.generate_step(
                current_ids, past_key_values=past_kv,
            )

            # Sample / argmax next token
            next_id = self._sample_token(
                step_output.next_token_logits, temperature, top_k
            )
            next_str = self.model.decode(next_id)

            # Track tokens
            current_ids.append(next_id)
            generated_tokens.append(next_str)
            all_token_strings.append(next_str)

            # Compute step perplexity
            log_probs = torch.nn.functional.log_softmax(
                step_output.next_token_logits, dim=-1
            )
            token_log_prob = log_probs[next_id].item()
            running_log_prob_sum += token_log_prob
            avg_nll = -running_log_prob_sum / step
            perplexity = min(torch.exp(torch.tensor(avg_nll)).item(), 1e6)

            # Update cache with new KV + attention
            past_kv = self.cache.update(
                step_output.past_key_values,
                step_output.attention_weights,
            )

            # Yield step state
            yield {
                "step": step,
                "phase": "decode",
                "token": next_str,
                "token_id": next_id,
                "text_so_far": prompt + "".join(generated_tokens),
                "prompt_tokens": token_strings,
                "generated_tokens": list(generated_tokens),
                "all_tokens": list(all_token_strings),
                "visualization": self.cache.get_visualization_state(),
                "perplexity": perplexity,
            }

            # Stop on EOS
            if next_id == self.model.tokenizer.eos_token_id:
                logger.info("EOS reached at step %d", step)
                break

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> GenerationResult:
        """
        Blocking generation — runs the full loop and returns a summary.
        """
        t0 = time.time()
        steps = []
        perplexities = []
        prompt_tokens = []
        gen_tokens = []

        for state in self.generate_stream(prompt, max_new_tokens, temperature, top_k):
            steps.append(state)
            perplexities.append(state.get("perplexity", 0.0))
            if state["step"] == 0:
                prompt_tokens = state["prompt_tokens"]
            if state.get("token"):
                gen_tokens.append(state["token"])

        elapsed = time.time() - t0
        gen_text = "".join(gen_tokens)

        logger.info(
            "Generation complete | %d steps in %.2fs (%.1f tok/s)",
            len(steps), elapsed, len(gen_tokens) / elapsed if elapsed > 0 else 0,
        )

        return GenerationResult(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            generated_tokens=gen_tokens,
            generated_text=gen_text,
            steps=steps,
            total_time=elapsed,
            perplexity_per_step=perplexities,
        )

    # ── Private helpers ─────────────────────────────────────────────

    @staticmethod
    def _sample_token(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> int:
        """Sample or argmax from logits."""
        if temperature <= 0 or top_k == 0:
            return logits.argmax(dim=-1).item()

        logits = logits / temperature
        if top_k > 0:
            top_k_vals, top_k_idx = torch.topk(logits, min(top_k, logits.shape[-1]))
            probs = torch.nn.functional.softmax(top_k_vals, dim=-1)
            sampled = torch.multinomial(probs, 1)
            return top_k_idx[sampled].item()

        probs = torch.nn.functional.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()
