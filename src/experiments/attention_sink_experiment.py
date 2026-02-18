"""
Attention Sink Experiment — automated experiment runner that tests whether
the attention sink (first token) is strictly necessary for small models.

Protocol:
  1. Baseline: Full cache (no eviction)
  2. StreamingLLM (n_sink=4, window=128): Standard config
  3. StreamingLLM (n_sink=1, window=131): Minimal sink
  4. StreamingLLM (n_sink=0, window=132): NO sink — does it collapse?
  5. Window-Only (window=128): No sink, same budget as #2
  6. Random eviction: Worst case baseline
"""

from __future__ import annotations

import logging
from typing import Optional

from ..model_backend import ModelBackend
from ..cache_manager import KVCacheManager
from ..orchestrator import GenerationOrchestrator
from ..eviction_policies import get_policy
from .utils import save_experiment_results

logger = logging.getLogger(__name__)


class AttentionSinkExperiment:
    """
    Runs the attention sink experiment across multiple policies
    and saves results for analysis.
    """

    DEFAULT_CONFIGS = [
        {
            "name": "no_eviction",
            "policy": "no_eviction",
            "params": {},
            "max_cache_size": 9999,
        },
        {
            "name": "streaming_4sink",
            "policy": "streaming_llm",
            "params": {"n_sink": 4, "window_size": 128},
            "max_cache_size": 132,
        },
        {
            "name": "streaming_1sink",
            "policy": "streaming_llm",
            "params": {"n_sink": 1, "window_size": 131},
            "max_cache_size": 132,
        },
        {
            "name": "streaming_0sink",
            "policy": "streaming_llm",
            "params": {"n_sink": 0, "window_size": 132},
            "max_cache_size": 132,
        },
        {
            "name": "window_only",
            "policy": "window_only",
            "params": {"window_size": 128},
            "max_cache_size": 128,
        },
        {
            "name": "random",
            "policy": "random",
            "params": {},
            "max_cache_size": 132,
        },
    ]

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-0.5B-Instruct",
        quantization: Optional[str] = None,
        max_new_tokens: int = 200,
    ):
        self.model_name = model_name
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.backend = ModelBackend(model_name, quantization=quantization)

    def run(
        self,
        prompts: list[str] | None = None,
        configs: list[dict] | None = None,
    ) -> dict:
        """
        Run the full experiment suite.

        Args:
            prompts: List of prompts to test. If None, uses defaults.
            configs: Policy configs to test. If None, uses DEFAULT_CONFIGS.

        Returns:
            Dict mapping config_name → results dict.
        """
        if prompts is None:
            prompts = [
                "The key insight behind attention mechanisms in transformers is that",
                "In a distant galaxy, a civilization of sentient machines had evolved to",
            ]

        if configs is None:
            configs = self.DEFAULT_CONFIGS

        self.backend.load()

        all_results = {}

        for config in configs:
            config_name = config["name"]
            logger.info("=" * 50)
            logger.info("Running config: %s", config_name)
            logger.info("=" * 50)

            policy = get_policy(config["policy"], **config["params"])
            prompt_results = []

            for prompt_idx, prompt in enumerate(prompts):
                logger.info(
                    "  Prompt %d/%d: %s...", prompt_idx + 1, len(prompts), prompt[:50]
                )

                cache_mgr = KVCacheManager(policy, config["max_cache_size"])
                orch = GenerationOrchestrator(self.backend, cache_mgr)
                result = orch.generate(prompt, max_new_tokens=self.max_new_tokens)

                # Collect attention-on-token-0 history
                attn0_history = []
                for step_data in result.steps:
                    vis = step_data.get("visualization", {})
                    attn0_history.append(vis.get("attention_on_token0", 0.0))

                prompt_results.append({
                    "prompt": prompt,
                    "generated_text": result.generated_text,
                    "perplexity_per_step": result.perplexity_per_step,
                    "final_perplexity": (
                        result.perplexity_per_step[-1]
                        if result.perplexity_per_step else float("inf")
                    ),
                    "attention_on_token0": attn0_history,
                    "total_tokens_evicted": len(
                        result.steps[-1].get("visualization", {}).get(
                            "cumulative_evicted", []
                        )
                    ) if result.steps else 0,
                    "total_time": result.total_time,
                })

                logger.info(
                    "    Final PPL: %.2f | Time: %.1fs",
                    prompt_results[-1]["final_perplexity"],
                    result.total_time,
                )

            # Average across prompts
            avg_ppl = sum(
                r["final_perplexity"] for r in prompt_results
            ) / len(prompt_results)

            all_results[config_name] = {
                "prompt_results": prompt_results,
                "final_perplexity": avg_ppl,
                "perplexity_per_step": prompt_results[0]["perplexity_per_step"],
                "generated_text": prompt_results[0]["generated_text"],
                "attention_on_token0": prompt_results[0]["attention_on_token0"],
                "total_tokens_evicted": prompt_results[0]["total_tokens_evicted"],
            }

            logger.info("  Avg final perplexity: %.2f", avg_ppl)

        # Save all results
        save_experiment_results(
            experiment_name="attention_sink",
            results=all_results,
            metadata={
                "model": self.model_name,
                "quantization": self.quantization,
                "max_new_tokens": self.max_new_tokens,
                "num_prompts": len(prompts),
            },
        )

        self._print_summary(all_results)
        return all_results

    def _print_summary(self, results: dict):
        """Print a summary table of results."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("  EXPERIMENT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"{'Config':<25} {'Avg PPL':>10} {'Evicted':>10}")
        logger.info("-" * 50)
        for name, data in results.items():
            logger.info(
                f"{name:<25} {data['final_perplexity']:>10.2f} "
                f"{data['total_tokens_evicted']:>10}"
            )
        logger.info("=" * 60)


def run_quick_experiment(
    model_name: str = "Qwen/Qwen2-0.5B-Instruct",
    prompt: str = "The key insight behind attention mechanisms in transformers is that",
    max_new_tokens: int = 100,
) -> dict:
    """
    Convenience function for a quick experiment with a single prompt.
    """
    exp = AttentionSinkExperiment(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
    )
    return exp.run(prompts=[prompt])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    run_quick_experiment()
