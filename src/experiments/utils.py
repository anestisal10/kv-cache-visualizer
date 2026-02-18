"""
Experiment Utilities — perplexity calculations, logging, and result saving.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"


def ensure_dirs():
    """Create results directories if they don't exist."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def compute_perplexity(log_probs: list[float]) -> float:
    """
    Compute perplexity from a list of per-token log probabilities.
    PPL = exp(-1/N * Σ log P(token_i))
    """
    if not log_probs:
        return float("inf")
    avg_nll = -np.mean(log_probs)
    return float(np.exp(avg_nll))


def save_experiment_results(
    experiment_name: str,
    results: dict,
    metadata: dict | None = None,
):
    """
    Save experiment results as JSON + CSV.

    Args:
        experiment_name: Name of the experiment run.
        results: Dict mapping policy_name → {perplexities, generated_text, ...}.
        metadata: Optional metadata (model, timestamp, etc.).
    """
    ensure_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{experiment_name}_{timestamp}"

    # Save JSON
    json_path = LOGS_DIR / f"{base_name}.json"
    output = {
        "experiment": experiment_name,
        "timestamp": timestamp,
        "metadata": metadata or {},
        "results": {},
    }
    for policy_name, data in results.items():
        output["results"][policy_name] = {
            "perplexity_per_step": data.get("perplexity_per_step", []),
            "final_perplexity": data.get("final_perplexity", 0),
            "generated_text": data.get("generated_text", ""),
            "attention_on_token0": data.get("attention_on_token0", []),
            "total_tokens_evicted": data.get("total_tokens_evicted", 0),
        }

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", json_path)

    # Save CSV summary
    csv_path = LOGS_DIR / f"{base_name}_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "policy", "final_perplexity", "tokens_evicted",
            "avg_attn_on_token0", "generated_text_preview"
        ])
        for policy_name, data in results.items():
            attn0 = data.get("attention_on_token0", [])
            avg_attn0 = np.mean(attn0) if attn0 else 0
            writer.writerow([
                policy_name,
                f"{data.get('final_perplexity', 0):.4f}",
                data.get("total_tokens_evicted", 0),
                f"{avg_attn0:.6f}",
                data.get("generated_text", "")[:100],
            ])
    logger.info("Summary CSV saved to %s", csv_path)

    return json_path, csv_path
