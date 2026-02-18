"""
Main Gradio Application — interactive KV-Cache Eviction Visualizer.

Assembles all UI components into a cohesive interface with:
- Prompt input & model/policy selection
- Token grid showing alive/evicted tokens
- Attention heatmap per layer/head
- Perplexity & metrics dashboard
- Step-by-step replay slider
- Side-by-side comparison mode
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import gradio as gr
import torch

from ..model_backend import ModelBackend
from ..cache_manager import KVCacheManager
from ..orchestrator import GenerationOrchestrator
from ..eviction_policies import get_policy, list_policies, POLICY_REGISTRY
from .token_grid import render_token_grid, render_empty_grid
from .heatmap import create_attention_heatmap, create_attention_sink_chart
from .metrics import (
    create_metrics_table,
    create_perplexity_chart,
    create_cache_utilization_chart,
)

logger = logging.getLogger(__name__)

# ── Global state ─────────────────────────────────────────────────────
# We cache the model backend to avoid reloading on every generation
_model_cache: dict[str, ModelBackend] = {}
_lock = threading.Lock()


def _get_model(model_name: str, quantization: Optional[str] = None) -> ModelBackend:
    """Get or create a cached model backend."""
    key = f"{model_name}|{quantization}"
    with _lock:
        if key not in _model_cache:
            logger.info("Loading model: %s (quant=%s)", model_name, quantization)
            backend = ModelBackend(model_name, quantization=quantization)
            backend.load()
            _model_cache[key] = backend
        return _model_cache[key]


# ── Policy parameter builders ───────────────────────────────────────

def _build_policy(policy_name: str, n_sink: int, window_size: int, hh_budget: int):
    """Build a policy instance from UI controls."""
    if policy_name == "streaming_llm":
        return get_policy("streaming_llm", n_sink=n_sink, window_size=window_size)
    elif policy_name == "h2o":
        return get_policy("h2o", heavy_hitter_budget=hh_budget, recent_window=window_size)
    elif policy_name == "window_only":
        return get_policy("window_only", window_size=window_size)
    elif policy_name == "random":
        return get_policy("random")
    elif policy_name == "no_eviction":
        return get_policy("no_eviction")
    else:
        return get_policy(policy_name)


# ── Core generation function ────────────────────────────────────────

def run_generation(
    prompt: str,
    model_name: str,
    policy_name: str,
    max_new_tokens: int,
    max_cache_size: int,
    n_sink: int,
    window_size: int,
    hh_budget: int,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Main generation callback — runs the full pipeline and returns
    all UI component updates.
    """
    if not prompt.strip():
        yield (
            render_empty_grid(),  # token_grid
            create_attention_heatmap(None),  # heatmap
            create_perplexity_chart([]),  # ppl_chart
            create_cache_utilization_chart([]),  # cache_chart
            create_attention_sink_chart([]),  # sink_chart
            None,  # metrics_table
            "",  # generated_text
            gr.Slider(value=0, maximum=1, visible=False),  # step_slider
        )
        return

    # Load model
    quant = None
    if "Phi-3" in model_name:
        quant = "4bit"  # Phi-3 needs quantization on 6GB
    backend = _get_model(model_name, quant)

    # Build policy
    policy = _build_policy(policy_name, n_sink, window_size, hh_budget)
    cache_mgr = KVCacheManager(policy, max_cache_size)
    orchestrator = GenerationOrchestrator(backend, cache_mgr)

    # Collect all steps
    all_steps = []
    perplexities = []
    cache_history = []
    attn_sink_history = []
    all_tokens = []
    prompt_tokens = []

    for state in orchestrator.generate_stream(prompt, max_new_tokens=max_new_tokens):
        all_steps.append(state)
        perplexities.append(state.get("perplexity", 0.0))

        vis = state.get("visualization", {})
        cache_size = vis.get("cache_size", 0)
        cache_history.append((cache_size, max_cache_size))
        attn_sink_history.append(vis.get("attention_on_token0", 0.0))

        if state["step"] == 0:
            prompt_tokens = state.get("prompt_tokens", [])

        all_tokens = state.get("all_tokens", state.get("prompt_tokens", []))

        # Real-time updates during generation
        step_vis = state.get("visualization", {})
        alive = step_vis.get("alive_tokens", [])
        evicted = step_vis.get("cumulative_evicted", [])

        grid_html = render_token_grid(
            all_tokens=all_tokens,
            alive_indices=alive,
            cumulative_evicted=evicted,
            n_sink=n_sink if policy_name == "streaming_llm" else 0,
            current_step=state["step"],
            prompt_length=len(prompt_tokens),
        )

        attn_heatmap_data = step_vis.get("attention_heatmap", None)
        heatmap_fig = create_attention_heatmap(
            attn_heatmap_data,
            token_labels=all_tokens[-50:] if all_tokens else None,  # Last 50 for readability
            layer=-1,
        )

        ppl_chart = create_perplexity_chart(perplexities, policy.name)
        cache_chart = create_cache_utilization_chart(cache_history)
        sink_chart = create_attention_sink_chart(attn_sink_history)

        metrics_df = create_metrics_table(
            step=state["step"],
            perplexity=state.get("perplexity", 0.0),
            cache_size=cache_size,
            max_cache_size=max_cache_size,
            tokens_evicted=len(evicted),
            total_tokens=len(all_tokens),
            policy_name=policy.description,
            attention_on_token0=step_vis.get("attention_on_token0", 0.0),
        )

        gen_text = state.get("text_so_far", prompt)

        slider = gr.Slider(
            value=state["step"],
            maximum=max(state["step"], 1),
            step=1,
            label="Generation Step",
            visible=True,
        )

        yield (
            grid_html,
            heatmap_fig,
            ppl_chart,
            cache_chart,
            sink_chart,
            metrics_df,
            gen_text,
            slider,
        )


# ── Comparison mode ──────────────────────────────────────────────────

def run_comparison(
    prompt: str,
    model_name: str,
    policy_a: str,
    policy_b: str,
    max_new_tokens: int,
    max_cache_size: int,
    n_sink: int,
    window_size: int,
    hh_budget: int,
):
    """Run two policies and return side-by-side results."""
    quant = "4bit" if "Phi-3" in model_name else None
    backend = _get_model(model_name, quant)

    results = {}
    for pname in [policy_a, policy_b]:
        policy = _build_policy(pname, n_sink, window_size, hh_budget)
        cache_mgr = KVCacheManager(policy, max_cache_size)
        orch = GenerationOrchestrator(backend, cache_mgr)
        result = orch.generate(prompt, max_new_tokens=max_new_tokens)
        results[pname] = result

    # Build comparison outputs
    outputs = []
    for pname in [policy_a, policy_b]:
        r = results[pname]
        last_step = r.steps[-1] if r.steps else {}
        vis = last_step.get("visualization", {})
        all_tok = last_step.get("all_tokens", last_step.get("prompt_tokens", []))
        alive = vis.get("alive_tokens", [])
        evicted = vis.get("cumulative_evicted", [])

        grid = render_token_grid(
            all_tokens=all_tok,
            alive_indices=alive,
            cumulative_evicted=evicted,
            n_sink=n_sink if pname == "streaming_llm" else 0,
            current_step=last_step.get("step", 0),
            prompt_length=len(r.prompt_tokens),
        )
        ppl_chart = create_perplexity_chart(r.perplexity_per_step, pname)
        outputs.extend([grid, r.generated_text, ppl_chart])

    return outputs


# ── App Builder ──────────────────────────────────────────────────────

def create_app() -> gr.Blocks:
    """Build and return the complete Gradio app."""

    available_models = [
        "Qwen/Qwen2-0.5B-Instruct",
        "Qwen/Qwen2-1.5B-Instruct",
        "microsoft/Phi-3-mini-4k-instruct",
    ]

    policy_names = list_policies()

    with gr.Blocks(
        title="KV-Cache Eviction Visualizer",
    ) as app:

        # ── Header ──────────────────────────────────────────────
        gr.Markdown("""
        # 🧠 KV-Cache Eviction Visualizer
        *See which tokens survive and which are forgotten — explore attention sinks in small LLMs*
        """)

        with gr.Tabs():
            # ════════════════════════════════════════════════════
            # TAB 1 — Single Policy Explorer
            # ════════════════════════════════════════════════════
            with gr.Tab("🔍 Explorer", id="explorer"):
                with gr.Row():
                    # Left panel — controls
                    with gr.Column(scale=1, min_width=280):
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="Enter your prompt here…",
                            value="The key insight behind attention mechanisms in transformers is that",
                            lines=3,
                        )
                        model_select = gr.Dropdown(
                            choices=available_models,
                            value=available_models[0],
                            label="Model",
                        )
                        policy_select = gr.Dropdown(
                            choices=policy_names,
                            value="streaming_llm",
                            label="Eviction Policy",
                        )

                        with gr.Accordion("⚙️ Policy Parameters", open=True):
                            n_sink_slider = gr.Slider(
                                0, 16, value=4, step=1,
                                label="Sink Tokens (n_sink)",
                                info="Number of initial tokens to always keep",
                            )
                            window_slider = gr.Slider(
                                16, 512, value=252, step=4,
                                label="Window Size",
                                info="Number of recent tokens to keep",
                            )
                            hh_budget_slider = gr.Slider(
                                16, 256, value=128, step=8,
                                label="Heavy-Hitter Budget",
                                info="H2O: number of high-attention tokens to keep",
                            )

                        with gr.Accordion("🎛️ Generation Settings", open=False):
                            max_tokens_slider = gr.Slider(
                                10, 300, value=100, step=10,
                                label="Max New Tokens",
                            )
                            cache_size_slider = gr.Slider(
                                32, 512, value=256, step=16,
                                label="Max Cache Size",
                            )

                        generate_btn = gr.Button(
                            "🚀 Generate",
                            variant="primary",
                            size="lg",
                        )

                    # Right panel — visualizations
                    with gr.Column(scale=3):
                        token_grid = gr.HTML(
                            value=render_empty_grid(),
                            label="Token Grid",
                        )

                        step_slider = gr.Slider(
                            0, 1, value=0, step=1,
                            label="Generation Step",
                            visible=False,
                        )

                        generated_text = gr.Textbox(
                            label="Generated Text",
                            lines=3,
                            interactive=False,
                        )

                        with gr.Row():
                            with gr.Column():
                                heatmap_plot = gr.Plot(label="Attention Heatmap")
                            with gr.Column():
                                sink_chart = gr.Plot(label="Attention Sink Analysis")

                        with gr.Row():
                            with gr.Column():
                                ppl_chart = gr.Plot(label="Perplexity")
                            with gr.Column():
                                cache_chart = gr.Plot(label="Cache Utilization")

                        metrics_table = gr.Dataframe(
                            label="Metrics",
                            interactive=False,
                        )

                # Wire up generation
                generate_btn.click(
                    fn=run_generation,
                    inputs=[
                        prompt_input,
                        model_select,
                        policy_select,
                        max_tokens_slider,
                        cache_size_slider,
                        n_sink_slider,
                        window_slider,
                        hh_budget_slider,
                    ],
                    outputs=[
                        token_grid,
                        heatmap_plot,
                        ppl_chart,
                        cache_chart,
                        sink_chart,
                        metrics_table,
                        generated_text,
                        step_slider,
                    ],
                )

            # ════════════════════════════════════════════════════
            # TAB 2 — Side-by-Side Comparison
            # ════════════════════════════════════════════════════
            with gr.Tab("⚖️ Compare Policies", id="compare"):
                gr.Markdown("### Compare two eviction policies side by side")

                with gr.Row():
                    cmp_prompt = gr.Textbox(
                        label="Prompt",
                        value="The key insight behind attention mechanisms in transformers is that",
                        lines=2,
                    )

                with gr.Row():
                    cmp_model = gr.Dropdown(
                        choices=available_models,
                        value=available_models[0],
                        label="Model",
                    )
                    cmp_policy_a = gr.Dropdown(
                        choices=policy_names,
                        value="streaming_llm",
                        label="Policy A",
                    )
                    cmp_policy_b = gr.Dropdown(
                        choices=policy_names,
                        value="window_only",
                        label="Policy B",
                    )

                with gr.Row():
                    cmp_max_tokens = gr.Slider(10, 200, value=80, step=10, label="Max Tokens")
                    cmp_cache_size = gr.Slider(32, 512, value=256, step=16, label="Cache Size")
                    cmp_n_sink = gr.Slider(0, 16, value=4, step=1, label="n_sink")
                    cmp_window = gr.Slider(16, 512, value=252, step=4, label="Window")
                    cmp_hh = gr.Slider(16, 256, value=128, step=8, label="HH Budget")

                cmp_btn = gr.Button("⚖️ Compare", variant="primary", size="lg")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Policy A")
                        cmp_grid_a = gr.HTML(value=render_empty_grid())
                        cmp_text_a = gr.Textbox(label="Generated (A)", interactive=False, lines=3)
                        cmp_ppl_a = gr.Plot(label="Perplexity (A)")
                    with gr.Column():
                        gr.Markdown("#### Policy B")
                        cmp_grid_b = gr.HTML(value=render_empty_grid())
                        cmp_text_b = gr.Textbox(label="Generated (B)", interactive=False, lines=3)
                        cmp_ppl_b = gr.Plot(label="Perplexity (B)")

                cmp_btn.click(
                    fn=run_comparison,
                    inputs=[
                        cmp_prompt, cmp_model,
                        cmp_policy_a, cmp_policy_b,
                        cmp_max_tokens, cmp_cache_size,
                        cmp_n_sink, cmp_window, cmp_hh,
                    ],
                    outputs=[
                        cmp_grid_a, cmp_text_a, cmp_ppl_a,
                        cmp_grid_b, cmp_text_b, cmp_ppl_b,
                    ],
                )

            # ════════════════════════════════════════════════════
            # TAB 3 — About
            # ════════════════════════════════════════════════════
            with gr.Tab("📖 About", id="about"):
                gr.Markdown("""
                ## What is this?

                This tool visualizes **KV-Cache eviction** in Large Language Models.
                When generating text with limited GPU memory, models must decide
                which previous tokens to *keep* and which to *forget*.

                ### Eviction Policies

                | Policy | Strategy |
                |--------|----------|
                | **StreamingLLM** | Keep first N "sink" tokens + sliding window |
                | **H2O** | Keep tokens with highest cumulative attention + recent window |
                | **Window-Only** | Pure sliding window (no sink) — baseline |
                | **Random** | Randomly evict — worst case |
                | **No Eviction** | Keep everything — quality upper bound |

                ### Research Question

                > *Is keeping the first token (the attention sink) strictly necessary
                > for small models like Qwen-2 0.5B / 1.5B?*

                ### References

                - **StreamingLLM**: Xiao et al., ICLR 2024 — [arXiv](https://arxiv.org/abs/2309.17453)
                - **H2O**: Zhang et al., NeurIPS 2023
                - **Attention Sinks**: First token absorbs excess attention mass via softmax
                """)

    return app
