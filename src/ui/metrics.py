"""
Metrics Panel — renders perplexity, cache utilization, and generation stats.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def create_metrics_table(
    step: int,
    perplexity: float,
    cache_size: int,
    max_cache_size: int,
    tokens_evicted: int,
    total_tokens: int,
    policy_name: str,
    attention_on_token0: float = 0.0,
) -> pd.DataFrame:
    """
    Create a DataFrame for the metrics panel.
    """
    utilization = cache_size / max_cache_size * 100 if max_cache_size > 0 else 0

    data = {
        "Metric": [
            "Policy",
            "Generation Step",
            "Perplexity",
            "Cache Size",
            "Cache Utilization",
            "Tokens Evicted",
            "Total Tokens",
            "Attn on Token 0",
        ],
        "Value": [
            policy_name,
            str(step),
            f"{perplexity:.2f}",
            f"{cache_size} / {max_cache_size}",
            f"{utilization:.1f}%",
            str(tokens_evicted),
            str(total_tokens),
            f"{attention_on_token0:.4f}",
        ],
    }

    return pd.DataFrame(data)


def create_perplexity_chart(
    perplexity_history: list[float],
    policy_name: str = "",
) -> go.Figure:
    """
    Line chart of perplexity over generation steps.
    """
    if not perplexity_history:
        fig = go.Figure()
        fig.add_annotation(
            text="Generate tokens to see perplexity",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color="#6b7280", size=14),
        )
        fig.update_layout(
            plot_bgcolor="#0f1117", paper_bgcolor="#0f1117", height=280,
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        return fig

    steps = list(range(len(perplexity_history)))

    # Clamp extreme values for display
    clamped = [min(p, 1000) for p in perplexity_history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps,
        y=clamped,
        mode="lines",
        line=dict(color="#a78bfa", width=2),
        fill="tozeroy",
        fillcolor="rgba(167, 139, 250, 0.1)",
        name=policy_name or "Perplexity",
        hovertemplate="Step %{x}: PPL = %{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="Perplexity Over Generation Steps",
            font=dict(color="#e5e7eb", size=14),
        ),
        xaxis=dict(
            title="Step",
            title_font=dict(color="#9ca3af"),
            tickfont=dict(color="#9ca3af"),
            gridcolor="rgba(255,255,255,0.05)",
        ),
        yaxis=dict(
            title="Perplexity",
            title_font=dict(color="#9ca3af"),
            tickfont=dict(color="#9ca3af"),
            gridcolor="rgba(255,255,255,0.05)",
        ),
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font=dict(color="#e5e7eb"),
        margin=dict(l=60, r=20, t=50, b=50),
        height=280,
        showlegend=False,
    )

    return fig


def create_cache_utilization_chart(
    cache_history: list[tuple[int, int]],
) -> go.Figure:
    """
    Area chart showing cache size vs max over time.

    Args:
        cache_history: List of (cache_size, max_cache_size) per step.
    """
    if not cache_history:
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor="#0f1117", paper_bgcolor="#0f1117", height=200,
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        return fig

    steps = list(range(len(cache_history)))
    sizes = [h[0] for h in cache_history]
    maxes = [h[1] for h in cache_history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=maxes,
        mode="lines",
        line=dict(color="#6b7280", width=1, dash="dash"),
        name="Budget",
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=sizes,
        mode="lines",
        line=dict(color="#4ade80", width=2),
        fill="tozeroy",
        fillcolor="rgba(74, 222, 128, 0.1)",
        name="Cache Size",
        hovertemplate="Step %{x}: %{y} tokens<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Cache Utilization", font=dict(color="#e5e7eb", size=13)),
        xaxis=dict(title="Step", title_font=dict(color="#9ca3af"), tickfont=dict(color="#9ca3af"),
                   gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Tokens", title_font=dict(color="#9ca3af"), tickfont=dict(color="#9ca3af"),
                   gridcolor="rgba(255,255,255,0.05)"),
        plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
        font=dict(color="#e5e7eb"),
        margin=dict(l=50, r=20, t=40, b=40),
        height=220,
        legend=dict(x=0.02, y=0.98, font=dict(size=10)),
    )

    return fig
