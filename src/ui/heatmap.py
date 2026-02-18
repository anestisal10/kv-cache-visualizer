"""
Attention Heatmap — interactive Plotly heatmap for visualizing attention patterns.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go


def create_attention_heatmap(
    attention_matrix: np.ndarray | None,
    token_labels: list[str] | None = None,
    title: str = "Attention Weights",
    layer: int = -1,
    head: int | None = None,
) -> go.Figure:
    """
    Create an interactive Plotly heatmap of attention weights.

    Args:
        attention_matrix: Shape (n_heads, q_len, kv_len) or (q_len, kv_len).
        token_labels: Labels for both axes.
        title: Plot title.
        layer: Layer index (for display only).
        head: Specific head to show. None = mean over heads.

    Returns:
        Plotly Figure object.
    """
    if attention_matrix is None:
        return _empty_heatmap("No attention data available")

    # Convert tensor to numpy if needed
    if hasattr(attention_matrix, "numpy"):
        attention_matrix = attention_matrix.detach().cpu().float().numpy()

    # Select head or average
    if attention_matrix.ndim == 3:
        n_heads = attention_matrix.shape[0]
        if head is not None and 0 <= head < n_heads:
            data = attention_matrix[head]
            head_label = f"Head {head}"
        else:
            data = attention_matrix.mean(axis=0)
            head_label = "Mean"
    elif attention_matrix.ndim == 2:
        data = attention_matrix
        head_label = ""
    else:
        return _empty_heatmap("Invalid attention shape")

    # Truncate labels if too long
    if token_labels:
        max_tokens = min(len(token_labels), data.shape[-1])
        x_labels = [f"{i}:{_trunc(t)}" for i, t in enumerate(token_labels[:max_tokens])]
        y_labels = x_labels[-data.shape[0]:]
    else:
        x_labels = [str(i) for i in range(data.shape[-1])]
        y_labels = [str(i) for i in range(data.shape[0])]

    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=x_labels,
        y=y_labels,
        colorscale=[
            [0.0, "#0f1117"],
            [0.2, "#1e1b4b"],
            [0.4, "#4338ca"],
            [0.6, "#7c3aed"],
            [0.8, "#c084fc"],
            [1.0, "#fbbf24"],
        ],
        colorbar=dict(
            title="Attn",
            title_font=dict(color="#9ca3af", size=11),
            tickfont=dict(color="#9ca3af", size=10),
        ),
        hovertemplate="Query: %{y}<br>Key: %{x}<br>Attention: %{z:.4f}<extra></extra>",
    ))

    display_title = f"{title} — Layer {layer}"
    if head_label:
        display_title += f" | {head_label}"

    fig.update_layout(
        title=dict(text=display_title, font=dict(color="#e5e7eb", size=14)),
        xaxis=dict(
            title="Key Position",
            tickfont=dict(color="#9ca3af", size=8),
            title_font=dict(color="#9ca3af", size=11),
            tickangle=-45,
        ),
        yaxis=dict(
            title="Query Position",
            tickfont=dict(color="#9ca3af", size=8),
            title_font=dict(color="#9ca3af", size=11),
            autorange="reversed",
        ),
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font=dict(color="#e5e7eb"),
        margin=dict(l=60, r=20, t=50, b=80),
        height=450,
    )

    return fig


def create_attention_sink_chart(
    attention_on_token0_history: list[float],
) -> go.Figure:
    """
    Line chart showing attention mass on token 0 over generation steps.
    Useful for visualizing the 'attention sink' phenomenon.
    """
    if not attention_on_token0_history:
        return _empty_heatmap("No attention sink data yet")

    steps = list(range(len(attention_on_token0_history)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps,
        y=attention_on_token0_history,
        mode="lines+markers",
        line=dict(color="#fbbf24", width=2),
        marker=dict(size=4, color="#fbbf24"),
        name="Attn on Token 0",
        hovertemplate="Step %{x}: %{y:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="Attention Mass on Token 0 (Sink Analysis)",
            font=dict(color="#e5e7eb", size=14),
        ),
        xaxis=dict(
            title="Generation Step",
            title_font=dict(color="#9ca3af"),
            tickfont=dict(color="#9ca3af"),
            gridcolor="rgba(255,255,255,0.05)",
        ),
        yaxis=dict(
            title="Avg Attention on Token 0",
            title_font=dict(color="#9ca3af"),
            tickfont=dict(color="#9ca3af"),
            gridcolor="rgba(255,255,255,0.05)",
            range=[0, max(attention_on_token0_history) * 1.2 + 0.01],
        ),
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font=dict(color="#e5e7eb"),
        margin=dict(l=60, r=20, t=50, b=50),
        height=300,
    )

    return fig


def _empty_heatmap(message: str) -> go.Figure:
    """Return an empty figure with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(color="#6b7280", size=14),
    )
    fig.update_layout(
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        height=350,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _trunc(s: str, max_len: int = 8) -> str:
    """Truncate a token string for axis labels."""
    s = s.strip()
    if len(s) > max_len:
        return s[:max_len - 1] + "…"
    return s
