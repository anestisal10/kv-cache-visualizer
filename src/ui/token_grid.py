"""
Token Grid Visualization — renders tokens as a colored HTML grid.

Colors:
  • Green (#4ade80)  — token is alive in the KV-Cache
  • Red (#f87171)    — token was evicted
  • Gold (#fbbf24)   — attention sink token (position 0)
  • Blue (#60a5fa)   — most recently generated token
"""

from __future__ import annotations

import html


def render_token_grid(
    all_tokens: list[str],
    alive_indices: list[int],
    cumulative_evicted: list[int],
    n_sink: int = 4,
    current_step: int = 0,
    prompt_length: int = 0,
) -> str:
    """
    Render the full token sequence as a colored HTML grid.

    Args:
        all_tokens: All tokens in the sequence (prompt + generated).
        alive_indices: Token indices currently alive in cache.
        cumulative_evicted: Token indices evicted so far.
        n_sink: Number of sink tokens (highlighted in gold).
        current_step: Current generation step.
        prompt_length: Length of the prompt in tokens.

    Returns:
        HTML string for Gradio's gr.HTML component.
    """
    alive_set = set(alive_indices)
    evicted_set = set(cumulative_evicted)

    tokens_html = []
    for idx, tok in enumerate(all_tokens):
        tok_display = html.escape(tok).replace(" ", "&nbsp;")
        if not tok_display.strip():
            tok_display = repr(tok).strip("'\"")
            tok_display = html.escape(tok_display)

        # Determine color
        if idx == len(all_tokens) - 1 and idx >= prompt_length:
            # Latest generated token
            color = "#60a5fa"
            bg = "rgba(96, 165, 250, 0.15)"
            border = "2px solid #60a5fa"
            label = "new"
        elif idx in evicted_set:
            color = "#f87171"
            bg = "rgba(248, 113, 113, 0.12)"
            border = "1px solid rgba(248, 113, 113, 0.3)"
            label = "evicted"
        elif idx < n_sink and idx in alive_set:
            color = "#fbbf24"
            bg = "rgba(251, 191, 36, 0.15)"
            border = "1px solid rgba(251, 191, 36, 0.4)"
            label = "sink"
        elif idx in alive_set:
            color = "#4ade80"
            bg = "rgba(74, 222, 128, 0.12)"
            border = "1px solid rgba(74, 222, 128, 0.3)"
            label = "alive"
        else:
            color = "#6b7280"
            bg = "rgba(107, 114, 128, 0.08)"
            border = "1px solid rgba(107, 114, 128, 0.2)"
            label = "unknown"

        # Token badge
        token_html = f"""
        <span class="token-badge token-{label}" style="
            display: inline-flex;
            align-items: center;
            gap: 3px;
            padding: 3px 7px;
            margin: 2px;
            border-radius: 6px;
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 12px;
            color: {color};
            background: {bg};
            border: {border};
            cursor: default;
            transition: all 0.2s ease;
        " title="Position {idx} | {label}">
            <span style="opacity:0.4; font-size:9px; margin-right:2px;">{idx}</span>
            {tok_display}
        </span>
        """
        tokens_html.append(token_html)

    # Legend
    legend = """
    <div style="
        display: flex; gap: 16px; margin-bottom: 12px; padding: 8px 12px;
        background: rgba(255,255,255,0.03); border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.06); font-size: 12px;
    ">
        <span style="color: #fbbf24;">● Sink</span>
        <span style="color: #4ade80;">● Alive</span>
        <span style="color: #f87171;">● Evicted</span>
        <span style="color: #60a5fa;">● Latest</span>
        <span style="color: #6b7280; margin-left: auto;">
            Step {step} | Cache: {alive}/{total} tokens
        </span>
    </div>
    """.format(
        step=current_step,
        alive=len(alive_indices),
        total=len(all_tokens),
    )

    body = '<div style="display: flex; flex-wrap: wrap; gap: 1px; padding: 8px;">'
    body += "".join(tokens_html)
    body += "</div>"

    return f"""
    <div style="
        background: #0f1117;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(255,255,255,0.08);
    ">
        {legend}
        {body}
    </div>
    """


def render_empty_grid() -> str:
    """Render a placeholder when no generation has run yet."""
    return """
    <div style="
        background: #0f1117;
        border-radius: 12px;
        padding: 40px;
        text-align: center;
        color: #6b7280;
        border: 1px solid rgba(255,255,255,0.08);
        font-family: 'Inter', sans-serif;
    ">
        <p style="font-size: 16px; margin: 0;">Enter a prompt and click Generate to see token-level KV-Cache eviction</p>
    </div>
    """
