"""
KV-Cache Eviction Visualizer — Entry Point

Launch the Gradio app:
    python app.py
"""

import logging
import sys

import gradio as gr

# ── Logging setup ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("kv_cache_visualizer")


def main():
    """Launch the Gradio app."""
    from src.ui.app import create_app

    logger.info("=" * 60)
    logger.info("  KV-Cache Eviction Visualizer")
    logger.info("=" * 60)

    app = create_app()
    app.queue()  # Enable queuing for streaming updates
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Base(
            primary_hue="violet",
            secondary_hue="amber",
            neutral_hue="slate",
            font=["Inter", "system-ui", "sans-serif"],
        ),
        css="""
        .gradio-container {
            max-width: 1400px !important;
            background: #0a0b0f !important;
        }
        .dark {
            background: #0a0b0f !important;
        }
        footer { display: none !important; }
        """,
    )


if __name__ == "__main__":
    main()
