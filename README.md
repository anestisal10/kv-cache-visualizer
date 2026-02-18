<div align="center">

# 🧠 KV-Cache Eviction Visualizer

**An interactive tool for exploring KV-Cache eviction strategies in Large Language Models**

See which tokens survive, which are forgotten, and whether the *attention sink* phenomenon is real.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-6.0+-ff7c00?logo=gradio&logoColor=white)](https://gradio.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-28%2F28_passing-brightgreen)]()

</div>

---

## 🎯 What Is This?

When LLMs generate text, they store **Key-Value (KV) pairs** for every previous token to avoid recomputation. On consumer GPUs (like the RTX 2060 with 6 GB VRAM), this cache quickly fills up. **Eviction policies** decide which tokens to discard — and this choice has a dramatic effect on output quality.

This visualizer lets you **watch eviction happen in real time**, compare policies side-by-side, and investigate the [attention sink](https://arxiv.org/abs/2309.17453) hypothesis.

### Research Question

> *Is keeping the first token (the "attention sink") strictly necessary for small models like Qwen-2 0.5B?*

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎨 **Token Grid** | Color-coded tokens: 🟢 alive, 🔴 evicted, 🟡 sink, 🔵 latest |
| 🔥 **Attention Heatmaps** | Interactive Plotly heatmaps per layer/head |
| 📊 **Metrics Dashboard** | Perplexity, cache utilization, attention sink tracking |
| ⚖️ **Policy Comparison** | Run two policies side-by-side on the same prompt |
| 🔬 **Automated Experiments** | Test attention sink hypothesis across 6 configurations |
| 🚀 **Real-time Streaming** | Watch tokens appear and get evicted step-by-step |

---

## 📐 Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Gradio UI  │────▶│   Orchestrator   │────▶│  Model Backend  │
│  (3 tabs)   │     │  (gen loop)      │     │  (HF models)    │
└─────────────┘     └────────┬─────────┘     └─────────────────┘
                             │
                    ┌────────▼─────────┐
                    │  Cache Manager   │
                    │  (evict + track) │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        StreamingLLM       H2O        Window-Only
        (sink+window)  (heavy-hitter)  (baseline)
```

---

## ⚡ Eviction Policies

| Policy | Strategy | Key Insight |
|--------|----------|-------------|
| **StreamingLLM** | Keep first N "sink" tokens + sliding window | First token absorbs excess attention via softmax |
| **H2O** | Keep highest cumulative attention + recent window | ~5% of tokens receive >90% of attention mass |
| **Window-Only** | Pure sliding window (no sink) | Does removing the sink actually hurt small models? |
| **Random** | Randomly evict — worst case baseline | Lower bound on quality |
| **No Eviction** | Keep everything — quality upper bound | Reference for measuring degradation |

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/KV-Cache-Visualizer.git
cd KV-Cache-Visualizer

# Install dependencies
pip install -r requirements.txt

# Launch the visualizer
python app.py
```

Then open **http://localhost:7860** in your browser.

> **First run** will download Qwen2-0.5B-Instruct (~1 GB). Subsequent runs use the cached model.

---

## 🔬 Running Experiments

Test whether the attention sink is necessary:

```bash
python -m src.experiments.attention_sink_experiment
```

This runs **6 configurations** (no eviction, StreamingLLM with 4/1/0 sinks, window-only, random) across multiple prompts and saves results to `results/`.

---

## 🧪 Running Tests

```bash
# Fast tests (no GPU or model download required)
pytest tests/ -v -m "not slow"

# Full suite including model integration tests
pytest tests/ -v -m slow
```

**28 tests** covering all eviction policies and the cache manager.

---

## 🛠️ Target Hardware

Designed for **consumer GPUs**. Tested on **NVIDIA RTX 2060 (6 GB VRAM)**.

| Model | VRAM (FP16) | Quantization |
|-------|------------|--------------|
| Qwen2-0.5B-Instruct | ~1 GB | None needed |
| Qwen2-1.5B-Instruct | ~3 GB | None needed |
| Phi-3-mini-4k-instruct | ~4 GB | 4-bit auto-applied |

---

## 📁 Project Structure

```
├── app.py                              # Entry point — launches Gradio
├── requirements.txt                    # Python dependencies
├── configs/
│   └── default.yaml                    # Default configuration
├── src/
│   ├── model_backend.py                # HuggingFace model loading & attention extraction
│   ├── cache_manager.py                # KV-Cache wrapper with eviction & history tracking
│   ├── orchestrator.py                 # Token-by-token generation loop
│   ├── eviction_policies/
│   │   ├── base.py                     # Abstract base class
│   │   ├── streaming_llm.py            # Sink + sliding window
│   │   ├── h2o.py                      # Heavy-Hitter Oracle
│   │   ├── window_only.py              # Pure sliding window
│   │   ├── random_evict.py             # Random baseline
│   │   └── no_eviction.py              # Full cache upper bound
│   ├── ui/
│   │   ├── app.py                      # Main Gradio layout (3 tabs)
│   │   ├── token_grid.py               # Colored HTML token visualization
│   │   ├── heatmap.py                  # Plotly attention heatmaps
│   │   └── metrics.py                  # Perplexity & stats charts
│   └── experiments/
│       ├── attention_sink_experiment.py # Automated experiment runner
│       └── utils.py                    # Result saving utilities
├── tests/                              # Unit tests (28 tests)
├── collective_intelligence.md          # Detailed design document
└── LICENSE
```

---

## 📚 References

- **StreamingLLM** — Xiao et al., *"Efficient Streaming Language Models with Attention Sinks"*, ICLR 2024 — [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
- **H2O** — Zhang et al., *"H₂O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models"*, NeurIPS 2023
- **Attention Sinks** — The observation that the first token absorbs disproportionate attention mass due to softmax normalization

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
