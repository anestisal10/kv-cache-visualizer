"""
Model Backend — loads HuggingFace LLMs, manages tokenization, and extracts
attention weights at every generation step for KV-Cache visualization.

Designed for consumer GPU inference (RTX 2060, 6 GB VRAM).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class StepOutput:
    """Output of a single forward-pass generation step."""
    next_token_logits: torch.Tensor             # (vocab_size,)
    attention_weights: list[torch.Tensor]        # list[n_layers] of (n_heads, seq_len, seq_len)
    past_key_values: tuple                       # HuggingFace DynamicCache or tuple of (K, V)
    next_token_id: int                           # Greedy-decoded token id


class ModelBackend:
    """
    Wrapper around a HuggingFace CausalLM for step-by-step generation
    with attention weight extraction.

    Usage:
        backend = ModelBackend("Qwen/Qwen2-0.5B-Instruct")
        token_ids, token_strings = backend.tokenize("Hello world")
        step = backend.generate_step(token_ids)
    """

    # ── Supported models (tested) ────────────────────────────────────
    SUPPORTED_MODELS = [
        "Qwen/Qwen2-0.5B-Instruct",
        "Qwen/Qwen2-1.5B-Instruct",
        "microsoft/Phi-3-mini-4k-instruct",
    ]

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-0.5B-Instruct",
        quantization: Optional[str] = None,
        device: str = "cuda",
    ):
        """
        Args:
            model_name: HuggingFace model identifier.
            quantization: None (FP16), "4bit", or "8bit".
            device: "cuda" or "cpu".
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.quantization = quantization

        self.model = None
        self.tokenizer = None
        self._loaded = False

        logger.info(
            "ModelBackend created | model=%s  quant=%s  device=%s",
            model_name, quantization, self.device,
        )

    # ── Public API ──────────────────────────────────────────────────

    def load(self) -> "ModelBackend":
        """Load model and tokenizer. Returns self for chaining."""
        if self._loaded:
            logger.info("Model already loaded, skipping.")
            return self

        logger.info("Loading tokenizer for %s …", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading model for %s (quantization=%s) …", self.model_name, self.quantization)
        load_kwargs = self._build_load_kwargs()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            attn_implementation="eager",   # Need full attention matrices, not flash-attn
            **load_kwargs,
        )
        self.model.eval()

        # Move to device only if not quantized (bitsandbytes handles placement)
        if self.quantization is None:
            self.model = self.model.to(self.device)

        self._loaded = True
        logger.info(
            "Model loaded | params=%s  dtype=%s  device=%s",
            f"{sum(p.numel() for p in self.model.parameters()):,}",
            next(self.model.parameters()).dtype,
            next(self.model.parameters()).device,
        )
        return self

    def tokenize(self, text: str) -> tuple[list[int], list[str]]:
        """
        Tokenize text and return (token_ids, token_strings).

        Returns:
            token_ids: List of integer token IDs.
            token_strings: List of decoded token strings (human-readable).
        """
        self._ensure_loaded()
        encoding = self.tokenizer(text, return_tensors="pt")
        token_ids = encoding["input_ids"].squeeze(0).tolist()
        token_strings = [self.tokenizer.decode([tid]) for tid in token_ids]
        return token_ids, token_strings

    def decode(self, token_id: int) -> str:
        """Decode a single token ID to its string representation."""
        self._ensure_loaded()
        return self.tokenizer.decode([token_id])

    @torch.no_grad()
    def generate_step(
        self,
        input_ids: list[int] | torch.Tensor,
        past_key_values=None,
    ) -> StepOutput:
        """
        Run ONE forward pass and return logits, attention weights, and updated cache.

        If `past_key_values` is provided, only the last token is fed as input
        (KV-Cache continuation). Otherwise, the full sequence is processed.

        Args:
            input_ids: Full token sequence so far (list or 1-D tensor).
            past_key_values: KV-Cache from previous step (or None for first step).

        Returns:
            StepOutput with logits, attention weights, cache, and greedy next token.
        """
        self._ensure_loaded()

        # Prepare input tensor
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_ids = input_ids.to(self.device)

        # If we have a cache, only feed the LAST token (cache has the rest)
        if past_key_values is not None:
            model_input_ids = input_ids[-1:].unsqueeze(0)  # (1, 1)
        else:
            model_input_ids = input_ids.unsqueeze(0)  # (1, seq_len)

        # Forward pass
        outputs = self.model(
            input_ids=model_input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )

        # Extract outputs
        # Logits for the last position → (vocab_size,)
        next_token_logits = outputs.logits[0, -1, :]

        # Attention weights: list of (1, n_heads, q_len, kv_len) → squeeze batch
        attention_weights = [
            attn.squeeze(0)  # → (n_heads, q_len, kv_len)
            for attn in outputs.attentions
        ]

        # Greedy next token
        next_token_id = next_token_logits.argmax(dim=-1).item()

        return StepOutput(
            next_token_logits=next_token_logits,
            attention_weights=attention_weights,
            past_key_values=outputs.past_key_values,
            next_token_id=next_token_id,
        )

    @torch.no_grad()
    def prefill(
        self, input_ids: list[int] | torch.Tensor
    ) -> tuple[list[torch.Tensor], object]:
        """
        Run the prefill (prompt processing) phase.
        Returns attention weights and the KV-Cache for all prompt tokens.

        Args:
            input_ids: The full prompt token IDs.

        Returns:
            (attention_weights, past_key_values)
        """
        self._ensure_loaded()

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_ids = input_ids.to(self.device).unsqueeze(0)  # (1, seq_len)

        outputs = self.model(
            input_ids=input_ids,
            past_key_values=None,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )

        attention_weights = [
            attn.squeeze(0) for attn in outputs.attentions
        ]

        return attention_weights, outputs.past_key_values

    # ── Model info helpers ──────────────────────────────────────────

    @property
    def num_layers(self) -> int:
        """Number of transformer layers."""
        self._ensure_loaded()
        return self.model.config.num_hidden_layers

    @property
    def num_attention_heads(self) -> int:
        """Number of query attention heads."""
        self._ensure_loaded()
        return self.model.config.num_attention_heads

    @property
    def num_kv_heads(self) -> int:
        """Number of key/value heads (may differ from query heads in GQA)."""
        self._ensure_loaded()
        config = self.model.config
        return getattr(config, "num_key_value_heads", config.num_attention_heads)

    @property
    def vocab_size(self) -> int:
        self._ensure_loaded()
        return self.model.config.vocab_size

    # ── Private helpers ─────────────────────────────────────────────

    def _ensure_loaded(self):
        """Auto-load model if not loaded yet."""
        if not self._loaded:
            self.load()

    def _build_load_kwargs(self) -> dict:
        """Build kwargs for AutoModelForCausalLM.from_pretrained."""
        kwargs = {}
        if self.quantization == "4bit":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            kwargs["device_map"] = "auto"
        elif self.quantization == "8bit":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            kwargs["device_map"] = "auto"
        else:
            # FP16 / BF16 — pick the best dtype for this GPU
            kwargs["dtype"] = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            ) if self.device == "cuda" else torch.float32

        return kwargs
