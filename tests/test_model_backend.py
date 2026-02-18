"""
Tests for the model backend.

Note: Tests that require model downloads are marked with @pytest.mark.slow
and skipped by default. Run them with: pytest -m slow
"""

import pytest
from src.model_backend import ModelBackend


class TestModelBackendInit:
    """Test ModelBackend initialization (no model download)."""

    def test_default_init(self):
        backend = ModelBackend()
        assert backend.model_name == "Qwen/Qwen2-0.5B-Instruct"
        assert backend.quantization is None
        assert backend._loaded is False

    def test_custom_model(self):
        backend = ModelBackend("Qwen/Qwen2-1.5B-Instruct", quantization="4bit")
        assert backend.model_name == "Qwen/Qwen2-1.5B-Instruct"
        assert backend.quantization == "4bit"

    def test_supported_models_list(self):
        assert len(ModelBackend.SUPPORTED_MODELS) >= 3


@pytest.mark.slow
class TestModelBackendWithModel:
    """Tests that require model download — run with pytest -m slow."""

    @pytest.fixture(scope="class")
    def backend(self):
        b = ModelBackend("Qwen/Qwen2-0.5B-Instruct", device="cpu")
        b.load()
        return b

    def test_load_model(self, backend):
        assert backend._loaded is True
        assert backend.model is not None
        assert backend.tokenizer is not None

    def test_tokenize(self, backend):
        ids, strings = backend.tokenize("Hello world")
        assert len(ids) > 0
        assert len(ids) == len(strings)
        assert all(isinstance(i, int) for i in ids)

    def test_decode(self, backend):
        ids, _ = backend.tokenize("test")
        decoded = backend.decode(ids[0])
        assert isinstance(decoded, str)

    def test_generate_step(self, backend):
        ids, _ = backend.tokenize("Hello")
        step = backend.generate_step(ids)
        assert step.next_token_id >= 0
        assert step.next_token_logits.shape[0] == backend.vocab_size
        assert len(step.attention_weights) == backend.num_layers

    def test_model_info_properties(self, backend):
        assert backend.num_layers > 0
        assert backend.num_attention_heads > 0
        assert backend.num_kv_heads > 0
        assert backend.vocab_size > 0
