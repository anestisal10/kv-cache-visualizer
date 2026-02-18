"""
Eviction Policy Registry — import all policies and provide a lookup by name.
"""

from .base import EvictionPolicy
from .streaming_llm import StreamingLLMPolicy
from .h2o import H2OPolicy
from .window_only import WindowOnlyPolicy
from .random_evict import RandomEvictionPolicy
from .no_eviction import NoEvictionPolicy

# ── Registry ────────────────────────────────────────────────────────
POLICY_REGISTRY: dict[str, type[EvictionPolicy]] = {
    "streaming_llm": StreamingLLMPolicy,
    "h2o": H2OPolicy,
    "window_only": WindowOnlyPolicy,
    "random": RandomEvictionPolicy,
    "no_eviction": NoEvictionPolicy,
}


def get_policy(name: str, **kwargs) -> EvictionPolicy:
    """
    Instantiate an eviction policy by name.

    Args:
        name: One of 'streaming_llm', 'h2o', 'window_only', 'random', 'no_eviction'.
        **kwargs: Policy-specific parameters (e.g., n_sink, window_size).

    Returns:
        Instantiated EvictionPolicy.

    Raises:
        KeyError: If the policy name is not registered.
    """
    if name not in POLICY_REGISTRY:
        available = ", ".join(POLICY_REGISTRY.keys())
        raise KeyError(f"Unknown policy '{name}'. Available: {available}")
    return POLICY_REGISTRY[name](**kwargs)


def list_policies() -> list[str]:
    """Return list of registered policy names."""
    return list(POLICY_REGISTRY.keys())


__all__ = [
    "EvictionPolicy",
    "StreamingLLMPolicy",
    "H2OPolicy",
    "WindowOnlyPolicy",
    "RandomEvictionPolicy",
    "NoEvictionPolicy",
    "get_policy",
    "list_policies",
    "POLICY_REGISTRY",
]
