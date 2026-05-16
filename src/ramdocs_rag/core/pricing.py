"""OpenAI per-model pricing table for cost accounting.

Cost figures are per 1M tokens (USD). Update the constants here when
OpenAI revises the rates — kept in a dedicated module so the SDK wrapper
in ``core/llm.py`` does not need to be touched.

``PRICING`` maps a model identifier to ``(input_price_per_1m, output_price_per_1m)``.
Models not present in the table default to ``(0.0, 0.0)`` — costs simply
will not be tracked for them.
"""

from __future__ import annotations

PRICING: dict[str, tuple[float, float]] = {
    # OpenAI rates as of mid-2026. Update when OpenAI revises the prices.
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Estimate the USD cost of one chat completion call."""
    p_in, p_out = PRICING.get(model, (0.0, 0.0))
    return (tokens_in * p_in + tokens_out * p_out) / 1_000_000
