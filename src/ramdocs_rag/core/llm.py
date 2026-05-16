"""Thin wrapper around the OpenAI SDK plus a deterministic ``MockLLM`` for tests.

Design:
- Every pipeline accepts an ``LLMClient`` through dependency injection.
  Tests swap in a ``MockLLM`` — no real API calls.
- ``OpenAIClient`` computes cost from a manually-maintained pricing table.
- All calls are strict-JSON (via ``response_format=json_schema``), so
  pipelines can rely on the parsed payload conforming to the schema.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class LLMCallResult:
    """What an ``LLMClient`` returns: parsed JSON plus operational metrics."""

    parsed: dict
    raw_text: str
    cost_usd: float
    tokens_in: int
    tokens_out: int
    model: str


class LLMClient(Protocol):
    """Minimal LLM-client contract — everything a pipeline needs."""

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        schema_name: str,
        temperature: float = 0.0,
    ) -> LLMCallResult: ...


# ---------- MockLLM for tests ----------


def _prompt_hash(system: str, user: str) -> str:
    h = hashlib.sha256()
    h.update(system.encode("utf-8"))
    h.update(b"\x00")
    h.update(user.encode("utf-8"))
    return h.hexdigest()[:16]


@dataclass
class MockLLM:
    """Deterministic mock that returns pre-scripted JSON responses.

    Usage::

        mock = MockLLM(
            responses={"<hash>": {...}},
            default={"stance": "no_answer", ...},
        )

    If ``responses[hash(system, user)]`` is present it is returned;
    otherwise the call falls back to ``default``. Every call increments
    ``.calls`` and the (system, user) pair is appended to ``.last_prompts``.
    """

    responses: dict[str, dict] = field(default_factory=dict)
    default: dict | None = None
    model: str = "mock-llm"
    calls: int = 0
    last_prompts: list[tuple[str, str]] = field(default_factory=list)

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        schema_name: str,
        temperature: float = 0.0,
    ) -> LLMCallResult:
        self.calls += 1
        self.last_prompts.append((system, user))
        key = _prompt_hash(system, user)
        parsed: dict | None = self.responses.get(key)
        if parsed is None:
            if self.default is None:
                raise KeyError(
                    f"MockLLM has no scripted response for prompt hash {key} "
                    f"and no default configured. schema_name={schema_name!r}"
                )
            parsed = self.default
        raw_text = json.dumps(parsed, ensure_ascii=False)
        return LLMCallResult(
            parsed=parsed,
            raw_text=raw_text,
            cost_usd=0.0,
            tokens_in=len(system) + len(user),
            tokens_out=len(raw_text),
            model=self.model,
        )

    def script(self, system: str, user: str, response: dict) -> None:
        """Record a scripted response for an exact ``(system, user)`` pair."""
        self.responses[_prompt_hash(system, user)] = response


# ---------- OpenAI client ----------

# Pricing per 1M tokens, current as of 2026-05. Update by hand when OpenAI
# rev the rates; cost-tracking is best-effort, not used for routing.
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    p_in, p_out = _PRICING.get(model, (0.0, 0.0))
    return (tokens_in * p_in + tokens_out * p_out) / 1_000_000


@dataclass
class OpenAIClient:
    """Thin wrapper around the OpenAI SDK.

    The SDK is imported lazily so that unit tests can run without the
    ``openai`` package installed. When ``LANGFUSE_PUBLIC_KEY`` and
    ``LANGFUSE_SECRET_KEY`` are present in the environment the client
    transparently switches to ``langfuse.openai.OpenAI`` — same surface
    API, but every ``chat.completions.create`` call is auto-recorded as
    a Langfuse trace.
    """

    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None

    def __post_init__(self) -> None:
        from openai import OpenAI  # noqa: F401  — fail-fast presence check
        self._client = None  # actual instantiation happens in _ensure_client

    def _ensure_client(self) -> Any:
        if self._client is None:
            # Langfuse drop-in: same surface as ``openai.OpenAI``, but every
            # ``chat.completions.create`` call is auto-recorded as a Langfuse
            # trace. Enabled only when both Langfuse keys are present in env.
            import os

            if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get(
                "LANGFUSE_SECRET_KEY"
            ):
                from langfuse.openai import OpenAI
            else:
                from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        schema_name: str,
        temperature: float = 0.0,
    ) -> LLMCallResult:
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            },
        )
        raw_text = resp.choices[0].message.content or "{}"
        parsed = json.loads(raw_text)
        usage = resp.usage
        tokens_in = getattr(usage, "prompt_tokens", 0) if usage else 0
        tokens_out = getattr(usage, "completion_tokens", 0) if usage else 0
        return LLMCallResult(
            parsed=parsed,
            raw_text=raw_text,
            cost_usd=_estimate_cost(self.model, tokens_in, tokens_out),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=self.model,
        )
