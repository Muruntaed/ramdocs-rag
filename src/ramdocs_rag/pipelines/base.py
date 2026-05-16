"""Pipeline contract. Every version (v1, v2, ...) implements this.

Design:
- ``Pipeline`` is an ABC, not a Protocol, so common helpers can live here.
- ``run`` must return a ``RunResult`` within a reasonable time; all
  IO plumbing (trace dumping, latency timing, exception handling) is
  the runner's job (see ``eval.runner``).
- The LLM client is injected through the constructor — that's the
  hook for mocking in tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.llm import LLMClient
from ..core.types import Question, RunResult


class Pipeline(ABC):
    """Base pipeline class. Concrete versions live in ``pipelines/vN_M_*/``."""

    #: Fully-qualified version name, e.g. ``"v1.0_madam_lite"``.
    name: str

    #: Free-form semver, e.g. ``"1.0.0"``. Bump manually before a freeze run.
    version: str

    def __init__(self, llm: LLMClient, config: dict | None = None) -> None:
        self.llm = llm
        self.config: dict = config or {}

    @abstractmethod
    def run(self, question: Question) -> RunResult:
        """Run one question through the pipeline and return a structured result.

        The implementation **must not** catch generic exceptions — that is
        the runner's responsibility. Set ``error`` only for domain-level
        failures (LLM returned invalid JSON, no retrieval candidates, ...).
        """

    def describe(self) -> dict:
        """Snapshot for ``config.yaml`` inside the run artefacts."""
        return {
            "name": self.name,
            "version": self.version,
            "config": self.config,
            "llm_model": getattr(self.llm, "model", "unknown"),
        }
