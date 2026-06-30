"""
Competition Base Agent — Abstract Interface
=============================================
Defines the abstract base for all competition agents.
Every competition implements run(env, config) -> list[AttackCandidate].

The Winnex stack (HMC navigation, MadhavaCore, PiPrimeAnchors) is
used as the core search/intelligence layer behind each agent.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any


class CompetitionConfig:
    """Runtime configuration for a competition run."""

    def __init__(self, config: Mapping[str, Any] | None = None):
        self.raw = dict(config or {})
        self.time_budget_s: float = float(self.raw.get("time_budget_s", 120.0))
        self.max_steps: int = int(self.raw.get("max_steps", 10**9))
        self.max_tool_hops: int = int(self.raw.get("max_tool_hops", 10))
        self.seed: int = int(self.raw.get("seed", 42))
        # Winnex stack overrides
        self.search_method: str = str(self.raw.get("search_method", "hmc"))
        self.n_anchors: int = int(self.raw.get("n_anchors", 30))
        self.prompt_bank_size: int = int(self.raw.get("prompt_bank_size", 200))
        self.branch_batch: int = int(self.raw.get("branch_batch", 12))
        self.max_turns: int = int(self.raw.get("max_turns", 20))

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


class BaseCompetitionAgent(ABC):
    """
    Abstract base for competition agents.

    Subclasses implement:
      run(env, config) -> list[AttackCandidate]
      name -> str
    """

    def __init__(self, config: Mapping[str, Any] | None = None):
        self.cfg = CompetitionConfig(config)

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable agent name."""
        ...

    @abstractmethod
    def run(self, env: Any, config: Any) -> list[Any]:
        """Execute the competition agent. Returns list of candidates."""
        ...

    def build_prompt_bank(self, prompt_bank: list[str] | None = None) -> list[str]:
        """Build or augment prompt bank for this competition."""
        return prompt_bank or []
