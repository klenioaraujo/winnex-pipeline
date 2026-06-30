"""
Archive — Exploration Archive for Competition Agents
======================================================
Cell-based archive for Go-Explore style search.
Tracks visited states, scores, and exploration statistics.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import random


@dataclass
class Exemplar:
    """A frontier cell in the exploration archive."""
    snapshot: dict[str, Any]          # env snapshot for restore
    user_messages: list[str]          # message chain
    cell_hash: str                    # unique cell signature
    score_hint: float = 0.0
    visits: int = 0
    depth: int = 0
    predicates: list[dict[str, Any]] = field(default_factory=list)
    real_attacks: list[dict[str, Any]] = field(default_factory=list)
    family: int = 0                   # anchor family (HMC navigation)
    tool_signature: str = ""          # tool pattern for similarity


class Archive:
    """
    Exploration archive with weighted cell selection.

    Supports:
      - HMC-weighted selection (via family scores)
      - Novelty-weighted selection (visit count)
      - Score-weighted selection (predicate rewards)
    """

    def __init__(self, selection_mode: str = "weighted"):
        self._cells: dict[str, Exemplar] = {}
        self.selection_mode = selection_mode

    def __len__(self) -> int:
        return len(self._cells)

    def __contains__(self, h: str) -> bool:
        return h in self._cells

    def __getitem__(self, h: str) -> Exemplar:
        return self._cells[h]

    def items(self):
        return self._cells.items()

    def values(self):
        return self._cells.values()

    def add(self, exemplar: Exemplar) -> bool:
        """Add cell to archive. Returns True if novel (first time)."""
        h = exemplar.cell_hash
        if h not in self._cells:
            self._cells[h] = exemplar
            return True
        return False

    def select(self, rng: random.Random) -> Exemplar:
        """
        Select a cell from the archive using weighted probability.

        Weight = visit_penalty * score * depth_bonus
        """
        candidates = list(self._cells.values())
        if not candidates:
            raise RuntimeError("Empty archive")

        max_visits = max(e.visits for e in candidates) + 1
        max_score = max(e.score_hint for e in candidates) + 1

        weights = []
        for ex in candidates:
            visit_w = (max_visits - ex.visits) / max_visits
            score_w = (ex.score_hint + 1) / max_score
            depth_w = 1.0 / (1.0 + abs(ex.depth - 3))
            weights.append(visit_w * 2.0 + score_w * 1.5 + depth_w * 0.5)

        total = sum(weights)
        r = rng.uniform(0, total)
        cum = 0.0
        for ex, w in zip(candidates, weights):
            cum += w
            if r <= cum:
                return ex
        return candidates[-1]

    def stats(self) -> dict[str, Any]:
        """Return archive statistics."""
        if not self._cells:
            return {"size": 0}
        depths = [e.depth for e in self._cells.values()]
        scores = [e.score_hint for e in self._cells.values()]
        families = {}
        for e in self._cells.values():
            families[e.family] = families.get(e.family, 0) + 1
        return {
            "size": len(self._cells),
            "depth_mean": sum(depths) / len(depths),
            "depth_max": max(depths),
            "score_max": max(scores),
            "families": families,
        }
