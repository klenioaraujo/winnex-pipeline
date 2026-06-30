"""
Competition Manager — Multi-Competition Registry
==================================================
Central registry for all competition agents.
Each competition has a config, an agent class, and a kernel generator.

Usage:
    from winnex_pipeline.compete import CompetitionManager

    cm = CompetitionManager()
    cm.list_competitions()
    agent = cm.get_agent("ai-agent-security-multi-step-tool-attacks")
    findings = agent.run(env, config)
    cm.generate_kernel("ai-agent-security-multi-step-tool-attacks")
"""
from __future__ import annotations
import os, json, importlib, inspect
from collections.abc import Mapping
from typing import Any
from pathlib import Path

from .base import BaseCompetitionAgent, CompetitionConfig

_COMPETITIONS: dict[str, dict[str, Any]] = {}
_COMPETITION_DIR = Path(__file__).parent / "configs"
_AGENTS_DIR = Path(__file__).parent / "agents"


def _discover_competitions():
    """Scan configs/ and agents/ for available competitions."""
    global _COMPETITIONS

    # Scan configs/*.json
    if _COMPETITION_DIR.exists():
        for f in sorted(_COMPETITION_DIR.glob("*.json")):
            cid = f.stem
            if cid not in _COMPETITIONS:
                with open(f) as fh:
                    data = json.load(fh)
                _COMPETITIONS[cid] = {
                    "id": cid,
                    "config_path": str(f),
                    "config": data,
                    "name": data.get("name", cid),
                    "description": data.get("description", ""),
                }

    # Scan agents/*.py
    if _AGENTS_DIR.exists():
        for f in sorted(_AGENTS_DIR.glob("*.py")):
            if f.stem == "__init__":
                continue
            cid = f.stem
            if cid not in _COMPETITIONS:
                _COMPETITIONS[cid] = {
                    "id": cid,
                    "agent_path": str(f),
                    "name": cid,
                    "description": f"Agent module: {cid}",
                }


# Initialize on import
_discover_competitions()


class CompetitionManager:
    """
    Central manager for all competition agents.

    Provides discovery, loading, kernel generation, and submission.
    """

    def __init__(self):
        self._agents: dict[str, BaseCompetitionAgent] = {}

    def list_competitions(self) -> list[dict[str, Any]]:
        """List all registered competitions."""
        _discover_competitions()
        return [v for v in _COMPETITIONS.values()]

    def get_agent(self, competition_id: str, config: Mapping[str, Any] | None = None) -> BaseCompetitionAgent:
        """
        Get a competition agent by competition ID.

        Args:
            competition_id: e.g. "ai-agent-security-multi-step-tool-attacks"
            config: Optional config overrides

        Returns:
            BaseCompetitionAgent instance
        """
        if competition_id not in _COMPETITIONS:
            raise KeyError(f"Unknown competition: {competition_id}. Available: {list(_COMPETITIONS.keys())}")

        comp = _COMPETITIONS[competition_id]

        # Try importing the agent module
        module_name = f"winnex_pipeline.compete.agents.{competition_id.replace('-', '_')}"
        try:
            mod = importlib.import_module(module_name)
        except ImportError:
            raise ImportError(
                f"Agent module not found: {module_name}. "
                f"Expected at: {_AGENTS_DIR / competition_id.replace('-', '_')}.py"
            )

        # Find the AttackAlgorithm class
        for name, obj in inspect.getmembers(mod):
            if inspect.isclass(obj) and issubclass(obj, BaseCompetitionAgent) and name != "BaseCompetitionAgent":
                return obj(config)

        raise RuntimeError(f"No BaseCompetitionAgent subclass found in {module_name}")

    def load(self, competition_id: str, env: Any, config: Any) -> list[Any]:
        """Load and run a competition agent."""
        agent = self.get_agent(competition_id)
        return agent.run(env, config)

    def run(self, env: Any, config: Any) -> list[Any]:
        """Alias for load()."""
        return self.load("", env, config)

    def generate_kernel(self, competition_id: str, output_dir: str | None = None) -> str:
        """Generate a Kaggle kernel for a competition."""
        from .lib.submission import build_kaggle_kernel

        if competition_id not in _COMPETITIONS:
            raise KeyError(f"Unknown competition: {competition_id}")

        comp = _COMPETITIONS[competition_id]
        output_dir = output_dir or str(Path(__file__).parent / "kernels")

        # Read agent source code
        agent_path = _AGENTS_DIR / f"{competition_id.replace('-', '_')}.py"
        if not agent_path.exists():
            agent_path = _AGENTS_DIR / f"{competition_id}.py"
        if not agent_path.exists():
            raise FileNotFoundError(f"Agent file not found: {agent_path}")

        with open(agent_path) as f:
            agent_code = f.read()

        return build_kaggle_kernel(
            competition_id=competition_id,
            agent_code=agent_code,
            config=comp.get("config"),
            output_dir=str(Path(__file__).parent / output_dir) if output_dir else output_dir,
        )
