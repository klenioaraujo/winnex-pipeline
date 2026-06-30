"""
Winnex Compete — Multi-Competition Agent Framework
====================================================
Leverages the Winnex stack (MadhavaCore, HMC, PiPrimeAnchors, QJL)
to compete in Kaggle AI competitions.

Architecture:
  compete/
  ├── __init__.py           # Module loader + registry
  ├── manager.py            # CompetitionManager (central registry)
  ├── base.py               # BaseCompetitionAgent (abstract)
  ├── configs/              # Competition-specific configs
  │   └── {competition_id}.json
  ├── agents/               # Competition-specific agents
  │   ├── __init__.py
  │   └── {competition_id}.py
  ├── kernels/              # Generated Kaggle notebook kernels
  │   └── {competition_id}/  # Auto-generated from templates
  └── lib/                  # Shared competition utilities
      ├── __init__.py
      ├── archive.py        # Archive/exploration strategies
      ├── mutation.py       # HMC navigation + prompt mutation
      └── submission.py     # Kernel and submission builder

Usage:
    from winnex_pipeline.compete import CompetitionManager

    cm = CompetitionManager()
    print(cm.list_competitions())
    agent = cm.get_agent("ai-agent-security-multi-step-tool-attacks")
    kernel_path = cm.generate_kernel("ai-agent-security-multi-step-tool-attacks")
"""

from .manager import CompetitionManager
from .base import BaseCompetitionAgent, CompetitionConfig

__all__ = [
    "CompetitionManager",
    "BaseCompetitionAgent",
    "CompetitionConfig",
]
