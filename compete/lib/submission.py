"""
Submission — Build Competition Submissions and Kernels
=======================================================
Generates Kaggle kernels and submission files from compete agents.
Each competition gets a self-contained notebook kernel.
"""
from __future__ import annotations
import os, json, textwrap
from typing import Any


def build_submission(candidates: list[Any], competition_id: str = "") -> list[Any]:
    """
    Build final submission from attack candidates.

    Deduplicates, sorts by score, and caps at 800 candidates.
    """
    seen = set()
    unique = []
    for c in candidates:
        msgs = tuple(c.user_messages) if hasattr(c, 'user_messages') else tuple(c)
        if msgs not in seen:
            seen.add(msgs)
            unique.append(c)
    return unique[:800]


def build_kaggle_kernel(
    competition_id: str,
    agent_code: str,
    config: dict[str, Any] | None = None,
    output_dir: str = "kernels",
) -> str:
    """
    Generate a self-contained Kaggle notebook kernel.

    Args:
        competition_id: Kaggle competition ID
        agent_code: Python code for the AttackAlgorithm class
        config: Competition configuration dict
        output_dir: Output directory for the kernel

    Returns:
        Path to the generated kernel file.
    """
    cfg = config or {}
    kernel_dir = os.path.join(output_dir, competition_id)
    os.makedirs(kernel_dir, exist_ok=True)

    # Build the notebook
    notebook = _build_notebook(competition_id, agent_code, cfg)

    kernel_path = os.path.join(kernel_dir, f"{competition_id.replace('-','_')}_kernel.ipynb")
    with open(kernel_path, "w") as f:
        json.dump(notebook, f, indent=1)

    # Build kernel-metadata.json
    metadata = {
        "id": f"kleniopadilha/{competition_id}-winnex",
        "title": f"Winnex {competition_id}",
        "code_file": f"{competition_id.replace('-','_')}_kernel.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": True,
        "enable_internet": True,
    }
    meta_path = os.path.join(kernel_dir, "kernel-metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=1)

    return kernel_path


def _build_notebook(competition_id: str, agent_code: str, config: dict[str, Any]) -> dict:
    """Build a Jupyter notebook with the agent code."""
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# Winnex AI — {competition_id}\n",
                    "\n",
                    "Auto-generated competition kernel using Winnex pipeline.\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "import sys, os, json, itertools, logging, math, random, re, time, hashlib, base64\n",
                    "from pathlib import Path\n",
                    "from typing import Any\n",
                    "\n",
                    "# SDK path resolution\n",
                    "def _add_sdk_root():\n",
                    "    h = Path(__file__).resolve().parent\n",
                    "    for r in [h/'data', h, Path('/kaggle/input'), Path('/mnt/data')]:\n",
                    "        if r.exists() and (r/'aicomp_sdk').exists() and (r/'kaggle_evaluation').exists():\n",
                    "            if str(r) not in sys.path: sys.path.insert(0, str(r)); return\n",
                    "        for c in r.rglob('kaggle_evaluation'):\n",
                    "            p2 = c.parent\n",
                    "            if (p2/'aicomp_sdk').exists():\n",
                    "                if str(p2) not in sys.path: sys.path.insert(0, str(p2)); return\n",
                    "    for c in glob.glob('/kaggle/input/**/kaggle_evaluation', recursive=True):\n",
                    "        p2 = str(Path(c).parent)\n",
                    "        if p2 not in sys.path: sys.path.insert(0, p2); return\n",
                    "_add_sdk_root()\n",
                    "from aicomp_sdk.attacks import AttackAlgorithmBase, AttackCandidate\n",
                    "from aicomp_sdk.core.predicates import eval_predicates\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    f"# ── Agent Configuration ──\n",
                    f"CONFIG = {json.dumps(config, indent=2)}\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# ── Agent Implementation ──\n",
                    agent_code,
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# ── Inference Server ──\n",
                    "if __name__ == '__main__' or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:\n",
                    "    from aicomp_sdk.agents import build_agent\n",
                    "    from aicomp_sdk.attacks.baselines.guardrail_rules import Guardrail\n",
                    "    from aicomp_sdk.core.env.sandbox import SandboxEnv\n",
                    "    \n",
                    "    env = SandboxEnv(\n",
                    "        seed=42,\n",
                    "        fixtures_dir=Path('/kaggle/input/fixtures'),\n",
                    "        agent=build_agent('deterministic'),\n",
                    "        guardrail=Guardrail(),\n",
                    "    )\n",
                    "    atk = AttackAlgorithm(CONFIG)\n",
                    "    findings = atk.run(env, None)\n",
                    "    print(f'Findings: {{len(findings)}}')\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
