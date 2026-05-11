from __future__ import annotations

import sys
from functools import lru_cache
from importlib import import_module
from pathlib import Path


def get_external_era_root() -> Path:
    root = Path(__file__).resolve().parents[2] / "external" / "event-retrieve-action-main"
    if not root.exists():
        raise FileNotFoundError(
            "External ERA repository not found. Expected: "
            f"{root}. Please clone http://github.com/JaimeFine/event-retrieve-action first."
        )
    return root


def ensure_external_era_on_path() -> Path:
    root = get_external_era_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


@lru_cache(maxsize=1)
def load_external_module(module_name: str):
    ensure_external_era_on_path()
    return import_module(module_name)


def load_external_agent_class():
    return load_external_module("agents.agent").EventCentricAgent


def load_external_encoder_class():
    return load_external_module("agents.encoder").EventEncoder


def load_external_memory_class():
    return load_external_module("agents.bank").KnowledgeBank


def load_external_stabilizer_class():
    return load_external_module("agents.stabilizer").LyapunovStabilizer


def load_external_environment_class():
    return load_external_module("trainer").Environment


def build_external_era_agent(latent_dim: int = 128):
    agent_cls = load_external_agent_class()
    return agent_cls(latent_dim=latent_dim)
