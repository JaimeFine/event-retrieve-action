from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class Experience:
    event_list: torch.Tensor
    action: torch.Tensor
    reward: Optional[torch.Tensor | float] = None
    next_event_list: Optional[torch.Tensor] = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    query_latent: torch.Tensor
    retrieved_actions: Optional[torch.Tensor] = None
    retrieved_latents: Optional[torch.Tensor] = None
    retrieved_next_latents: Optional[torch.Tensor] = None
    weights: Optional[torch.Tensor] = None
    distances: Optional[torch.Tensor] = None
    indices: Optional[torch.Tensor] = None
    stable_mask: Optional[torch.Tensor] = None
    selected_action: Optional[torch.Tensor] = None
    physics_penalty: Optional[torch.Tensor] = None


@dataclass
class EpisodeSummary:
    success: bool
    total_reward: float
    steps: int
    min_distance: float
    avg_latency_ms: float
    physics_violations: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
