from __future__ import annotations

from typing import Protocol

import torch


class PolicyAdapter(Protocol):
    name: str

    def reset(self) -> None: ...

    def act(self, event_list: torch.Tensor | None) -> torch.Tensor: ...


class GenericPolicyAdapter:
    def __init__(self, name: str, policy, device_name: str, metadata: dict | None = None):
        self.name = name
        self._policy = policy
        self._device = torch.device(device_name)
        self.metadata = dict(metadata or {})

    def reset(self) -> None:

        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def act(self, event_list: torch.Tensor | None) -> torch.Tensor:
        if event_list is None or event_list.numel() == 0:
            return torch.zeros(3, device=self._device)

        if hasattr(self._policy, "predict"):
            action = self._policy.predict(event_list)
        elif hasattr(self._policy, "act"):
            action = self._policy.act(event_list)
        elif callable(self._policy):
            action = self._policy(event_list)
        else:
            raise TypeError(f"Unsupported policy type: {type(self._policy)!r}")

        if isinstance(action, tuple):
            action = action[0]
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self._device)
        return action.detach().to(self._device).view(-1)
