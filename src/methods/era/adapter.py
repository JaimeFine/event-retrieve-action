from __future__ import annotations

import torch


class EraAdapter:
    def __init__(self, selector_fn, device_name: str):
        self.name = "era"
        self._selector_fn = selector_fn
        self._device = torch.device(device_name)

    def reset(self) -> None:
        return

    def act(self, event_list: torch.Tensor | None) -> torch.Tensor:
        if event_list is None or event_list.numel() == 0:
            return torch.zeros(3, device=self._device)
        output = self._selector_fn(event_list, k=5)
        action = output[0] if isinstance(output, tuple) else output
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self._device)
        return action.detach().to(self._device).view(-1)
