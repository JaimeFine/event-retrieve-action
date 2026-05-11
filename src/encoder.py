from __future__ import annotations

import torch
import torch.nn as nn

try:
    from bruce_code.config import ModelConfig
except ImportError:
    from config import ModelConfig


class EventEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.phi = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )
        self.rho = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.ReLU(),
            nn.Linear(config.latent_dim, config.latent_dim),
        )

    def forward(
        self,
        event_list: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        single = event_list.dim() == 2
        if single:
            event_list = event_list.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        if event_list.dim() != 3:
            raise ValueError(f"Expected 2D or 3D tensor, got {tuple(event_list.shape)}")

        batch_size = event_list.shape[0]
        if event_list.shape[1] == 0:
            zeros = torch.zeros(
                batch_size,
                self.config.latent_dim,
                dtype=event_list.dtype,
                device=event_list.device,
            )
            return zeros.squeeze(0) if single else zeros

        x = self.phi(event_list)

        if mask is None:
            mask = torch.ones(
                event_list.shape[:2], dtype=torch.bool, device=event_list.device
            )
        weights = mask.unsqueeze(-1).float()

        if self.config.use_inverse_norm_weights:
            weights = weights / (torch.norm(x, dim=-1, keepdim=True) + 1e-6)

        pooled = torch.sum(x * weights, dim=1)
        denom = torch.sum(weights, dim=1).clamp_min(1e-6)
        pooled = pooled / denom
        latent = self.rho(pooled)
        return latent.squeeze(0) if single else latent


__all__ = ["EventEncoder"]
