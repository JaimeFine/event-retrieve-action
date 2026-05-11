from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_event_mean(events: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).float()
    summed = torch.sum(events * weights, dim=1)
    denom = torch.sum(weights, dim=1).clamp_min(1e-6)
    return summed / denom


def metric_consistency_loss(
    latents: torch.Tensor,
    events: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if latents.shape[0] <= 1:
        return torch.zeros((), device=latents.device)
    shifted_latents = torch.roll(latents, shifts=1, dims=0)
    event_mean = masked_event_mean(events, mask)
    shifted_event_mean = torch.roll(event_mean, shifts=1, dims=0)
    d_latent = torch.norm(latents - shifted_latents, dim=1)
    d_phys = torch.norm(event_mean - shifted_event_mean, dim=1)
    return torch.mean(torch.abs(d_latent - d_phys))


def physics_consistency_regularizer(
    query_latent: torch.Tensor,
    retrieved_latents: torch.Tensor,
    weights: torch.Tensor,
    predicted_next_latents: torch.Tensor | None = None,
    retrieved_next_latents: torch.Tensor | None = None,
) -> torch.Tensor:
    weights = weights / weights.sum().clamp_min(1e-6)
    current_term = torch.sum(weights * torch.norm(query_latent.view(1, -1) - retrieved_latents, dim=1))
    if predicted_next_latents is None or retrieved_next_latents is None:
        return current_term
    next_term = torch.sum(weights * torch.norm(predicted_next_latents - retrieved_next_latents, dim=1))
    return current_term + next_term


def imitation_loss(pred_actions: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_actions, target_actions)
