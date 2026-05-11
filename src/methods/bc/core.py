from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bruce_code.config import ModelConfig, TrainConfig
from bruce_code.dataset import ExperienceDataset, collate_experiences
from bruce_code.encoder import EventEncoder
from bruce_code.losses import imitation_loss


def _goal_progress_regularizer(
    pred_actions: torch.Tensor,
    events: torch.Tensor,
    speed_floor: float = 0.35,
    progress_floor: float = 0.25,
) -> torch.Tensor:
    if events.dim() != 3 or events.shape[-1] < 13:
        return torch.zeros((), device=pred_actions.device)

    rel_goal = events[:, 0, 10:13]
    goal_norm = torch.norm(rel_goal, dim=1, keepdim=True).clamp_min(1e-6)
    goal_dir = rel_goal / goal_norm

    action_speed = torch.norm(pred_actions, dim=1)
    speed_penalty = F.relu(speed_floor - action_speed).pow(2)

    forward_progress = torch.sum(pred_actions * goal_dir, dim=1)
    progress_penalty = F.relu(progress_floor - forward_progress).pow(2)

    return torch.mean(speed_penalty + progress_penalty)


class BehavioralCloningPolicy(nn.Module):
    def __init__(self, model_config: ModelConfig | None = None, device: str = "cpu"):
        super().__init__()
        self.model_config = model_config or ModelConfig()
        self.device = torch.device(device)
        self.encoder = EventEncoder(self.model_config)
        self.head = nn.Sequential(
            nn.Linear(self.model_config.latent_dim, self.model_config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.model_config.hidden_dim, self.model_config.action_dim),
        )
        self.to(self.device)

    def forward(self, events: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        latent = self.encoder(events, mask=mask)
        return self.head(latent)

    @torch.no_grad()
    def predict(self, event_list: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        event_list = event_list.to(self.device)
        mask = None if mask is None else mask.to(self.device)
        action = self.forward(event_list, mask=mask)
        return action.view(-1) if action.dim() > 1 else action


def train_behavioral_cloning(
    model: BehavioralCloningPolicy,
    experiences,
    train_config: TrainConfig,
) -> BehavioralCloningPolicy:
    dataset = ExperienceDataset(experiences)
    loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=collate_experiences,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)

    imitation_weight = 1.0
    progress_reg_weight = 0.12

    for _ in range(train_config.epochs):
        model.train()
        for batch in loader:
            events = batch["events"].to(model.device)
            mask = batch["event_mask"].to(model.device)
            actions = batch["actions"].to(model.device)
            pred = model(events, mask=mask)
            loss_imitation = imitation_loss(pred, actions)
            loss_progress = _goal_progress_regularizer(pred, events)
            loss = imitation_weight * loss_imitation + progress_reg_weight * loss_progress
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def save_behavioral_cloning_policy(model: BehavioralCloningPolicy, path: str | Path) -> None:
    payload = {
        "model_config": model.model_config.to_dict(),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)


def load_behavioral_cloning_policy(path: str | Path, device: str = "cpu") -> BehavioralCloningPolicy:
    payload = torch.load(path, map_location=device)
    model_cfg = ModelConfig(**payload.get("model_config", {}))
    model = BehavioralCloningPolicy(model_config=model_cfg, device=device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
