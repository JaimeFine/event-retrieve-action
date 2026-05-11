from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .schemas import Experience


def ensure_tensor(value, dtype=torch.float32) -> torch.Tensor:
    if value is None:
        raise ValueError("value cannot be None")
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(dtype=dtype)
    return torch.tensor(value, dtype=dtype)


def ensure_event_tensor(value, feature_dim: int | None = None) -> torch.Tensor:
    tensor = ensure_tensor(value)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 2:
        raise ValueError(f"event_list must be 2D, got shape={tuple(tensor.shape)}")
    if feature_dim is not None and tensor.shape[-1] != feature_dim:
        raise ValueError(
            f"event feature dim mismatch: expected {feature_dim}, got {tensor.shape[-1]}"
        )
    return tensor


class ExperienceDataset(Dataset):
    def __init__(self, experiences: Iterable[Experience]):
        self.experiences = list(experiences)

    def __len__(self) -> int:
        return len(self.experiences)

    def __getitem__(self, idx: int) -> Experience:
        return self.experiences[idx]


def pad_event_lists(event_lists: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    padded = pad_sequence(event_lists, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([x.shape[0] for x in event_lists], dtype=torch.long)
    max_len = padded.shape[1]
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return padded, mask


def collate_experiences(batch: list[Experience]) -> dict[str, torch.Tensor | None]:
    events = [ensure_event_tensor(item.event_list) for item in batch]
    actions = torch.stack([ensure_tensor(item.action) for item in batch], dim=0)
    padded_events, event_mask = pad_event_lists(events)

    next_events = None
    next_mask = None
    if all(item.next_event_list is not None for item in batch):
        next_events_raw = [ensure_event_tensor(item.next_event_list) for item in batch]
        next_events, next_mask = pad_event_lists(next_events_raw)

    rewards = []
    for item in batch:
        reward = 0.0 if item.reward is None else item.reward
        rewards.append(float(reward) if not isinstance(reward, torch.Tensor) else float(reward.item()))

    return {
        "events": padded_events,
        "event_mask": event_mask,
        "actions": actions,
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "next_events": next_events,
        "next_mask": next_mask,
    }


def generate_synthetic_next_event_list(
    event_list: torch.Tensor,
    action: torch.Tensor,
    dt: float = 0.05,
) -> torch.Tensor:
    event_list = ensure_event_tensor(event_list)
    action = ensure_tensor(action)
    next_event = event_list.clone()

    if next_event.shape[-1] >= 7:
        next_event[:, 1:4] = event_list[:, 1:4] - (action[:3] - event_list[:, 4:7]) * dt
    if next_event.shape[-1] >= 10:
        next_event[:, 7:10] = action[:3]
    return next_event


def generate_synthetic_next_event_batch(
    padded_events: torch.Tensor,
    actions: torch.Tensor,
    mask: torch.Tensor,
    dt: float = 0.05,
) -> torch.Tensor:
    next_events = padded_events.clone()
    if padded_events.shape[-1] >= 7:
        next_events[:, :, 1:4] = padded_events[:, :, 1:4] - (
            actions[:, None, :3] - padded_events[:, :, 4:7]
        ) * dt
    if padded_events.shape[-1] >= 10:
        next_events[:, :, 7:10] = actions[:, None, :3]
    next_events[~mask] = 0.0
    return next_events


def serialize_experiences(experiences: list[Experience]) -> list[dict]:
    payload = []
    for item in experiences:
        payload.append(
            {
                "event_list": item.event_list.cpu(),
                "action": item.action.cpu(),
                "reward": item.reward,
                "next_event_list": None
                if item.next_event_list is None
                else item.next_event_list.cpu(),
                "meta": item.meta,
            }
        )
    return payload


def save_experiences(path: str | Path, experiences: list[Experience]) -> None:
    torch.save(serialize_experiences(experiences), path)


def load_experiences(path: str | Path) -> list[Experience]:
    raw = torch.load(path, map_location="cpu")
    experiences: list[Experience] = []

    if not isinstance(raw, list):
        raise ValueError(f"Unsupported dataset payload type: {type(raw)!r}")

    for item in raw:
        if isinstance(item, Experience):
            experiences.append(item)
            continue

        if isinstance(item, dict):
            experiences.append(
                Experience(
                    event_list=ensure_event_tensor(item["event_list"]),
                    action=ensure_tensor(item["action"]),
                    reward=item.get("reward"),
                    next_event_list=None
                    if item.get("next_event_list") is None
                    else ensure_event_tensor(item["next_event_list"]),
                    meta=item.get("meta", {}),
                )
            )
            continue

        if isinstance(item, (tuple, list)) and len(item) >= 2:
            experiences.append(
                Experience(
                    event_list=ensure_event_tensor(item[0]),
                    action=ensure_tensor(item[1]),
                    reward=item[2] if len(item) > 2 else None,
                    next_event_list=None if len(item) <= 3 or item[3] is None else ensure_event_tensor(item[3]),
                )
            )
            continue

        raise ValueError(f"Unsupported experience item: {type(item)!r}")

    return experiences
