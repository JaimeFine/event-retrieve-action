from __future__ import annotations

from pathlib import Path

import torch

from .external import build_external_era_agent


def _stack_rows(rows, device: str) -> torch.Tensor:
    if isinstance(rows, torch.Tensor):
        return rows.to(device)
    if not rows:
        return torch.empty((0, 0), device=device)
    stacked = [row.detach().view(-1).float() for row in rows]
    return torch.stack(stacked, dim=0).to(device)


def load_era_checkpoint(path: str | Path, device: str = "cpu"):
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported ERA checkpoint payload type: {type(payload)!r}")

    required = {"encoder", "Psi", "Gamma"}
    if not required.issubset(payload):
        raise ValueError(
            "Unsupported checkpoint format for load_era_checkpoint. "
            "Expected encoder/Psi/Gamma fields."
        )

    agent = build_external_era_agent(latent_dim=128).to(device)
    agent.encoder.load_state_dict(payload["encoder"])

    with torch.no_grad():
        agent.Psi.copy_(payload["Psi"].to(device))
        gamma = payload["Gamma"].to(device)
        if gamma.shape == agent.Gamma.shape:
            agent.Gamma.copy_(gamma)
        elif gamma.t().shape == agent.Gamma.shape:
            agent.Gamma.copy_(gamma.t())
        else:
            raise ValueError(
                f"Checkpoint Gamma shape mismatch: got {tuple(gamma.shape)}, "
                f"expected {tuple(agent.Gamma.shape)} or its transpose"
            )

    memory = payload.get("memory")
    if isinstance(memory, dict):
        latents = _stack_rows(memory.get("latents", []), device)
        actions = _stack_rows(memory.get("actions", []), device)
        rewards = memory.get("rewards", [])
        if isinstance(rewards, torch.Tensor):
            reliability = rewards.to(device).view(-1, 1).float()
        elif rewards:
            reliability = torch.stack(
                [reward.detach().view(-1).float().mean() for reward in rewards],
                dim=0,
            ).view(-1, 1).to(device)
        else:
            reliability = torch.ones((latents.shape[0], 1), device=device)
        reliability = reliability.clamp_min(1e-6)

        agent.memory.latents = latents
        agent.memory.actions = actions
        agent.memory.reliability = reliability

    agent.eval()
    return agent
