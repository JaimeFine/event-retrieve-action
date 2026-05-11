from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from .external import build_external_era_agent
from .training import load_era_checkpoint


def _apply_legacy_era_artifacts_to_sim(sim, checkpoint: dict, bank: dict, device_name: str) -> None:
    device = torch.device(device_name)

    sim.agent.encoder.load_state_dict(checkpoint["encoder"])
    sim.agent.Psi.data.copy_(checkpoint["Psi"].to(device))

    gamma = checkpoint["Gamma"].to(device)
    if gamma.shape == sim.agent.Gamma.shape:
        sim.agent.Gamma.data.copy_(gamma)
    elif gamma.t().shape == sim.agent.Gamma.shape:
        sim.agent.Gamma.data.copy_(gamma.t())
    else:
        raise ValueError(
            f"Legacy Gamma shape mismatch: {tuple(gamma.shape)} vs expected {tuple(sim.agent.Gamma.shape)} or transpose"
        )

    latents = bank["latents"]
    actions = bank["actions"]
    reliability = bank.get("reliability")

    memory = sim.agent.memory
    memory.latents = latents.to(device)
    memory.actions = actions.to(device)
    if reliability is None:
        memory.reliability = torch.ones((latents.shape[0], 1), device=device)
    else:
        memory.reliability = reliability.to(device).view(-1, 1)


def load_legacy_era_state_into_sim(sim, finetuned_path: str | Path, bank_path: str | Path, device_name: str) -> None:
    finetuned_path = Path(finetuned_path)
    bank_path = Path(bank_path)
    if not finetuned_path.exists():
        raise FileNotFoundError(f"finetuned checkpoint not found: {finetuned_path}")
    if not bank_path.exists():
        raise FileNotFoundError(f"knowledge bank snapshot not found: {bank_path}")

    checkpoint = torch.load(finetuned_path, map_location=device_name)
    bank = torch.load(bank_path, map_location=device_name)
    _apply_legacy_era_artifacts_to_sim(sim, checkpoint=checkpoint, bank=bank, device_name=device_name)


def load_fully_trained_era_agent(
    checkpoint_path: str | Path | None = None,
    finetuned_path: str | Path | None = None,
    bank_path: str | Path | None = None,
    device_name: str = "cpu",
) -> torch.nn.Module:
    if checkpoint_path is not None:
        return load_era_checkpoint(checkpoint_path, device=device_name)

    if finetuned_path is None or bank_path is None:
        raise ValueError("Either checkpoint_path or both finetuned_path/bank_path must be provided.")

    agent = build_external_era_agent(latent_dim=128).to(device_name)
    holder = SimpleNamespace(agent=agent)
    load_legacy_era_state_into_sim(
        holder,
        finetuned_path=finetuned_path,
        bank_path=bank_path,
        device_name=device_name,
    )
    agent.eval()
    return agent
