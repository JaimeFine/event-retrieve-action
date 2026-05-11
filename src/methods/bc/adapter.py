from __future__ import annotations

from pathlib import Path

from bruce_code.methods.bc.core import (
    BehavioralCloningPolicy,
    load_behavioral_cloning_policy,
    save_behavioral_cloning_policy,
    train_behavioral_cloning,
)
from bruce_code.config import TrainConfig

from ..common import GenericPolicyAdapter


def build_bc_adapter(
    experiences,
    device_name: str,
    epochs: int = 10,
    batch_size: int = 16,
    model_path: str | None = None,
):
    if model_path:
        path = Path(model_path)
        if path.exists():
            model = load_behavioral_cloning_policy(str(path), device=device_name)
            return GenericPolicyAdapter(
                "bc_il",
                model,
                device_name,
                metadata={"model_path": str(path), "load_mode": "pretrained"},
            )

    model = BehavioralCloningPolicy(device=device_name)
    model = train_behavioral_cloning(
        model,
        experiences,
        TrainConfig(epochs=epochs, batch_size=batch_size, device=device_name),
    )

    if model_path:
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_behavioral_cloning_policy(model, str(path))

    return GenericPolicyAdapter(
        "bc_il",
        model,
        device_name,
        metadata={"model_path": str(model_path) if model_path else "", "load_mode": "trained_now"},
    )
