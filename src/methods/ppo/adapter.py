from __future__ import annotations

import traceback
from pathlib import Path

from bruce_code.methods.ppo.core import (
    get_default_ppo_observation_dim,
    load_stable_baselines_policy_adapter,
    save_stable_baselines_policy_adapter,
    train_ppo_from_experiences,
)

from ..common import GenericPolicyAdapter


def _write_ppo_error_log(message: str) -> None:
    try:
        path = Path("/root/autodl-tmp/code/ppo_error.log")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(message)
            file.write("\n\n")
    except Exception:
        pass


def build_ppo_adapter(
    experiences,
    device_name: str,
    total_timesteps: int | None = None,
    epochs: int = 10,
    batch_size: int = 16,
    seed: int = 25,
    model_path: str | None = None,
    training_artifact_dir: str | None = None,
    run_name: str = "ppo",
):
    if model_path:
        path = Path(model_path)
        if path.exists():
            try:
                policy = load_stable_baselines_policy_adapter(
                    str(path),
                    observation_dim=get_default_ppo_observation_dim(),
                    device=device_name,
                )
                return GenericPolicyAdapter(
                    "ppo",
                    policy,
                    device_name,
                    metadata={
                        "architecture": getattr(policy, "architecture", "threat_aware_transformer"),
                        "model_path": str(path),
                        "load_mode": "pretrained",
                    },
                )
            except Exception as exc:
                err = traceback.format_exc()
                print(f"[PPO] failed to load pretrained model at {path}: {exc}")
                traceback.print_exc()
                _write_ppo_error_log(
                    f"[load_pretrained] path={path}\nerror={exc}\n{err}"
                )

    try:
        policy = train_ppo_from_experiences(
            experiences=experiences,
            total_timesteps=total_timesteps,
            device=device_name,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            training_artifact_dir=training_artifact_dir,
            run_name=run_name,
        )
    except Exception as exc:
        err = traceback.format_exc()
        print(f"[PPO] training failed: {exc}")
        traceback.print_exc()
        _write_ppo_error_log(
            f"[train] model_path={model_path} run_name={run_name}\nerror={exc}\n{err}"
        )
        return None

    if model_path:
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_stable_baselines_policy_adapter(policy, str(path))

    return GenericPolicyAdapter(
        "ppo",
        policy,
        device_name,
        metadata={
            "architecture": getattr(policy, "architecture", "threat_aware_transformer"),
            "model_path": str(model_path) if model_path else "",
            "training_artifacts": dict(getattr(policy, "training_artifacts", {})),
            "load_mode": "trained_now",
        },
    )
