from __future__ import annotations

from bruce_code.methods.trajectory_common import (
    build_method_trajectory_payload,
    flatten_trajectory_rows,
)


def build_era_trajectory_payload(
    *,
    seed: int,
    episode_index: int,
    episode_seed: int,
    difficulty_mode: str,
    difficulty_level: str,
    episode_metrics: dict,
    trajectory_points: list[dict],
) -> dict:
    return build_method_trajectory_payload(
        method_key="era",
        method_name="era",
        seed=seed,
        episode_index=episode_index,
        episode_seed=episode_seed,
        difficulty_mode=difficulty_mode,
        difficulty_level=difficulty_level,
        episode_metrics=episode_metrics,
        trajectory_points=trajectory_points,
    )


def flatten_era_trajectory_rows(payload: dict) -> list[dict]:
    return flatten_trajectory_rows(payload)
