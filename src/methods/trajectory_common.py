from __future__ import annotations


def build_method_trajectory_payload(
    *,
    method_key: str,
    method_name: str,
    seed: int,
    episode_index: int,
    episode_seed: int,
    difficulty_mode: str,
    difficulty_level: str,
    episode_metrics: dict,
    trajectory_points: list[dict],
    method_config: dict | None = None,
) -> dict:
    return {
        "method_key": str(method_key),
        "method_name": str(method_name),
        "seed": int(seed),
        "episode_index": int(episode_index),
        "episode_seed": int(episode_seed),
        "difficulty_config": {
            "mode": str(difficulty_mode),
            "level": str(difficulty_level),
        },
        "episode_metrics": dict(episode_metrics),
        "method_config": dict(method_config or {}),
        "trajectory_points": [dict(point) for point in trajectory_points],
    }


def flatten_trajectory_rows(payload: dict) -> list[dict]:
    difficulty = payload.get("difficulty_config", {}) or {}
    metrics = payload.get("episode_metrics", {}) or {}
    base_row = {
        "seed": int(payload.get("seed", 0)),
        "method_key": str(payload.get("method_key", "")),
        "method_name": str(payload.get("method_name", "")),
        "episode_index": int(payload.get("episode_index", 0)),
        "episode_seed": int(payload.get("episode_seed", 0)),
        "difficulty_mode": str(difficulty.get("mode", "")),
        "difficulty_level": str(difficulty.get("level", "")),
        "episode_success_flag": int(metrics.get("success_flag", 0)),
        "episode_final_distance": float(metrics.get("final_distance", 0.0)),
        "episode_effective_steps": int(metrics.get("effective_steps", 0)),
    }

    rows: list[dict] = []
    for point in payload.get("trajectory_points", []):
        row = dict(base_row)
        row.update(point)
        rows.append(row)
    return rows
