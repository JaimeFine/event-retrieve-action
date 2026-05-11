# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

try:
    from omni.isaac.kit import SimulationApp
except Exception as exc:
    raise RuntimeError(
        "Failed to import omni.isaac.kit.SimulationApp. "
        "Please run this script with Isaac Sim Python."
    ) from exc

from _bootstrap import ensure_local_package


def _detect_headless_from_argv() -> bool:
    if "--headless" not in sys.argv:
        return True
    try:
        idx = sys.argv.index("--headless")
        value = sys.argv[idx + 1].strip().lower()
        return value != "false"
    except Exception:
        return True


simulation_app = SimulationApp({"headless": _detect_headless_from_argv()})

ensure_local_package()

from bruce_code.methods.vpf import (
    VpfIsaacEnvironment,
    build_vpf_adapter,
    build_vpf_trajectory_payload,
    flatten_vpf_trajectory_rows,
)
from bruce_code.sim_shared.constants import device, seeds
from bruce_code.sim_shared.difficulty import get_difficulty_profile_names


def set_deterministic_seeds(seed_value: int) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def sample_goal(goal_rng: np.random.RandomState, ego_start: np.ndarray) -> np.ndarray:
    direction = goal_rng.normal(size=3)
    direction[2] = abs(direction[2]) + 0.1
    direction /= np.linalg.norm(direction) + 1e-8
    distance = 100.0
    return ego_start + direction * distance


def freeze_runtime_mutations(sim) -> None:
    def _zero_scalar(*args, **kwargs):
        return 0.0

    def _noop(*args, **kwargs):
        return None

    if hasattr(sim, "intruder_controller") and hasattr(sim.intruder_controller, "update"):
        sim.intruder_controller.update = _zero_scalar
    if hasattr(sim, "intruder_controller") and hasattr(sim.intruder_controller, "store"):
        sim.intruder_controller.store = _noop
    if hasattr(sim.agent, "memory") and hasattr(sim.agent.memory, "penalize_by_indices"):
        sim.agent.memory.penalize_by_indices = _noop


def install_adapter(sim, adapter):
    decision_times_ms: list[float] = []

    def _wrapped_select_action(event_list, k=5):
        start = time.perf_counter()
        action = adapter.act(event_list)
        decision_times_ms.append((time.perf_counter() - start) * 1000.0)
        return action, None, None, None, None

    sim.agent.select_action = _wrapped_select_action
    return decision_times_ms


def run_episode(sim, goal_rng: np.random.RandomState, steps: int, episode_seed: int) -> dict:
    sim.ego_goal = sample_goal(goal_rng, sim.ego_start)
    s, c, w, d, t, r_phys, j_perf, i_loss = sim.run(
        steps=steps,
        episode_seed=episode_seed,
    )
    safe_t = max(1, t)
    summary = getattr(sim, "last_episode_summary", {}) or {}
    return {
        "effective_steps": int(t),
        "success_rate": float((t - w - c) / safe_t),
        "collision_rate": float(c / safe_t),
        "warning_rate": float(w / safe_t),
        "final_distance": float(d),
        "phys_loss": float(r_phys),
        "perf_loss": float(j_perf),
        "intruder_loss": float(i_loss),
        "difficulty_name": str(summary.get("difficulty_name", "")),
        "difficulty_scalar": float(summary.get("difficulty_scalar", 0.0)),
        "success_flag": int(s),
    }


def evaluate_method(
    sim,
    adapter,
    method_key: str,
    method_name: str,
    episodes: int,
    steps: int,
    seed: int,
    difficulty_mode: str,
    difficulty_level: str,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    goal_rng = np.random.RandomState(seed)
    adapter.reset()

    if hasattr(sim, "scheduler") and hasattr(sim.scheduler, "recent_rewards"):
        sim.scheduler.recent_rewards.clear()

    decision_times_ms = install_adapter(sim, adapter)
    rows: list[dict] = []
    trajectory_payloads: list[dict] = []
    trajectory_rows: list[dict] = []

    for episode_index in range(episodes):
        decision_times_ms.clear()
        episode_seed = seed + episode_index
        episode = run_episode(sim, goal_rng=goal_rng, steps=steps, episode_seed=episode_seed)
        trajectory_payload = build_vpf_trajectory_payload(
            seed=seed,
            episode_index=episode_index,
            episode_seed=episode_seed,
            difficulty_mode=difficulty_mode,
            difficulty_level=difficulty_level,
            episode_metrics=episode,
            trajectory_points=getattr(sim, "last_episode_trajectory", []) or [],
            method_config=getattr(adapter, "metadata", {}) or {},
        )
        trajectory_payloads.append(trajectory_payload)
        trajectory_rows.extend(flatten_vpf_trajectory_rows(trajectory_payload))
        rows.append(
            {
                "seed": int(seed),
                "difficulty_mode": str(difficulty_mode),
                "difficulty_level": str(difficulty_level),
                "method_key": str(method_key),
                "method_name": str(method_name),
                "episode_index": int(episode_index),
                "episode_seed": int(episode_seed),
                "effective_steps": int(episode["effective_steps"]),
                "success_rate": float(episode["success_rate"]),
                "collision_rate": float(episode["collision_rate"]),
                "warning_rate": float(episode["warning_rate"]),
                "final_distance": float(episode["final_distance"]),
                "phys_loss": float(episode["phys_loss"]),
                "perf_loss": float(episode["perf_loss"]),
                "intruder_loss": float(episode["intruder_loss"]),
                "difficulty_name": str(episode["difficulty_name"]),
                "difficulty_scalar": float(episode["difficulty_scalar"]),
                "avg_reaction_time_ms": (
                    float(sum(decision_times_ms) / len(decision_times_ms)) if decision_times_ms else 0.0
                ),
                "num_reaction_samples": int(len(decision_times_ms)),
                "success_flag": int(episode["success_flag"]),
            }
        )

    summary = {
        "method": str(method_name),
        "episodes": int(episodes),
        "steps_per_episode": int(steps),
        "success_rate": float(np.mean([row["success_rate"] for row in rows])) if rows else 0.0,
        "collision_rate": float(np.mean([row["collision_rate"] for row in rows])) if rows else 0.0,
        "warning_rate": float(np.mean([row["warning_rate"] for row in rows])) if rows else 0.0,
        "avg_final_distance": float(np.mean([row["final_distance"] for row in rows])) if rows else 0.0,
        "avg_phys_loss": float(np.mean([row["phys_loss"] for row in rows])) if rows else 0.0,
        "avg_perf_loss": float(np.mean([row["perf_loss"] for row in rows])) if rows else 0.0,
        "avg_intruder_loss": float(np.mean([row["intruder_loss"] for row in rows])) if rows else 0.0,
        "avg_effective_steps": float(np.mean([row["effective_steps"] for row in rows])) if rows else 0.0,
        "avg_reaction_time_ms": float(np.mean([row["avg_reaction_time_ms"] for row in rows])) if rows else 0.0,
        "avg_difficulty_scalar": float(np.mean([row["difficulty_scalar"] for row in rows])) if rows else 0.0,
        "difficulty_labels": [str(row["difficulty_name"]) for row in rows],
    }
    return summary, rows, trajectory_payloads, trajectory_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="VPF-only online evaluation with per-episode CSV export."
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=seeds)
    parser.add_argument("--headless", choices=["true", "false"], default="true")
    parser.add_argument("--device", default=str(device))
    parser.add_argument("--difficulty-mode", choices=["curriculum", "fixed"], default="fixed")
    parser.add_argument("--difficulty-level", choices=get_difficulty_profile_names(), default="medium")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--save-trajectories", choices=["true", "false"], default="true")
    parser.add_argument("--output-trajectory-json", default=None)
    parser.add_argument("--output-trajectory-csv", default=None)
    parser.add_argument("--vpf-detection-threshold", type=float, default=5.0)
    parser.add_argument("--vpf-attractive-gain", type=float, default=3.0)
    parser.add_argument("--vpf-repulsive-gain", type=float, default=15.0)
    parser.add_argument("--vpf-max-speed", type=float, default=5.0)
    parser.add_argument("--vpf-stuck-threshold", type=float, default=1.0)
    parser.add_argument("--vpf-goal-far-threshold", type=float, default=2.0)
    parser.add_argument("--vpf-tangential-gain", type=float, default=2.0)
    return parser.parse_args()


def default_stem(args) -> str:
    return f"vpf_only_{args.difficulty_level}_seed{args.seed}_e{args.episodes}_s{args.steps}"


def build_environment(args):
    sim = VpfIsaacEnvironment(
        seed=args.seed,
        difficulty_mode=args.difficulty_mode,
        difficulty_level=args.difficulty_level,
    )
    sim.setup_environment()
    freeze_runtime_mutations(sim)
    return sim


def main() -> None:
    args = parse_args()
    set_deterministic_seeds(args.seed)

    output_dir = Path(__file__).resolve().parents[2] / "outputs"
    stem = default_stem(args)
    json_path = Path(args.output_json).expanduser().resolve() if args.output_json else output_dir / f"{stem}.json"
    csv_path = Path(args.output_csv).expanduser().resolve() if args.output_csv else output_dir / f"{stem}.csv"
    trajectory_json_path = (
        Path(args.output_trajectory_json).expanduser().resolve()
        if args.output_trajectory_json
        else output_dir / f"{stem}_trajectory.json"
    )
    trajectory_csv_path = (
        Path(args.output_trajectory_csv).expanduser().resolve()
        if args.output_trajectory_csv
        else output_dir / f"{stem}_trajectory_points.csv"
    )

    sim = build_environment(args)
    vpf_adapter = build_vpf_adapter(
        device_name=args.device,
        detection_threshold=args.vpf_detection_threshold,
        attractive_gain=args.vpf_attractive_gain,
        repulsive_gain=args.vpf_repulsive_gain,
        max_speed=args.vpf_max_speed,
        stuck_threshold=args.vpf_stuck_threshold,
        goal_far_threshold=args.vpf_goal_far_threshold,
        tangential_gain=args.vpf_tangential_gain,
    )
    vpf_summary, vpf_rows, trajectory_payloads, trajectory_rows = evaluate_method(
        sim,
        adapter=vpf_adapter,
        method_key="vpf",
        method_name="vpf",
        episodes=args.episodes,
        steps=args.steps,
        seed=args.seed,
        difficulty_mode=args.difficulty_mode,
        difficulty_level=args.difficulty_level,
    )

    rows = vpf_rows
    payload = {
        "difficulty_config": {
            "mode": str(args.difficulty_mode),
            "level": str(args.difficulty_level),
        },
        "vpf": vpf_summary,
        "per_episode_rows": rows,
        "vpf_config": dict(vpf_adapter.metadata),
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    write_csv(csv_path, rows)

    if args.save_trajectories == "true":
        trajectory_export = {
            "difficulty_config": {
                "mode": str(args.difficulty_mode),
                "level": str(args.difficulty_level),
            },
            "method": "vpf",
            "vpf_config": dict(vpf_adapter.metadata),
            "episodes": trajectory_payloads,
        }
        trajectory_json_path.parent.mkdir(parents=True, exist_ok=True)
        with trajectory_json_path.open("w", encoding="utf-8") as file:
            json.dump(trajectory_export, file, indent=2, ensure_ascii=False)
        write_csv(trajectory_csv_path, trajectory_rows)

    print(f"[run_era_vpf_comparison] json={json_path}")
    print(f"[run_era_vpf_comparison] csv={csv_path}")
    if args.save_trajectories == "true":
        print(f"[run_era_vpf_comparison] trajectory_json={trajectory_json_path}")
        print(f"[run_era_vpf_comparison] trajectory_csv={trajectory_csv_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
