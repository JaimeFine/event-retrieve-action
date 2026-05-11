# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import csv
import importlib.util
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

from bruce_code.dataset import load_experiences
from bruce_code.methods.acados import build_acados_adapter
from bruce_code.methods.bc import build_bc_adapter
from bruce_code.methods.era import (
    EraAdapter,
    EraIsaacEnvironment,
    build_era_trajectory_payload,
    flatten_era_trajectory_rows,
    load_legacy_era_state_into_sim,
)
from bruce_code.methods.ppo import build_ppo_adapter
from bruce_code.methods.vpf import (
    build_vpf_adapter,
    build_vpf_trajectory_payload,
    flatten_vpf_trajectory_rows,
)
from bruce_code.sim_shared.constants import BATCH_SIZE, EPOCHS, device, seeds
from bruce_code.sim_shared.difficulty import get_difficulty_profile_names


METHOD_ORDER = ("era", "vpf", "bc_il", "ppo", "acados")


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
        "trajectory_points": int(summary.get("trajectory_points", 0)),
        "env_steps": int(summary.get("env_steps", 0)),
        "decision_steps": int(summary.get("decision_steps", 0)),
    }


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


def build_generic_trajectory_payload(
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
    if method_key == "era":
        return build_era_trajectory_payload(
            seed=seed,
            episode_index=episode_index,
            episode_seed=episode_seed,
            difficulty_mode=difficulty_mode,
            difficulty_level=difficulty_level,
            episode_metrics=episode_metrics,
            trajectory_points=trajectory_points,
        )
    if method_key == "vpf":
        return build_vpf_trajectory_payload(
            seed=seed,
            episode_index=episode_index,
            episode_seed=episode_seed,
            difficulty_mode=difficulty_mode,
            difficulty_level=difficulty_level,
            episode_metrics=episode_metrics,
            trajectory_points=trajectory_points,
            method_config=method_config,
        )

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


def flatten_generic_trajectory_rows(payload: dict) -> list[dict]:
    if payload.get("method_key") == "era":
        return flatten_era_trajectory_rows(payload)
    if payload.get("method_key") == "vpf":
        return flatten_vpf_trajectory_rows(payload)

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


def evaluate_method(
    sim,
    adapter,
    *,
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
        trajectory_payload = build_generic_trajectory_payload(
            method_key=method_key,
            method_name=method_name,
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
        trajectory_rows.extend(flatten_generic_trajectory_rows(trajectory_payload))

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
                "trajectory_points": int(episode["trajectory_points"]),
                "env_steps": int(episode["env_steps"]),
                "decision_steps": int(episode["decision_steps"]),
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
        "adapter_metadata": dict(getattr(adapter, "metadata", {}) or {}),
    }
    return summary, rows, trajectory_payloads, trajectory_rows


def build_non_era_adapters(args):
    experiences = load_experiences(args.dataset)
    missing_ppo_dependencies = [
        module_name
        for module_name in ("stable_baselines3", "gymnasium")
        if importlib.util.find_spec(module_name) is None
    ]
    if missing_ppo_dependencies:
        print(
            "[run_all_methods_trajectory_sampling] skipping ppo: missing optional dependencies: "
            + ", ".join(missing_ppo_dependencies)
        )
    return {
        "bc_il": build_bc_adapter(
            experiences,
            device_name=args.device,
            epochs=args.train_epochs,
            batch_size=args.train_batch_size,
            model_path=args.bc_model_path,
        ),
        "ppo": (
            None
            if missing_ppo_dependencies
            else build_ppo_adapter(
                experiences,
                device_name=args.device,
                total_timesteps=(None if args.ppo_timesteps <= 0 else args.ppo_timesteps),
                epochs=args.train_epochs,
                batch_size=args.train_batch_size,
                seed=args.seed,
                model_path=args.ppo_model_path,
                training_artifact_dir=args.ppo_artifact_dir,
                run_name=f"ppo_transformer_seed{args.seed}",
            )
        ),
        "acados": build_acados_adapter(device_name=args.device, backend=args.acados_backend),
        "vpf": build_vpf_adapter(
            device_name=args.device,
            detection_threshold=args.vpf_detection_threshold,
            attractive_gain=args.vpf_attractive_gain,
            repulsive_gain=args.vpf_repulsive_gain,
            max_speed=args.vpf_max_speed,
            stuck_threshold=args.vpf_stuck_threshold,
            goal_far_threshold=args.vpf_goal_far_threshold,
            tangential_gain=args.vpf_tangential_gain,
        ),
    }


def parse_methods_arg(raw_methods: str) -> list[str]:
    requested = [item.strip().lower() for item in raw_methods.split(",") if item.strip()]
    if not requested:
        raise ValueError("At least one method must be provided via --methods.")

    unknown = [item for item in requested if item not in METHOD_ORDER]
    if unknown:
        raise ValueError(f"Unsupported methods in --methods: {', '.join(unknown)}")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in requested:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Sample trajectories for ERA, VPF, BC, PPO, and Acados with unified export."
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=seeds)
    parser.add_argument("--headless", choices=["true", "false"], default="true")
    parser.add_argument("--device", default=str(device))
    parser.add_argument("--difficulty-mode", choices=["curriculum", "fixed"], default="fixed")
    parser.add_argument("--difficulty-level", choices=get_difficulty_profile_names(), default="medium")
    parser.add_argument("--finetuned", default=str(root / "agent_finetuned.pt"))
    parser.add_argument("--bank", default=str(root / "knowledge_bank_snapshot.pt"))
    parser.add_argument("--dataset", default=str(root / "artifacts" / "expert_dataset.pt"))
    parser.add_argument("--train-epochs", type=int, default=EPOCHS)
    parser.add_argument("--train-batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--ppo-timesteps", type=int, default=0)
    parser.add_argument("--bc-model-path", default=str(root / "artifacts" / "bc_policy.pt"))
    parser.add_argument("--ppo-model-path", default=str(root / "artifacts" / "ppo_policy.zip"))
    parser.add_argument("--ppo-artifact-dir", default=str(root / "artifacts" / "ppo"))
    parser.add_argument("--acados-backend", choices=["auto", "acados", "grid"], default="auto")
    parser.add_argument(
        "--methods",
        default=",".join(METHOD_ORDER),
        help="Comma-separated subset of methods to run. Supported: era,vpf,bc_il,ppo,acados",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
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
    return f"all_methods_{args.difficulty_level}_seed{args.seed}_e{args.episodes}_s{args.steps}"


def build_environment(args):
    sim = EraIsaacEnvironment(
        seed=args.seed,
        difficulty_mode=args.difficulty_mode,
        difficulty_level=args.difficulty_level,
    )
    sim.setup_environment()
    freeze_runtime_mutations(sim)
    load_legacy_era_state_into_sim(
        sim,
        finetuned_path=args.finetuned,
        bank_path=args.bank,
        device_name=args.device,
    )
    return sim


def main() -> None:
    args = parse_args()
    set_deterministic_seeds(args.seed)
    selected_methods = parse_methods_arg(args.methods)

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
    results: dict[str, dict] = {
        "difficulty_config": {
            "mode": str(args.difficulty_mode),
            "level": str(args.difficulty_level),
            "shared_for_all_methods": True,
        },
        "sampling_config": {
            "episodes": int(args.episodes),
            "steps": int(args.steps),
            "seed": int(args.seed),
        },
        "artifacts": {
            "finetuned": str(Path(args.finetuned).expanduser().resolve()),
            "bank": str(Path(args.bank).expanduser().resolve()),
            "dataset": str(Path(args.dataset).expanduser().resolve()),
        },
    }

    all_rows: list[dict] = []
    all_trajectory_payloads: list[dict] = []
    all_trajectory_rows: list[dict] = []

    if "era" in selected_methods:
        era_adapter = EraAdapter(selector_fn=sim.agent.select_action, device_name=args.device)
        era_summary, era_rows, era_payloads, era_trajectory_rows = evaluate_method(
            sim,
            era_adapter,
            method_key="era",
            method_name="era",
            episodes=args.episodes,
            steps=args.steps,
            seed=args.seed,
            difficulty_mode=args.difficulty_mode,
            difficulty_level=args.difficulty_level,
        )
        results["era"] = era_summary
        all_rows.extend(era_rows)
        all_trajectory_payloads.extend(era_payloads)
        all_trajectory_rows.extend(era_trajectory_rows)

    non_era_methods = [method for method in selected_methods if method != "era"]
    adapters = build_non_era_adapters(args) if non_era_methods else {}
    for method_key in non_era_methods:
        adapter = adapters.get(method_key)
        if adapter is None:
            error_message = "adapter unavailable in current environment"
            if method_key == "ppo":
                error_message = "skipped: missing optional PPO dependencies (stable_baselines3 and/or gymnasium)"
            results[method_key] = {"method": method_key, "error": error_message}
            continue

        method_name = getattr(adapter, "name", method_key)
        summary, rows, payloads, trajectory_rows = evaluate_method(
            sim,
            adapter,
            method_key=method_key,
            method_name=method_name,
            episodes=args.episodes,
            steps=args.steps,
            seed=args.seed,
            difficulty_mode=args.difficulty_mode,
            difficulty_level=args.difficulty_level,
        )
        results[method_key] = summary
        all_rows.extend(rows)
        all_trajectory_payloads.extend(payloads)
        all_trajectory_rows.extend(trajectory_rows)

    results["per_episode_rows"] = all_rows

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    write_csv(csv_path, all_rows)

    trajectory_export = {
        "difficulty_config": {
            "mode": str(args.difficulty_mode),
            "level": str(args.difficulty_level),
        },
        "sampling_config": {
            "episodes": int(args.episodes),
            "steps": int(args.steps),
            "seed": int(args.seed),
        },
        "episodes": all_trajectory_payloads,
    }
    trajectory_json_path.parent.mkdir(parents=True, exist_ok=True)
    with trajectory_json_path.open("w", encoding="utf-8") as file:
        json.dump(trajectory_export, file, indent=2, ensure_ascii=False)
    write_csv(trajectory_csv_path, all_trajectory_rows)

    print(f"[run_all_methods_trajectory_sampling] json={json_path}")
    print(f"[run_all_methods_trajectory_sampling] csv={csv_path}")
    print(f"[run_all_methods_trajectory_sampling] trajectory_json={trajectory_json_path}")
    print(f"[run_all_methods_trajectory_sampling] trajectory_csv={trajectory_csv_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
