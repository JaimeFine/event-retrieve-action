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

from bruce_code.methods.era import (
    EraAdapter,
    EraIsaacEnvironment,
    build_era_trajectory_payload,
    flatten_era_trajectory_rows,
    load_legacy_era_state_into_sim,
)
from bruce_code.sim_shared.constants import device, seeds
from bruce_code.sim_shared.difficulty import get_difficulty_profile_names


ABLATION_SPECS = {
    "era_100": {
        "label": "ERA-100",
        "method_key": "era_100",
        "finetuned": Path("datasets/checkpoints/full/agent_finetuned_99.pt"),
        "bank": Path("datasets/checkpoints/full/knowledge_bank_snapshot_99.pt"),
    },
    "no_phys": {
        "label": "ERA w/o R_phys",
        "method_key": "era_no_phys",
        "finetuned": Path("datasets/checkpoints/ablations/no_phys_finetuned.pt"),
        "bank": Path("datasets/checkpoints/ablations/no_phys_bank.pt"),
    },
    "no_cws": {
        "label": "ERA w/o CBS",
        "method_key": "era_no_cws",
        "finetuned": Path("datasets/checkpoints/ablations/no_bay_finetuned.pt"),
        "bank": Path("datasets/checkpoints/ablations/no_bay_bank.pt"),
    },
}


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


def build_environment(seed: int, difficulty_mode: str, difficulty_level: str):
    sim = EraIsaacEnvironment(
        seed=seed,
        difficulty_mode=difficulty_mode,
        difficulty_level=difficulty_level,
    )
    sim.setup_environment()
    freeze_runtime_mutations(sim)
    return sim


def resolve_checkpoint(root: Path, spec_key: str, field: str) -> Path:
    rel = ABLATION_SPECS[spec_key][field]
    path = (root / rel).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint for {spec_key}:{field}: {path}")
    return path


def evaluate_variant(
    *,
    sim,
    root: Path,
    spec_key: str,
    episodes: int,
    steps: int,
    seed: int,
    difficulty_mode: str,
    difficulty_level: str,
    device_name: str,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    finetuned_path = resolve_checkpoint(root, spec_key, "finetuned")
    bank_path = resolve_checkpoint(root, spec_key, "bank")
    load_legacy_era_state_into_sim(
        sim,
        finetuned_path=finetuned_path,
        bank_path=bank_path,
        device_name=device_name,
    )
    adapter = EraAdapter(selector_fn=sim.agent.select_action, device_name=device_name)
    goal_rng = np.random.RandomState(seed)
    adapter.reset()

    if hasattr(sim, "scheduler") and hasattr(sim.scheduler, "recent_rewards"):
        sim.scheduler.recent_rewards.clear()

    decision_times_ms = install_adapter(sim, adapter)
    rows: list[dict] = []
    trajectory_payloads: list[dict] = []
    trajectory_rows: list[dict] = []

    label = str(ABLATION_SPECS[spec_key]["label"])
    method_key = str(ABLATION_SPECS[spec_key]["method_key"])

    for episode_index in range(episodes):
        decision_times_ms.clear()
        episode_seed = seed + episode_index
        episode = run_episode(sim, goal_rng=goal_rng, steps=steps, episode_seed=episode_seed)
        payload = build_era_trajectory_payload(
            seed=seed,
            episode_index=episode_index,
            episode_seed=episode_seed,
            difficulty_mode=difficulty_mode,
            difficulty_level=difficulty_level,
            episode_metrics=episode,
            trajectory_points=getattr(sim, "last_episode_trajectory", []) or [],
        )
        payload["method_key"] = method_key
        payload["method_name"] = label
        payload["method_config"] = {
            "variant_key": spec_key,
            "finetuned_path": str(finetuned_path),
            "bank_path": str(bank_path),
        }
        trajectory_payloads.append(payload)
        flattened = flatten_era_trajectory_rows(payload)
        for row in flattened:
            row["method_key"] = method_key
            row["method_name"] = label
        trajectory_rows.extend(flattened)

        rows.append(
            {
                "seed": int(seed),
                "difficulty_mode": str(difficulty_mode),
                "difficulty_level": str(difficulty_level),
                "method_key": method_key,
                "method_name": label,
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
                "variant_key": spec_key,
                "finetuned_path": str(finetuned_path),
                "bank_path": str(bank_path),
            }
        )

    summary = {
        "method": label,
        "variant_key": spec_key,
        "episodes": int(episodes),
        "steps_per_episode": int(steps),
        "success_rate": float(np.mean([row["success_rate"] for row in rows])) if rows else 0.0,
        "collision_rate": float(np.mean([row["collision_rate"] for row in rows])) if rows else 0.0,
        "warning_rate": float(np.mean([row["warning_rate"] for row in rows])) if rows else 0.0,
        "avg_final_distance": float(np.mean([row["final_distance"] for row in rows])) if rows else 0.0,
        "avg_effective_steps": float(np.mean([row["effective_steps"] for row in rows])) if rows else 0.0,
        "avg_reaction_time_ms": float(np.mean([row["avg_reaction_time_ms"] for row in rows])) if rows else 0.0,
        "goal_reach_rate": float(np.mean([row["success_flag"] for row in rows])) if rows else 0.0,
    }
    simulation_app.update()
    return summary, rows, trajectory_payloads, trajectory_rows


def parse_variants(raw_variants: str) -> list[str]:
    requested = [item.strip().lower() for item in raw_variants.split(",") if item.strip()]
    if not requested:
        raise ValueError("At least one ablation variant must be provided via --variants.")
    unknown = [item for item in requested if item not in ABLATION_SPECS]
    if unknown:
        raise ValueError(f"Unsupported --variants entries: {', '.join(unknown)}")
    seen: set[str] = set()
    result: list[str] = []
    for item in requested:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Sample trajectories for ERA-100 and ablation variants only."
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=seeds)
    parser.add_argument("--headless", choices=["true", "false"], default="true")
    parser.add_argument("--device", default=str(device))
    parser.add_argument("--difficulty-mode", choices=["curriculum", "fixed"], default="fixed")
    parser.add_argument("--difficulty-level", choices=get_difficulty_profile_names(), default="extreme")
    parser.add_argument(
        "--variants",
        default="era_100,no_phys,no_cws",
        help="Comma-separated subset of ablation variants: era_100,no_phys,no_cws",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-trajectory-json", default=None)
    parser.add_argument("--output-trajectory-csv", default=None)
    return parser.parse_args()


def default_stem(args) -> str:
    return f"ablation_era_only_{args.difficulty_level}_seed{args.seed}_e{args.episodes}_s{args.steps}"


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    set_deterministic_seeds(args.seed)
    selected_variants = parse_variants(args.variants)
    sim = build_environment(
        seed=args.seed,
        difficulty_mode=args.difficulty_mode,
        difficulty_level=args.difficulty_level,
    )

    output_dir = root / "outputs" / "ablation_runs"
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

    results: dict[str, dict] = {
        "difficulty_config": {
            "mode": str(args.difficulty_mode),
            "level": str(args.difficulty_level),
            "shared_for_all_variants": True,
        },
        "sampling_config": {
            "episodes": int(args.episodes),
            "steps": int(args.steps),
            "seed": int(args.seed),
        },
    }
    all_rows: list[dict] = []
    all_payloads: list[dict] = []
    all_trajectory_rows: list[dict] = []

    for variant_key in selected_variants:
        summary, rows, payloads, trajectory_rows = evaluate_variant(
            sim=sim,
            root=root,
            spec_key=variant_key,
            episodes=args.episodes,
            steps=args.steps,
            seed=args.seed,
            difficulty_mode=args.difficulty_mode,
            difficulty_level=args.difficulty_level,
            device_name=args.device,
        )
        results[variant_key] = summary
        all_rows.extend(rows)
        all_payloads.extend(payloads)
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
        "episodes": all_payloads,
    }
    trajectory_json_path.parent.mkdir(parents=True, exist_ok=True)
    with trajectory_json_path.open("w", encoding="utf-8") as file:
        json.dump(trajectory_export, file, indent=2, ensure_ascii=False)
    write_csv(trajectory_csv_path, all_trajectory_rows)

    print(f"[run_ablation_trajectory_sampling] json={json_path}")
    print(f"[run_ablation_trajectory_sampling] csv={csv_path}")
    print(f"[run_ablation_trajectory_sampling] trajectory_json={trajectory_json_path}")
    print(f"[run_ablation_trajectory_sampling] trajectory_csv={trajectory_csv_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
