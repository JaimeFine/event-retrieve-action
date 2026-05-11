from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Plot sampled trajectory results from run_all_methods_trajectory_sampling.py"
    )
    parser.add_argument(
        "--trajectory-json",
        default=str(root / "outputs" / "all_methods_medium_seed25_e3_s2000_trajectory.json"),
    )
    parser.add_argument("--method", default=None, help="Method key, e.g. era, vpf, bc_il, ppo, acados")
    parser.add_argument("--sample-index", type=int, default=0, help="Which one of the 3 samples to plot")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for figures. Defaults to outputs/plots/<method>/sample_<index>",
    )
    parser.add_argument(
        "--plot-mode",
        choices=["single", "combined-3d"],
        default="single",
        help="single: one method plots; combined-3d: all methods overlaid in one 3D figure for the same sample index",
    )
    parser.add_argument("--show", choices=["true", "false"], default="false")
    return parser.parse_args()


def load_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def select_episode(payload: dict, method_key: str, sample_index: int) -> dict:
    matches = [episode for episode in payload.get("episodes", []) if episode.get("method_key") == method_key]
    if not matches:
        raise ValueError(f"No episodes found for method_key={method_key!r}")
    if sample_index < 0 or sample_index >= len(matches):
        raise IndexError(f"sample_index={sample_index} is out of range for {len(matches)} samples")
    return matches[sample_index]


def select_episodes_for_sample(payload: dict, sample_index: int) -> list[dict]:
    matches = [episode for episode in payload.get("episodes", []) if int(episode.get("episode_index", -1)) == sample_index]
    if not matches:
        raise ValueError(f"No episodes found for sample_index={sample_index}")
    return matches


def ensure_output_dir(
    path: Path | None,
    method_key: str | None,
    sample_index: int,
    trajectory_json_path: Path,
    plot_mode: str,
) -> Path:
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
        return path
    if plot_mode == "combined-3d":
        output_dir = trajectory_json_path.parent / "plots" / "combined" / f"sample_{sample_index}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    if method_key is None:
        raise ValueError("method_key is required for single plot mode.")
    output_dir = trajectory_json_path.parent / "plots" / method_key / f"sample_{sample_index}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def extract_series(points: list[dict], key: str) -> np.ndarray:
    values = []
    for point in points:
        value = point.get(key)
        values.append(np.nan if value is None else float(value))
    return np.asarray(values, dtype=float)


def plot_3d_trajectory(points: list[dict], episode: dict, output_dir: Path) -> Path:
    x = extract_series(points, "ego_x")
    y = extract_series(points, "ego_y")
    z = extract_series(points, "ego_z")
    goal_x = extract_series(points, "goal_x")
    goal_y = extract_series(points, "goal_y")
    goal_z = extract_series(points, "goal_z")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, linewidth=2.2, color="#005f73", label="Robot trajectory")
    ax.scatter(x[0], y[0], z[0], color="#ee9b00", s=70, label="Start")
    ax.scatter(x[-1], y[-1], z[-1], color="#bb3e03", s=70, label="End")
    ax.scatter(goal_x[-1], goal_y[-1], goal_z[-1], color="#0a9396", s=90, marker="*", label="Goal")

    ax.set_title(
        f"3D trajectory: {episode['method_name']} sample {episode['episode_index']}",
        pad=18,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = output_dir / f"{episode['method_key']}_sample_{episode['episode_index']}_3d_path.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_combined_3d_trajectory(episodes: list[dict], sample_index: int, output_dir: Path) -> Path:
    method_colors = {
        "era": "#005f73",
        "vpf": "#0a9396",
        "bc_il": "#ee9b00",
        "ppo": "#bb3e03",
        "acados": "#3a86ff",
    }

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    goal_added = False

    for episode in episodes:
        points = episode.get("trajectory_points", []) or []
        if not points:
            continue
        method_key = str(episode.get("method_key", "unknown"))
        method_name = str(episode.get("method_name", method_key))
        color = method_colors.get(method_key, None)
        x = extract_series(points, "ego_x")
        y = extract_series(points, "ego_y")
        z = extract_series(points, "ego_z")
        goal_x = extract_series(points, "goal_x")
        goal_y = extract_series(points, "goal_y")
        goal_z = extract_series(points, "goal_z")

        ax.plot(x, y, z, linewidth=2.3, color=color, label=method_name)
        ax.scatter(x[0], y[0], z[0], color=color, s=28, alpha=0.8)
        ax.scatter(x[-1], y[-1], z[-1], color=color, s=45, marker="o", alpha=0.95)
        if not goal_added:
            ax.scatter(goal_x[-1], goal_y[-1], goal_z[-1], color="#6a040f", s=110, marker="*", label="Goal")
            goal_added = True

    ax.set_title(f"3D trajectory comparison of all methods, sample {sample_index}", pad=18)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = output_dir / f"all_methods_sample_{sample_index}_3d_path.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_distance_time_series(points: list[dict], episode: dict, output_dir: Path) -> Path:
    time_s = extract_series(points, "simulation_time_s")
    min_center = extract_series(points, "min_intruder_center_distance")
    min_surface = extract_series(points, "min_intruder_surface_distance")
    detected_intruders = extract_series(points, "detected_intruders")

    fig, ax1 = plt.subplots(figsize=(11, 6.5))
    ax1.plot(time_s, min_center, linewidth=2.0, color="#0a9396", label="Min center distance")
    ax1.plot(time_s, min_surface, linewidth=2.0, color="#9b2226", label="Min surface distance")
    ax1.axhline(0.0, linestyle="--", linewidth=1.2, color="#ae2012", label="Collision threshold")
    ax1.axhline(1.0, linestyle=":", linewidth=1.2, color="#ca6702", label="Safety threshold")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Distance (m)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.step(time_s, detected_intruders, where="post", linewidth=1.6, color="#3a86ff", label="Detected intruders")
    ax2.set_ylabel("Detected intruders")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    ax1.set_title(
        f"Distance to detected intruders over time: {episode['method_name']} sample {episode['episode_index']}"
    )

    fig.tight_layout()
    output_path = output_dir / f"{episode['method_key']}_sample_{episode['episode_index']}_distance_vs_time.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_sample_overview(payload: dict, method_key: str, output_dir: Path) -> Path:
    method_episodes = [episode for episode in payload.get("episodes", []) if episode.get("method_key") == method_key]
    sample_ids = [int(episode.get("episode_index", idx)) for idx, episode in enumerate(method_episodes)]
    final_distances = [float((episode.get("episode_metrics", {}) or {}).get("final_distance", 0.0)) for episode in method_episodes]
    warnings = [float((episode.get("episode_metrics", {}) or {}).get("warning_rate", 0.0)) for episode in method_episodes]
    collisions = [float((episode.get("episode_metrics", {}) or {}).get("collision_rate", 0.0)) for episode in method_episodes]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(sample_ids))
    width = 0.25
    ax.bar(x - width, final_distances, width=width, color="#0a9396", label="Final distance")
    ax.bar(x, warnings, width=width, color="#ee9b00", label="Warning rate")
    ax.bar(x + width, collisions, width=width, color="#ae2012", label="Collision rate")
    ax.set_xticks(x, [f"sample_{sample_id}" for sample_id in sample_ids])
    ax.set_title(f"Sample overview: {method_key}")
    ax.set_ylabel("Metric value")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    output_path = output_dir / f"{method_key}_sample_overview.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    trajectory_json_path = Path(args.trajectory_json).expanduser().resolve()
    payload = load_payload(trajectory_json_path)

    if args.plot_mode == "combined-3d":
        episodes = select_episodes_for_sample(payload, sample_index=args.sample_index)
        output_dir = ensure_output_dir(
            Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
            None,
            args.sample_index,
            trajectory_json_path,
            args.plot_mode,
        )
        combined_path = plot_combined_3d_trajectory(episodes, args.sample_index, output_dir)
        print(f"[plot_trajectory_samples] combined_3d={combined_path}")
        if args.show == "true":
            plt.show()
        return

    if not args.method:
        raise ValueError("--method is required when --plot-mode single is used.")

    episode = select_episode(payload, method_key=args.method, sample_index=args.sample_index)
    output_dir = ensure_output_dir(
        Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
        args.method,
        args.sample_index,
        trajectory_json_path,
        args.plot_mode,
    )
    points = episode.get("trajectory_points", []) or []
    if not points:
        raise ValueError("Selected episode has no trajectory points to plot.")

    path_3d = plot_3d_trajectory(points, episode, output_dir)
    path_distance = plot_distance_time_series(points, episode, output_dir)
    path_overview = plot_sample_overview(payload, args.method, output_dir)

    print(f"[plot_trajectory_samples] 3d={path_3d}")
    print(f"[plot_trajectory_samples] distance={path_distance}")
    print(f"[plot_trajectory_samples] overview={path_overview}")

    if args.show == "true":
        plt.show()


if __name__ == "__main__":
    main()
