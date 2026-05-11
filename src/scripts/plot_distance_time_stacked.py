from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ("era", "vpf", "bc_il", "ppo", "acados")
METHOD_LABELS = {
    "era": "ERA",
    "vpf": "VPF Expert",
    "bc_il": "BC-IL",
    "ppo": "PPO",
    "acados": "Acados",
}
METHOD_COLORS = {
    "era": "#005f73",
    "vpf": "#0a9396",
    "bc_il": "#ee9b00",
    "ppo": "#bb3e03",
    "acados": "#3a86ff",
}

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Plot five stacked distance-vs-time panels, one for each method."
    )
    parser.add_argument(
        "--trajectory-json",
        default=str(root / "outputs" / "all_methods_medium_seed25_e3_s2000_trajectory.json"),
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--output-path",
        default=None,
        help="Defaults to outputs/plots/stacked_distance/sample_<index>_stacked_distance.png",
    )
    parser.add_argument(
        "--difficulty-label",
        default=None,
        help="Optional label for the figure title, e.g. medium or extreme.",
    )
    parser.add_argument("--show", choices=["true", "false"], default="false")
    return parser.parse_args()


def load_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_series(points: list[dict], key: str) -> np.ndarray:
    values = []
    for point in points:
        value = point.get(key)
        values.append(np.nan if value is None else float(value))
    return np.asarray(values, dtype=float)


def select_episode(payload: dict, method_key: str, sample_index: int) -> dict:
    matches = [episode for episode in payload.get("episodes", []) if episode.get("method_key") == method_key]
    if not matches:
        raise ValueError(f"No episodes found for method_key={method_key!r}")
    if sample_index < 0 or sample_index >= len(matches):
        raise IndexError(f"sample_index={sample_index} is out of range for {len(matches)} samples of {method_key}")
    return matches[sample_index]


def resolve_output_path(trajectory_json_path: Path, sample_index: int, output_path: str | None) -> Path:
    if output_path:
        resolved = Path(output_path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved
    out_dir = trajectory_json_path.parent / "plots" / "stacked_distance"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"sample_{sample_index}_stacked_distance.png"


def build_title(payload: dict, sample_index: int, difficulty_label: str | None) -> str:
    if difficulty_label:
        return f"Distance-to-intruder comparison across methods, sample {sample_index}, {difficulty_label}"
    episodes = payload.get("episodes", []) or []
    if episodes:
        level = str((episodes[0].get("difficulty_config", {}) or {}).get("level", "")).strip()
        if level:
            return f"Distance-to-intruder comparison across methods, sample {sample_index}, {level}"
    return f"Distance-to-intruder comparison across methods, sample {sample_index}"


def plot_stacked_distance(payload: dict, sample_index: int, output_path: Path, difficulty_label: str | None) -> Path:
    fig, axes = plt.subplots(
        nrows=len(METHOD_ORDER),
        ncols=1,
        figsize=(13, 11.6),
        sharex=True,
        constrained_layout=True,
    )

    for ax, method_key in zip(axes, METHOD_ORDER):
        episode = select_episode(payload, method_key=method_key, sample_index=sample_index)
        points = episode.get("trajectory_points", []) or []
        if not points:
            raise ValueError(f"Selected episode has no trajectory points for method_key={method_key!r}")

        time_s = extract_series(points, "simulation_time_s")
        min_center = extract_series(points, "min_intruder_center_distance")
        min_surface = extract_series(points, "min_intruder_surface_distance")
        color = METHOD_COLORS[method_key]
        ax.plot(time_s, min_center, linewidth=2.2, color=color, label="Min center distance")
        ax.plot(time_s, min_surface, linewidth=2.0, color="#9b2226", label="Min surface distance")
        ax.axhline(0.0, linestyle="--", linewidth=1.2, color="#ae2012", alpha=0.9, label="Collision threshold")
        ax.axhline(1.0, linestyle=":", linewidth=1.2, color="#ca6702", alpha=0.9, label="Safety threshold")
        ax.set_ylabel("Distance (m)")
        ax.grid(True, alpha=0.28)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        finite_values = np.concatenate(
            [
                min_center[np.isfinite(min_center)],
                min_surface[np.isfinite(min_surface)],
            ]
        )
        if finite_values.size > 0:
            ymin = min(-0.15, float(np.min(finite_values)) - 0.2)
            ymax = max(4.0, float(np.max(finite_values)) + 0.55)
            ax.set_ylim(ymin, ymax)

        success_flag = int((episode.get("episode_metrics", {}) or {}).get("success_flag", 0))
        final_distance = float((episode.get("episode_metrics", {}) or {}).get("final_distance", 0.0))
        ax.set_title(
            f"{METHOD_LABELS[method_key]}  |  goal_reached={success_flag}  |  final_distance={final_distance:.2f} m",
            loc="left",
            fontsize=15,
            fontweight="bold",
        )

        if method_key == METHOD_ORDER[0]:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(build_title(payload, sample_index, difficulty_label), fontsize=18, fontweight="bold")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    trajectory_json_path = Path(args.trajectory_json).expanduser().resolve()
    payload = load_payload(trajectory_json_path)
    output_path = resolve_output_path(trajectory_json_path, args.sample_index, args.output_path)
    figure_path = plot_stacked_distance(
        payload=payload,
        sample_index=args.sample_index,
        output_path=output_path,
        difficulty_label=args.difficulty_label,
    )
    print(f"[plot_distance_time_stacked] figure={figure_path}")
    if args.show == "true":
        plt.show()


if __name__ == "__main__":
    main()
