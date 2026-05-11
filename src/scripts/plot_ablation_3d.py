from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_JSON = ROOT / "outputs" / "ablation_runs" / "ablation_era_only_extreme_seed25_e10_s2000_trajectory.json"
DEFAULT_OUTPUT_PDF = ROOT / "outputs" / "ablation_runs" / "ablation_3d_sample1.pdf"

METHOD_ORDER = ("era_100", "era_no_phys", "era_no_cws")
METHOD_LABELS = {
    "era_100": "ERA-100",
    "era_no_phys": "ERA w/o R_phys",
    "era_no_cws": "ERA w/o CWS",
}
METHOD_COLORS = {
    "era_100": "#005f73",
    "era_no_phys": "#ee9b00",
    "era_no_cws": "#bb3e03",
}
METHOD_LINESTYLES = {
    "era_100": "-",
    "era_no_phys": "--",
    "era_no_cws": "-.",
}

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a 3-in-1 ablation 3D trajectory comparison.")
    parser.add_argument("--trajectory-json", default=str(DEFAULT_INPUT_JSON))
    parser.add_argument("--sample-index", type=int, default=1)
    parser.add_argument("--output-pdf", default=str(DEFAULT_OUTPUT_PDF))
    parser.add_argument("--difficulty-label", default=None)
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
        raise IndexError(f"sample_index={sample_index} is out of range for {len(matches)} samples")
    return matches[sample_index]


def build_title(payload: dict, sample_index: int, difficulty_label: str | None) -> str:
    if difficulty_label:
        return f"Ablation 3D trajectory comparison, sample {sample_index}, {difficulty_label}"
    episodes = payload.get("episodes", []) or []
    if episodes:
        level = str((episodes[0].get("difficulty_config", {}) or {}).get("level", "")).strip()
        if level:
            return f"Ablation 3D trajectory comparison, sample {sample_index}, {level}"
    return f"Ablation 3D trajectory comparison, sample {sample_index}"


def main() -> None:
    args = parse_args()
    trajectory_json = Path(args.trajectory_json).expanduser().resolve()
    output_pdf = Path(args.output_pdf).expanduser().resolve()
    payload = load_payload(trajectory_json)

    fig = plt.figure(figsize=(11.5, 8.8))
    ax = fig.add_subplot(111, projection="3d")
    goal_added = False
    plotted_labels: list[str] = []

    for method_key in METHOD_ORDER:
        episode = select_episode(payload, method_key=method_key, sample_index=args.sample_index)
        points = episode.get("trajectory_points", []) or []
        if not points:
            raise ValueError(f"No trajectory points found for method_key={method_key!r}, sample_index={args.sample_index}")

        x = extract_series(points, "ego_x")
        y = extract_series(points, "ego_y")
        z = extract_series(points, "ego_z")
        goal_x = extract_series(points, "goal_x")
        goal_y = extract_series(points, "goal_y")
        goal_z = extract_series(points, "goal_z")
        color = METHOD_COLORS[method_key]

        ax.plot(
            x,
            y,
            z,
            linewidth=3.0,
            linestyle=METHOD_LINESTYLES[method_key],
            color=color,
            alpha=0.58,
            label=METHOD_LABELS[method_key],
        )
        plotted_labels.append(METHOD_LABELS[method_key])
        ax.scatter(x[0], y[0], z[0], color=color, s=34, alpha=0.85)
        ax.scatter(x[-1], y[-1], z[-1], color=color, s=52, marker="o", alpha=0.98)
        if not goal_added:
            ax.scatter(goal_x[-1], goal_y[-1], goal_z[-1], color="#6a040f", s=130, marker="*", label="Goal")
            goal_added = True

    ax.set_title(build_title(payload, args.sample_index, args.difficulty_label), pad=18, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    handles, labels = ax.get_legend_handles_labels()
    if "Goal" in labels:
        ordered = list(METHOD_ORDER)
        ordered_labels = [METHOD_LABELS[key] for key in ordered] + ["Goal"]
        label_to_handle = {label: handle for handle, label in zip(handles, labels)}
        ordered_handles = [label_to_handle[label] for label in ordered_labels if label in label_to_handle]
        ordered_labels = [label for label in ordered_labels if label in label_to_handle]
        ax.legend(ordered_handles, ordered_labels, loc="best")
    else:
        ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_ablation_3d] source={trajectory_json}")
    print(f"[plot_ablation_3d] output={output_pdf}")


if __name__ == "__main__":
    main()
