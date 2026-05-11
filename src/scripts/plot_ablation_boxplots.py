from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = ROOT / "outputs" / "ablation_runs" / "ablation_era_only_extreme_seed25_e10_s2000_trajectory_points.csv"
DEFAULT_OUTPUT_PDF = ROOT / "outputs" / "ablation_runs" / "ablation_boxplots.pdf"

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
SMOOTH_WINDOW = 7

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ablation boxplots for ERA-100, ERA w/o R_phys, and ERA w/o CWS."
    )
    parser.add_argument("--trajectory-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--success-only", choices=["true", "false"], default="true")
    parser.add_argument("--output-pdf", default=str(DEFAULT_OUTPUT_PDF))
    return parser.parse_args()


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) < window:
        return x.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xpad, kernel, mode="valid")


def gradient_1d(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.gradient(y, t, edge_order=1)


def finite_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def finite_norm(v: np.ndarray, axis: int = 1, eps: float = 1e-10) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis)
    return np.maximum(n, eps)


def did_reach_goal(rows: list[dict]) -> bool:
    for row in rows:
        raw = row.get("success_flag")
        if raw not in (None, "") and float(raw) >= 0.5:
            return True
    last = rows[-1]
    final_distance = last.get("episode_final_distance") or last.get("goal_distance")
    return final_distance not in (None, "") and float(final_distance) < 2.0


def compute_episode_metrics(rows: list[dict]) -> dict | None:
    rows = sorted(rows, key=lambda item: int(item["step_index"]))
    if len(rows) < 7:
        return None
    t = np.asarray([float(row["simulation_time_s"]) for row in rows], dtype=float)
    x = np.asarray([float(row["ego_x"]) for row in rows], dtype=float)
    y = np.asarray([float(row["ego_y"]) for row in rows], dtype=float)
    z = np.asarray([float(row["ego_z"]) for row in rows], dtype=float)
    vx = np.asarray([float(row["ego_vx"]) for row in rows], dtype=float)
    vy = np.asarray([float(row["ego_vy"]) for row in rows], dtype=float)
    vz = np.asarray([float(row["ego_vz"]) for row in rows], dtype=float)

    if np.any(~np.isfinite(t)) or np.any(np.diff(t) <= 1e-10):
        return None

    pos = np.column_stack([x, y, z])
    start = pos[0].astype(float)
    last = rows[-1]
    goal = np.asarray([float(last["goal_x"]), float(last["goal_y"]), float(last["goal_z"])], dtype=float)
    goal_vec = goal - start
    goal_dist = float(np.linalg.norm(goal_vec))
    if goal_dist < 1e-8:
        return None
    goal_hat = goal_vec / goal_dist

    diffs = np.diff(pos, axis=0)
    path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
    if path_length < 1e-8:
        return None

    rel = pos - start
    final_proj = float(np.clip(rel[-1] @ goal_hat, 0.0, goal_dist))
    goal_projection_efficiency = final_proj / path_length
    excess_path_ratio = 100.0 * (path_length / goal_dist - 1.0)

    x_s = moving_average(x, SMOOTH_WINDOW)
    y_s = moving_average(y, SMOOTH_WINDOW)
    z_s = moving_average(z, SMOOTH_WINDOW)
    vx_s = moving_average(vx, SMOOTH_WINDOW)
    vy_s = moving_average(vy, SMOOTH_WINDOW)
    vz_s = moving_average(vz, SMOOTH_WINDOW)

    ax_ = gradient_1d(vx_s, t)
    ay_ = gradient_1d(vy_s, t)
    az_ = gradient_1d(vz_s, t)
    jx = gradient_1d(ax_, t)
    jy = gradient_1d(ay_, t)
    jz = gradient_1d(az_, t)
    jerk_mag = np.sqrt(jx**2 + jy**2 + jz**2)
    mean_jerk = finite_mean(jerk_mag)

    rx = gradient_1d(x_s, t)
    ry = gradient_1d(y_s, t)
    rz = gradient_1d(z_s, t)
    rxx = gradient_1d(rx, t)
    ryy = gradient_1d(ry, t)
    rzz = gradient_1d(rz, t)

    r1 = np.column_stack([rx, ry, rz])
    r2 = np.column_stack([rxx, ryy, rzz])
    speed_geom = finite_norm(r1, axis=1)
    curvature = np.linalg.norm(np.cross(r1, r2), axis=1) / np.maximum(speed_geom**3, 1e-10)
    mean_abs_curvature = finite_mean(curvature)

    return {
        "goal_projection_efficiency": goal_projection_efficiency,
        "excess_path_ratio": excess_path_ratio,
        "mean_jerk": mean_jerk,
        "mean_abs_curvature": mean_abs_curvature,
    }


def load_metrics(path: Path, success_only: bool) -> tuple[dict[str, list[dict]], dict[str, int]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            grouped[(str(row["method_key"]), str(row["episode_index"]))].append(row)

    metrics_by_method = {method: [] for method in METHOD_ORDER}
    counts_by_method = {method: 0 for method in METHOD_ORDER}

    for (method_key, _episode_index), rows in grouped.items():
        if method_key not in METHOD_ORDER:
            continue
        if success_only and not did_reach_goal(rows):
            continue
        metrics = compute_episode_metrics(rows)
        if metrics is None:
            continue
        metrics_by_method[method_key].append(metrics)
        counts_by_method[method_key] += 1

    return metrics_by_method, counts_by_method


def style_axis(ax) -> None:
    ax.grid(True, axis="y", color="#d3d3d3", linewidth=0.8, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_boxplot(ax, metric_values: dict[str, np.ndarray], ylabel: str, title: str, counts_by_method: dict[str, int]) -> None:
    methods = [method for method in METHOD_ORDER if metric_values[method].size > 0]
    data = [metric_values[method] for method in methods]
    labels = [f"{METHOD_LABELS[method]}\n(n={counts_by_method[method]})" for method in methods]
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.58,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.3),
        boxprops=dict(linewidth=1.0),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
    )
    for patch, method in zip(bp["boxes"], methods):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.36)
        patch.set_edgecolor(METHOD_COLORS[method])

    rng = np.random.default_rng(0)
    for idx, method in enumerate(methods, start=1):
        vals = np.asarray(metric_values[method], dtype=float)
        vals = vals[np.isfinite(vals)]
        jitter = rng.normal(0.0, 0.045, size=len(vals))
        ax.scatter(
            np.full(len(vals), idx) + jitter,
            vals,
            s=28,
            color=METHOD_COLORS[method],
            alpha=0.72,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )

    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    style_axis(ax)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.trajectory_csv).expanduser().resolve()
    output_pdf = Path(args.output_pdf).expanduser().resolve()
    success_only = args.success_only == "true"

    metrics_by_method, counts_by_method = load_metrics(input_csv, success_only=success_only)
    metric_values = {
        "goal_projection_efficiency": {
            method: np.asarray([item["goal_projection_efficiency"] for item in metrics_by_method[method]], dtype=float)
            for method in METHOD_ORDER
        },
        "excess_path_ratio": {
            method: np.asarray([item["excess_path_ratio"] for item in metrics_by_method[method]], dtype=float)
            for method in METHOD_ORDER
        },
        "mean_jerk": {
            method: np.asarray([item["mean_jerk"] for item in metrics_by_method[method]], dtype=float)
            for method in METHOD_ORDER
        },
        "mean_abs_curvature": {
            method: np.asarray([item["mean_abs_curvature"] for item in metrics_by_method[method]], dtype=float)
            for method in METHOD_ORDER
        },
    }

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.8), constrained_layout=True)
    axes = axes.ravel()
    draw_boxplot(axes[0], metric_values["goal_projection_efficiency"], "Projection efficiency", "(a) Goal-Projection Efficiency", counts_by_method)
    draw_boxplot(axes[1], metric_values["excess_path_ratio"], "Excess path ratio (%)", "(b) Excess Path Ratio", counts_by_method)
    draw_boxplot(axes[2], metric_values["mean_jerk"], "Mean jerk magnitude", "(c) Mean Jerk Magnitude", counts_by_method)
    draw_boxplot(axes[3], metric_values["mean_abs_curvature"], "Mean absolute curvature", "(d) Mean Absolute Curvature", counts_by_method)

    suffix = "successful episodes only" if success_only else "all valid episodes"
    fig.suptitle(f"Ablation trajectory-quality metrics ({suffix})", fontsize=18, fontweight="bold")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_ablation_boxplots] source={input_csv}")
    print(f"[plot_ablation_boxplots] output={output_pdf}")


if __name__ == "__main__":
    main()
