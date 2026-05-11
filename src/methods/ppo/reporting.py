from __future__ import annotations

import csv
import json
from pathlib import Path


def _normalize_series(points: list[dict], y_key: str) -> list[tuple[float, float]]:
    data = []
    for point in points:
        if y_key not in point or "timesteps" not in point:
            continue
        data.append((float(point["timesteps"]), float(point[y_key])))
    return data


def _build_polyline(points: list[tuple[float, float]], width: int, height: int, padding: int) -> str:
    if not points:
        return ""

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if abs(max_x - min_x) < 1e-6:
        max_x = min_x + 1.0
    if abs(max_y - min_y) < 1e-6:
        max_y = min_y + 1.0

    coords = []
    for x, y in points:
        px = padding + (x - min_x) / (max_x - min_x) * (width - padding * 2)
        py = height - padding - (y - min_y) / (max_y - min_y) * (height - padding * 2)
        coords.append(f"{px:.2f},{py:.2f}")
    return " ".join(coords)


def _write_svg(path: Path, title: str, series: list[tuple[float, float]], color: str) -> None:
    width = 960
    height = 420
    padding = 48
    polyline = _build_polyline(series, width=width, height=height, padding=padding)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f8fafc"/>
  <line x1="{padding}" y1="{height-padding}" x2="{width-padding}" y2="{height-padding}" stroke="#94a3b8" stroke-width="2"/>
  <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height-padding}" stroke="#94a3b8" stroke-width="2"/>
  <text x="{padding}" y="28" fill="#0f172a" font-size="24" font-family="Segoe UI, Arial, sans-serif">{title}</text>
  <polyline fill="none" stroke="{color}" stroke-width="3" points="{polyline}"/>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_training_curves(training_history: dict, output_dir: str | Path, run_name: str) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    history_json = output_path / f"{run_name}_training_metrics.json"
    reward_csv = output_path / f"{run_name}_reward_curve.csv"
    convergence_csv = output_path / f"{run_name}_convergence_curve.csv"
    reward_svg = output_path / f"{run_name}_reward_curve.svg"
    convergence_svg = output_path / f"{run_name}_convergence_curve.svg"

    history_json.write_text(json.dumps(training_history, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(reward_csv, training_history.get("reward_curve", []))
    _write_csv(convergence_csv, training_history.get("convergence_curve", []))

    reward_points = _normalize_series(training_history.get("reward_curve", []), y_key="reward")
    convergence_rows = training_history.get("convergence_curve", [])
    convergence_key = "train/loss"
    if not any(convergence_key in row for row in convergence_rows):
        for fallback in ("train/value_loss", "train/policy_gradient_loss", "rollout/ep_rew_mean"):
            if any(fallback in row for row in convergence_rows):
                convergence_key = fallback
                break
    convergence_points = _normalize_series(convergence_rows, y_key=convergence_key)

    _write_svg(reward_svg, "PPO Reward Curve", reward_points, color="#0f766e")
    _write_svg(convergence_svg, f"PPO Convergence Curve ({convergence_key})", convergence_points, color="#b45309")

    return {
        "metrics_json": str(history_json),
        "reward_curve_csv": str(reward_csv),
        "convergence_curve_csv": str(convergence_csv),
        "reward_curve_svg": str(reward_svg),
        "convergence_curve_svg": str(convergence_svg),
        "convergence_metric": convergence_key,
    }
