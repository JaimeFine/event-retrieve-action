from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = ROOT / "outputs"
REFERENCE_DIR = ROOT / "outputs" / "bc_limited_regen"


@dataclass(frozen=True)
class RowSpec:
    difficulty: str
    group_label: str
    method_label: str
    episodes_label: str
    csv_path: Path
    bank_size: str = "N/A"
    method_key: str | None = None


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _did_reach_goal(row: dict) -> bool:
    raw_success_flag = row.get("success_flag")
    if raw_success_flag not in (None, ""):
        return float(raw_success_flag) >= 0.5
    final_distance = row.get("final_distance")
    return final_distance not in (None, "") and float(final_distance) < 2.0


def load_csv_metrics(path: Path, method_key: str | None = None) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    if method_key is not None:
        rows = [row for row in rows if str(row.get("method_key", "")) == method_key]
    if not rows:
        raise ValueError(f"No rows found in {path} for method_key={method_key!r}")

    reached_rows = [row for row in rows if _did_reach_goal(row)]
    goal_reach_rate = len(reached_rows) / len(rows)
    if not reached_rows:
        return {
            "goal_reach_rate": goal_reach_rate,
            "reached_episode_count": 0,
            "total_episode_count": len(rows),
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "warning_rate": 0.0,
            "avg_effective_steps": 0.0,
            "avg_reaction_time_ms": 0.0,
            "avg_final_distance": 0.0,
        }

    return {
        "goal_reach_rate": goal_reach_rate,
        "reached_episode_count": len(reached_rows),
        "total_episode_count": len(rows),
        "success_rate": _mean(float(row["success_rate"]) for row in reached_rows),
        "collision_rate": _mean(float(row["collision_rate"]) for row in reached_rows),
        "warning_rate": _mean(float(row["warning_rate"]) for row in reached_rows),
        "avg_effective_steps": _mean(float(row["effective_steps"]) for row in reached_rows),
        "avg_reaction_time_ms": _mean(float(row["avg_reaction_time_ms"]) for row in reached_rows),
        "avg_final_distance": _mean(float(row["final_distance"]) for row in reached_rows),
    }


def build_specs() -> list[RowSpec]:
    medium = "medium"
    extreme = "extreme"
    return [
        RowSpec(medium, "medium", "ERA", "0", OUTPUTS_DIR / "metrics_regen" / "era" / "formal_medium_0.csv", bank_size="27075", method_key="era"),
        RowSpec(medium, "medium", "ERA", "100", OUTPUTS_DIR / "metrics_regen" / "all_methods" / "all_methods_medium_seed25_e10_s2000.csv", bank_size="30650", method_key="era"),
        RowSpec(medium, "medium", "ERA", "200", OUTPUTS_DIR / "metrics_regen" / "era" / "formal_medium_199.csv", bank_size="None", method_key="era"),
        RowSpec(medium, "medium", "ERA", "300", OUTPUTS_DIR / "metrics_regen" / "era" / "formal_medium_299.csv", bank_size="37521", method_key="era"),
        RowSpec(medium, "medium", "ERA", "400", OUTPUTS_DIR / "metrics_regen" / "era" / "formal_medium_399.csv", bank_size="None", method_key="era"),
        RowSpec(medium, "medium", "ERA", "500", OUTPUTS_DIR / "metrics_regen" / "era" / "formal_medium_499.csv", bank_size="43677", method_key="era"),
        RowSpec(medium, "medium", "ERA", "600", OUTPUTS_DIR / "metrics_regen" / "era" / "formal_medium_599.csv", bank_size="None", method_key="era"),
        RowSpec(medium, "medium", "ERA", "700", OUTPUTS_DIR / "metrics_regen" / "era" / "formal_medium_699.csv", bank_size="49545", method_key="era"),
        RowSpec(medium, "medium", "ERA w/o R_phys", "100", OUTPUTS_DIR / "metrics_regen" / "ablations" / "formal_medium_no_phys.csv", bank_size="30650", method_key="era"),
        RowSpec(medium, "medium", "ERA w/o CBS", "100", OUTPUTS_DIR / "metrics_regen" / "ablations" / "formal_medium_no_bayesian.csv", bank_size="28034", method_key="era"),
        RowSpec(medium, "medium", "VPF Expert", "-", OUTPUTS_DIR / "metrics_regen" / "all_methods" / "all_methods_medium_seed25_e10_s2000.csv", bank_size="N/A", method_key="vpf"),
        RowSpec(medium, "medium", "BC-IL", "-", OUTPUTS_DIR / "metrics_regen" / "all_methods" / "all_methods_medium_seed25_e10_s2000.csv", bank_size="N/A", method_key="bc_il"),
        RowSpec(medium, "medium", "Acados", "-", OUTPUTS_DIR / "metrics_regen" / "all_methods" / "all_methods_medium_seed25_e10_s2000.csv", bank_size="N/A", method_key="acados"),
        RowSpec(medium, "medium", "PPO", "-", OUTPUTS_DIR / "metrics_regen" / "all_methods" / "all_methods_medium_seed25_e10_s2000.csv", bank_size="N/A", method_key="ppo"),
        RowSpec(medium, "medium", "ERA 25%", "100", OUTPUTS_DIR / "metrics_regen" / "reduced25" / "formal_medium_99.csv", bank_size="10064", method_key="era"),
        RowSpec(
            medium,
            "medium",
            "BC-IL 25%",
            "-",
            REFERENCE_DIR / "reduced25" / "all_methods_25_medium_seed25_e10_s2000.csv",
            bank_size="N/A",
            method_key="bc_il",
        ),
        RowSpec(medium, "medium", "ERA 50%", "100", OUTPUTS_DIR / "metrics_regen" / "reduced50" / "formal_medium_99.csv", bank_size="15812", method_key="era"),
        RowSpec(
            medium,
            "medium",
            "BC-IL 50%",
            "-",
            REFERENCE_DIR / "reduced50" / "all_methods_50_medium_seed25_e10_s2000.csv",
            bank_size="N/A",
            method_key="bc_il",
        ),
        RowSpec(extreme, "extreme", "ERA", "0", OUTPUTS_DIR / "metrics_regen" / "era" / "formal_extreme_0.csv", bank_size="27075", method_key="era"),
        RowSpec(extreme, "extreme", "ERA", "100", OUTPUTS_DIR / "metrics_regen" / "all_methods" / "all_methods_extreme_seed25_e10_s2000.csv", bank_size="30650", method_key="era"),
        RowSpec(extreme, "extreme", "ERA", "200", OUTPUTS_DIR / "metrics_regen" / "era" / "formal_extreme_199.csv", bank_size="None", method_key="era"),
        RowSpec(extreme, "extreme", "ERA", "300", OUTPUTS_DIR / "metrics_regen" / "era" / "formal_extreme_299.csv", bank_size="37521", method_key="era"),
        RowSpec(extreme, "extreme", "ERA", "400", OUTPUTS_DIR / "metrics_regen" / "era" / "formal_extreme_399.csv", bank_size="None", method_key="era"),
        RowSpec(extreme, "extreme", "ERA", "500", OUTPUTS_DIR / "metrics_regen" / "era" / "formal_extreme_499.csv", bank_size="43677", method_key="era"),
        RowSpec(extreme, "extreme", "ERA", "600", OUTPUTS_DIR / "metrics_regen" / "era" / "formal_extreme_599.csv", bank_size="None", method_key="era"),
        RowSpec(extreme, "extreme", "ERA", "700", OUTPUTS_DIR / "metrics_regen" / "era" / "formal_extreme_699.csv", bank_size="49545", method_key="era"),
        RowSpec(extreme, "extreme", "ERA w/o R_phys", "100", OUTPUTS_DIR / "metrics_regen" / "ablations" / "formal_extreme_no_phys.csv", bank_size="30650", method_key="era"),
        RowSpec(extreme, "extreme", "ERA w/o CBS", "100", OUTPUTS_DIR / "metrics_regen" / "ablations" / "formal_extreme_no_bayesian.csv", bank_size="28034", method_key="era"),
        RowSpec(extreme, "extreme", "VPF Expert", "-", OUTPUTS_DIR / "metrics_regen" / "all_methods" / "all_methods_extreme_seed25_e10_s2000.csv", bank_size="N/A", method_key="vpf"),
        RowSpec(extreme, "extreme", "BC-IL", "-", OUTPUTS_DIR / "metrics_regen" / "all_methods" / "all_methods_extreme_seed25_e10_s2000.csv", bank_size="N/A", method_key="bc_il"),
        RowSpec(extreme, "extreme", "Acados", "-", OUTPUTS_DIR / "metrics_regen" / "all_methods" / "all_methods_extreme_seed25_e10_s2000.csv", bank_size="N/A", method_key="acados"),
        RowSpec(extreme, "extreme", "PPO", "-", OUTPUTS_DIR / "metrics_regen" / "all_methods" / "all_methods_extreme_seed25_e10_s2000.csv", bank_size="N/A", method_key="ppo"),
        RowSpec(extreme, "extreme", "ERA 25%", "100", OUTPUTS_DIR / "metrics_regen" / "reduced25" / "formal_extreme_99.csv", bank_size="10064", method_key="era"),
        RowSpec(
            extreme,
            "extreme",
            "BC-IL 25%",
            "-",
            REFERENCE_DIR / "reduced25" / "all_methods_25_extreme_seed25_e10_s2000.csv",
            bank_size="N/A",
            method_key="bc_il",
        ),
        RowSpec(extreme, "extreme", "ERA 50%", "100", OUTPUTS_DIR / "metrics_regen" / "reduced50" / "formal_extreme_99.csv", bank_size="15812", method_key="era"),
        RowSpec(
            extreme,
            "extreme",
            "BC-IL 50%",
            "-",
            REFERENCE_DIR / "reduced50" / "all_methods_50_extreme_seed25_e10_s2000.csv",
            bank_size="N/A",
            method_key="bc_il",
        ),
    ]


def resolve_metrics(spec: RowSpec) -> dict:
    return load_csv_metrics(spec.csv_path, method_key=spec.method_key)


def build_rows() -> list[dict]:
    rows: list[dict] = []
    for spec in build_specs():
        metrics = resolve_metrics(spec)
        rows.append(
            {
                "difficulty": spec.difficulty,
                "group_label": spec.group_label,
                "method": spec.method_label,
                "episodes": spec.episodes_label,
                "goal_reach_rate": metrics["goal_reach_rate"],
                "reached_episode_count": metrics["reached_episode_count"],
                "total_episode_count": metrics["total_episode_count"],
                "success_rate_goal_reached_only": metrics["success_rate"],
                "collision_rate_goal_reached_only": metrics["collision_rate"],
                "warning_rate_goal_reached_only": metrics["warning_rate"],
                "avg_steps_goal_reached_only": metrics["avg_effective_steps"],
                "avg_final_distance_goal_reached_only": metrics["avg_final_distance"],
                "reaction_time_ms_goal_reached_only": metrics["avg_reaction_time_ms"],
                "bank_size": spec.bank_size,
                "source_kind": "csv_goal_reached_only",
                "source_path": str(spec.csv_path),
            }
        )
    return rows


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a table-ready CSV averaging metrics only over episodes that reached the goal."
    )
    parser.add_argument(
        "--output-csv",
        default=str(OUTPUTS_DIR / "metrics_regen" / "unified_metrics_table_goal_reached_only.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows()
    write_csv(Path(args.output_csv).expanduser().resolve(), rows)
    print(f"[build_metrics_table_goal_reached_only_csv] rows={len(rows)}")
    print(f"[build_metrics_table_goal_reached_only_csv] output={Path(args.output_csv).expanduser().resolve()}")


if __name__ == "__main__":
    main()
