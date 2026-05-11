Trajectory Evaluation and Plotting Bundle

This repository is the standalone bundle used to regenerate the corrected ERA evaluation, paper tables, and paper figures.

It contains:
- the runnable local evaluation code
- the bundled checkpoints and model artifacts
- regenerated CSV / JSON benchmark outputs
- regenerated figures used in the paper
- deployment-side scalability analysis utilities for the Knowledge Bank appendix
- a small set of supporting notes that were useful during analysis

The bundle is meant to be deliverable on its own, without depending on the original working repositories.

## Repository Overview

Current top-level structure:

- `bruce_code/`
  Main local codebase for ERA, baselines, simulation wrappers, plotting, and table building.
- `datasets/`
  Bundled checkpoints and model files used by the evaluation scripts.
- `outputs/`
  Generated evaluation results, trajectory exports, and figure outputs.
- `src/deployment/`
  Deployment-oriented utilities, including latency, memory, and Knowledge Bank scalability analysis scripts.
- `boxplots.pdf`
  Regenerated comparative trajectory-quality boxplot figure.
- `run_table_regeneration.sh`
  End-to-end regeneration script for the corrected metrics pipeline.
- `TEST_RESULTS_CSV_EXPLANATION.md`
  Field-level explanation of the exported per-episode CSV format.
- `training_results.csv`
  Training summary data kept for reference.
- `why-we-freeze-gamma.md`
  Short design note kept for reference.
- `README.txt`
  Older short bundle note; this `README.md` is the main up-to-date guide.

## What This Bundle Is For

This bundle supports four main tasks:

1. Comparative evaluation across five methods:
   - `ERA`
   - `VPF Expert`
   - `BC-IL`
   - `PPO`
   - `Acados`
2. ERA checkpoint scaling across knowledge-bank sizes.
3. Ablation evaluation for:
   - `ERA-100`
   - `ERA w/o R_phys`
   - `ERA w/o CWS`
4. Regeneration of paper-ready tables and figures from exported results.
5. Deployment-facing scalability analysis of Knowledge Bank memory and retrieval latency.

## Important Interpretation Notes

These points are essential for reading the outputs correctly.

- The stale-intruder issue from the older evaluation was fixed in this copied bundle.
- The exported field `success_rate` is not a binary goal-reached metric.
  It is a dense per-step safe-progress proxy.
- Actual episode completion is tracked by `success_flag`.
- The rebuilt table pipeline therefore reports both:
  - `success_rate`
  - `goal_reach_rate`
- In the corrected benchmark, `R_phys` appears weak.
- In the checked extreme ablation export, `ERA-100` and `ERA w/o R_phys` are point-for-point identical in trajectory space.
- `CWS` has a more visible effect on deployed trajectories than `R_phys`.

## What To Trust For The Paper

If you only need the corrected outputs that support the current paper interpretation, use these locations first.

### Corrected tables

- `outputs/metrics_regen/unified_metrics_table.csv`
- `outputs/metrics_regen/unified_metrics_table_goal_reached_only.csv`

These are the main regenerated summary tables.

### Corrected ablation outputs

- `outputs/ablation_runs/ablation_era_only_extreme_seed25_e10_s2000.csv`
- `outputs/ablation_runs/ablation_era_only_extreme_seed25_e10_s2000_trajectory.json`
- `outputs/ablation_runs/ablation_3d_sample9.pdf`
- `outputs/ablation_runs/ablation_better_boxplots.pdf`
- `outputs/ablation_runs/ablation_combined_horizontal.pdf`

### Corrected comparative figures

- `outputs/plots/stacked_distance/sample_1_stacked_distance_largefont.pdf`
- `outputs/plots/stacked_distance/sample_1_stacked_distance_compact.png`
- `boxplots.pdf`

## Outputs Folder Guide

The `outputs/` folder contains a mix of final regenerated outputs and smaller exploratory exports. The most important subfolders are:

- `outputs/metrics_regen/`
  Main corrected benchmark outputs used to rebuild the paper tables.
- `outputs/ablation_runs/`
  Main corrected ablation exports and ablation figures.
- `outputs/plots/stacked_distance/`
  Stacked distance-to-intruder figure outputs.
- `outputs/bc_limited_regen/`
  Reduced-data BC comparison exports used by the unified table builders.
- `outputs/medium-plots/`
  Medium-difficulty per-sample visualization outputs.
- `outputs/extreme-plots/`
  Extreme-difficulty per-sample visualization outputs.

There are also some root-level files in `outputs/` such as:

- `all_methods_*`
- `formal_*_trajectory.*`

These are still useful for visualization and inspection, but the corrected paper-summary tables should be taken from `outputs/metrics_regen/`.

## Main Code Locations

### Evaluation scripts

- `bruce_code/scripts/run_all_methods_trajectory_sampling.py`
  Runs the five-method comparative benchmark and exports per-episode plus trajectory data.

- `bruce_code/scripts/run_era_only_comparison.py`
  Runs ERA-only evaluation for a specified checkpoint.

- `bruce_code/scripts/run_ablation_trajectory_sampling.py`
  Runs the ablation trajectory export for `ERA-100`, `ERA w/o R_phys`, and `ERA w/o CWS`.

### Plotting scripts

- `bruce_code/scripts/plot_trajectory_samples.py`
  Per-method 3D and distance-time plots plus combined 3D multi-method overlays.

- `bruce_code/scripts/plot_distance_time_stacked.py`
  Five-panel stacked distance-to-intruder figure.

- `bruce_code/scripts/plot_boxplots_from_outputs.py`
  Comparative boxplots from successful episodes only.

- `bruce_code/scripts/plot_ablation_boxplots.py`
  Ablation boxplots.

- `bruce_code/scripts/plot_ablation_3d.py`
  Ablation 3D trajectory figure.

### Deployment appendix utilities

- `src/deployment/scalability_analysis/benchmark_kb_scalability.py`
  Benchmarks Knowledge Bank growth by loading saved snapshots and measuring payload memory plus retrieval latency.

- `src/deployment/scalability_analysis/plot_kb_scalability.py`
  Builds the dual-axis scalability figure used for the deployment / appendix discussion.

- `src/deployment/scalability_analysis/outputs/`
  Default location for the generated CSV, summary note, and scalability figures.

### Table builders

- `bruce_code/scripts/build_metrics_table_csv.py`
  Builds the corrected unified metrics table.

- `bruce_code/scripts/build_metrics_table_goal_reached_only_csv.py`
  Builds the goal-reached-only version of the metrics table.

## Bundled Models And Checkpoints

Important model artifacts live in:

- `datasets/checkpoints/full/`
  Full-data ERA checkpoints including pretrained, Ep. 100, and later checkpoints.
- `datasets/checkpoints/ablations/`
  Ablation checkpoints for `no_phys` and `no_bay` / no-CWS runs.
- `datasets/checkpoints/reduced25/`
  ERA checkpoints for the 25% reduced-data setting.
- `datasets/checkpoints/reduced50/`
  ERA checkpoints for the 50% reduced-data setting.

Additional bundled runtime artifacts include:

- `bruce_code/artifacts/expert_dataset.pt`
- `bruce_code/artifacts/bc_policy.pt`
- `bruce_code/artifacts/ppo_policy.zip`

## Runtime Requirements

Two Python environments are usually involved.

- Isaac Sim Python:
  needed for scripts that instantiate the simulator
- Standard Python:
  enough for table generation and plotting from already-exported CSV / JSON files

Use Isaac Sim Python for:

- `bruce_code/scripts/run_all_methods_trajectory_sampling.py`
- `bruce_code/scripts/run_era_only_comparison.py`
- `bruce_code/scripts/run_ablation_trajectory_sampling.py`

Use standard Python for:

- `bruce_code/scripts/build_metrics_table_csv.py`
- `bruce_code/scripts/build_metrics_table_goal_reached_only_csv.py`
- the plotting scripts that only read exported files
- `src/deployment/scalability_analysis/benchmark_kb_scalability.py`
- `src/deployment/scalability_analysis/plot_kb_scalability.py`

## Typical Workflow

### 1. Comparative benchmark

Example:

```powershell
<isaac-sim-python> bruce_code\scripts\run_all_methods_trajectory_sampling.py `
  --episodes 10 `
  --steps 2000 `
  --seed 25 `
  --difficulty-mode fixed `
  --difficulty-level medium `
  --headless true
```

Repeat with `--difficulty-level extreme` for the extreme benchmark.

### 2. ERA-only checkpoint evaluation

Example:

```powershell
<isaac-sim-python> bruce_code\scripts\run_era_only_comparison.py `
  --episodes 10 `
  --steps 2000 `
  --seed 25 `
  --difficulty-mode fixed `
  --difficulty-level extreme `
  --headless true `
  --finetuned datasets\checkpoints\full\agent_finetuned_699.pt `
  --bank datasets\checkpoints\full\knowledge_bank_snapshot_699.pt
```

### 3. Ablation export

Example:

```powershell
<isaac-sim-python> bruce_code\scripts\run_ablation_trajectory_sampling.py `
  --episodes 10 `
  --steps 2000 `
  --seed 25 `
  --difficulty-mode fixed `
  --difficulty-level extreme `
  --variants era_100,no_phys,no_cws
```

### 4. Rebuild corrected tables

Main pipeline:

```bash
./run_table_regeneration.sh
```

Direct goal-reached-only table rebuild:

```powershell
python bruce_code\scripts\build_metrics_table_goal_reached_only_csv.py
```

### 5. Rebuild figures

Stacked distance figure:

```powershell
python bruce_code\scripts\plot_distance_time_stacked.py `
  --trajectory-json outputs\all_methods_medium_seed25_e3_s2000_trajectory.json `
  --sample-index 1 `
  --difficulty-label medium
```

Combined 3D comparative figure:

```powershell
python bruce_code\scripts\plot_trajectory_samples.py `
  --trajectory-json outputs\all_methods_medium_seed25_e3_s2000_trajectory.json `
  --plot-mode combined-3d `
  --sample-index 1
```

Comparative boxplots:

```powershell
python bruce_code\scripts\plot_boxplots_from_outputs.py `
  --difficulty extreme `
  --success-only true `
  --output-pdf boxplots.pdf
```

### 6. Regenerate deployment scalability appendix artifacts

This workflow benchmarks the deployed Knowledge Bank retrieval kernel directly
from the bundled snapshots in `datasets/checkpoints/full/`.

Jetson / CUDA example:

```powershell
python src\deployment\scalability_analysis\benchmark_kb_scalability.py --device gpu
python src\deployment\scalability_analysis\plot_kb_scalability.py
```

Default outputs:

- `src/deployment/scalability_analysis/outputs/era_kb_scalability.csv`
- `src/deployment/scalability_analysis/outputs/era_kb_scalability_summary.md`
- `src/deployment/scalability_analysis/outputs/era_kb_scalability_dual_axis.pdf`
- `src/deployment/scalability_analysis/outputs/era_kb_scalability_dual_axis.png`

## Supporting Reference Files

These files are kept because they were useful during paper analysis or reproduction:

- `training_results.csv`
  training summary reference
- `bruce_code/training_results.csv`
  code-local copy of training results
- `why-we-freeze-gamma.md`
  short note about one training / evaluation design decision
- `TEST_RESULTS_CSV_EXPLANATION.md`
  explanation of output CSV fields and intended interpretation

## Notes On Legacy Or Exploratory Artifacts

Some files in this repository are convenience artifacts rather than core paper deliverables, for example:

- `outputs/medium-plots/`
- `outputs/extreme-plots/`
- root-level sampled `outputs/all_methods_*_e3_*` files
- older short notes such as `README.txt`

They are still useful for inspection, visual checks, and traceability, but the corrected paper-facing summaries should come from the trusted locations listed above.

## Submission Notes

This repository is intended to accompany the manuscript submission.

The most important items to preserve are:

- `bruce_code/`
- `datasets/`
- `outputs/metrics_regen/`
- `outputs/ablation_runs/`
- `outputs/plots/stacked_distance/`
- `boxplots.pdf`
- `README.md`

The `.gitignore` file is set up to avoid uploading Python cache files and other local noise.
