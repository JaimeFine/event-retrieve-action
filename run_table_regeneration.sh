#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$ROOT_DIR/bruce_code/scripts"
OUTPUT_DIR="$ROOT_DIR/outputs/metrics_regen"
ALL_METHODS_DIR="$OUTPUT_DIR/all_methods"
ERA_DIR="$OUTPUT_DIR/era"
ABLATIONS_DIR="$OUTPUT_DIR/ablations"
REDUCED25_DIR="$OUTPUT_DIR/reduced25"
REDUCED50_DIR="$OUTPUT_DIR/reduced50"
REDUCED25_ALL_METHODS_DIR="$OUTPUT_DIR/reduced25_all_methods"
REDUCED50_ALL_METHODS_DIR="$OUTPUT_DIR/reduced50_all_methods"
REFERENCE_DIR="$ROOT_DIR/datasets/reference_results"

EPISODES="${EPISODES:-10}"
STEPS="${STEPS:-2000}"
SEED="${SEED:-25}"
HEADLESS="${HEADLESS:-true}"
ISAAC_PYTHON="${ISAAC_PYTHON:-../python.sh}"
SYSTEM_PYTHON="${SYSTEM_PYTHON:-python}"
REDUCED25_DATASET="${REDUCED25_DATASET:-}"
REDUCED50_DATASET="${REDUCED50_DATASET:-}"

mkdir -p "$ALL_METHODS_DIR" "$ERA_DIR" "$ABLATIONS_DIR" "$REDUCED25_DIR" "$REDUCED50_DIR" \
  "$REDUCED25_ALL_METHODS_DIR" "$REDUCED50_ALL_METHODS_DIR" "$REFERENCE_DIR"

resolve_dataset_path() {
  local override="$1"
  shift
  if [[ -n "$override" && -f "$override" ]]; then
    echo "$override"
    return 0
  fi
  local candidate
  for candidate in "$@"; do
    if [[ -f "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

REDUCED25_DATASET_RESOLVED="$(
  resolve_dataset_path "$REDUCED25_DATASET" \
    "$ROOT_DIR/datasets/expert_dataset_25.pt" \
    "$ROOT_DIR/datasets/expert_dataset25.pt" \
    "$ROOT_DIR/datasets/25_expert_dataset.pt" \
    "$ROOT_DIR/datasets/dataset_25.pt" \
    "$ROOT_DIR/datasets/reduced25_expert_dataset.pt"
)" || {
  echo "Missing reduced 25% expert dataset .pt file."
  echo "Set REDUCED25_DATASET=/abs/path/to/your_25_dataset.pt or place one in $ROOT_DIR/datasets/"
  exit 1
}

REDUCED50_DATASET_RESOLVED="$(
  resolve_dataset_path "$REDUCED50_DATASET" \
    "$ROOT_DIR/datasets/expert_dataset_50.pt" \
    "$ROOT_DIR/datasets/expert_dataset50.pt" \
    "$ROOT_DIR/datasets/50_expert_dataset.pt" \
    "$ROOT_DIR/datasets/dataset_50.pt" \
    "$ROOT_DIR/datasets/reduced50_expert_dataset.pt"
)" || {
  echo "Missing reduced 50% expert dataset .pt file."
  echo "Set REDUCED50_DATASET=/abs/path/to/your_50_dataset.pt or place one in $ROOT_DIR/datasets/"
  exit 1
}

run_all_methods() {
  local difficulty="$1"
  echo "=== all methods :: $difficulty ==="
  "$ISAAC_PYTHON" "$SCRIPT_DIR/run_all_methods_trajectory_sampling.py" \
    --episodes "$EPISODES" \
    --steps "$STEPS" \
    --seed "$SEED" \
    --headless "$HEADLESS" \
    --difficulty-mode fixed \
    --difficulty-level "$difficulty" \
    --finetuned "$ROOT_DIR/datasets/checkpoints/full/agent_finetuned_99.pt" \
    --bank "$ROOT_DIR/datasets/checkpoints/full/knowledge_bank_snapshot_99.pt" \
    --dataset "$ROOT_DIR/bruce_code/artifacts/expert_dataset.pt" \
    --bc-model-path "$ROOT_DIR/bruce_code/artifacts/bc_policy.pt" \
    --ppo-model-path "$ROOT_DIR/bruce_code/artifacts/ppo_policy.zip" \
    --output-json "$ALL_METHODS_DIR/all_methods_${difficulty}_seed${SEED}_e${EPISODES}_s${STEPS}.json" \
    --output-csv "$ALL_METHODS_DIR/all_methods_${difficulty}_seed${SEED}_e${EPISODES}_s${STEPS}.csv" \
    --output-trajectory-json "$ALL_METHODS_DIR/all_methods_${difficulty}_seed${SEED}_e${EPISODES}_s${STEPS}_trajectory.json" \
    --output-trajectory-csv "$ALL_METHODS_DIR/all_methods_${difficulty}_seed${SEED}_e${EPISODES}_s${STEPS}_trajectory_points.csv"
}

run_era_series() {
  local difficulty="$1"
  shift
  local target_dir="$1"
  shift
  while (($#)); do
    local label="$1"
    local finetuned="$2"
    local bank="$3"
    shift 3
    echo "=== era-only :: $difficulty :: $label ==="
    "$ISAAC_PYTHON" "$SCRIPT_DIR/run_era_only_comparison.py" \
      --episodes "$EPISODES" \
      --steps "$STEPS" \
      --seed "$SEED" \
      --headless "$HEADLESS" \
      --difficulty-mode fixed \
      --difficulty-level "$difficulty" \
      --finetuned "$finetuned" \
      --bank "$bank" \
      --output-json "$target_dir/formal_${difficulty}_${label}.json" \
      --output-csv "$target_dir/formal_${difficulty}_${label}.csv"
  done
}

run_all_methods_reduced() {
  local tag="$1"
  local difficulty="$2"
  local dataset_path="$3"
  local finetuned="$4"
  local bank="$5"
  local output_dir="$6"
  local bc_model_path="$7"

  echo "=== reduced bc_il only :: $tag :: $difficulty ==="
  "$ISAAC_PYTHON" "$SCRIPT_DIR/run_all_methods_trajectory_sampling.py" \
    --episodes "$EPISODES" \
    --steps "$STEPS" \
    --seed "$SEED" \
    --headless "$HEADLESS" \
    --difficulty-mode fixed \
    --difficulty-level "$difficulty" \
    --finetuned "$finetuned" \
    --bank "$bank" \
    --dataset "$dataset_path" \
    --methods bc_il \
    --bc-model-path "$bc_model_path" \
    --output-json "$output_dir/all_methods_${tag}_${difficulty}_seed${SEED}_e${EPISODES}_s${STEPS}.json" \
    --output-csv "$output_dir/all_methods_${tag}_${difficulty}_seed${SEED}_e${EPISODES}_s${STEPS}.csv" \
    --output-trajectory-json "$output_dir/all_methods_${tag}_${difficulty}_seed${SEED}_e${EPISODES}_s${STEPS}_trajectory.json" \
    --output-trajectory-csv "$output_dir/all_methods_${tag}_${difficulty}_seed${SEED}_e${EPISODES}_s${STEPS}_trajectory_points.csv"
}

extract_bc_rows() {
  local source_csv="$1"
  local target_csv="$2"
  echo "=== extract bc_il rows :: $(basename "$target_csv") ==="
  awk -F, 'NR==1 || $4=="bc_il"' "$source_csv" > "$target_csv"
}

run_all_methods medium
run_all_methods extreme

run_era_series medium "$ERA_DIR" \
  "0" "$ROOT_DIR/datasets/checkpoints/full/agent_pretrained.pt" "$ROOT_DIR/datasets/checkpoints/full/knowledge_bank_snapshot_pretrained.pt" \
  "199" "$ROOT_DIR/datasets/checkpoints/full/agent_finetuned_199.pt" "$ROOT_DIR/datasets/checkpoints/full/knowledge_bank_snapshot_199.pt" \
  "299" "$ROOT_DIR/datasets/checkpoints/full/agent_finetuned_299.pt" "$ROOT_DIR/datasets/checkpoints/full/knowledge_bank_snapshot_299.pt" \
  "399" "$ROOT_DIR/datasets/checkpoints/full/agent_finetuned_399.pt" "$ROOT_DIR/datasets/checkpoints/full/knowledge_bank_snapshot_399.pt" \
  "499" "$ROOT_DIR/datasets/checkpoints/full/agent_finetuned_499.pt" "$ROOT_DIR/datasets/checkpoints/full/knowledge_bank_snapshot_499.pt" \
  "599" "$ROOT_DIR/datasets/checkpoints/full/agent_finetuned_599.pt" "$ROOT_DIR/datasets/checkpoints/full/knowledge_bank_snapshot_599.pt" \
  "699" "$ROOT_DIR/datasets/checkpoints/full/agent_finetuned_699.pt" "$ROOT_DIR/datasets/checkpoints/full/knowledge_bank_snapshot_699.pt"

run_era_series extreme "$ERA_DIR" \
  "0" "$ROOT_DIR/datasets/checkpoints/full/agent_pretrained.pt" "$ROOT_DIR/datasets/checkpoints/full/knowledge_bank_snapshot_pretrained.pt" \
  "199" "$ROOT_DIR/datasets/checkpoints/full/agent_finetuned_199.pt" "$ROOT_DIR/datasets/checkpoints/full/knowledge_bank_snapshot_199.pt" \
  "299" "$ROOT_DIR/datasets/checkpoints/full/agent_finetuned_299.pt" "$ROOT_DIR/datasets/checkpoints/full/knowledge_bank_snapshot_299.pt" \
  "399" "$ROOT_DIR/datasets/checkpoints/full/agent_finetuned_399.pt" "$ROOT_DIR/datasets/checkpoints/full/knowledge_bank_snapshot_399.pt" \
  "499" "$ROOT_DIR/datasets/checkpoints/full/agent_finetuned_499.pt" "$ROOT_DIR/datasets/checkpoints/full/knowledge_bank_snapshot_499.pt" \
  "599" "$ROOT_DIR/datasets/checkpoints/full/agent_finetuned_599.pt" "$ROOT_DIR/datasets/checkpoints/full/knowledge_bank_snapshot_599.pt" \
  "699" "$ROOT_DIR/datasets/checkpoints/full/agent_finetuned_699.pt" "$ROOT_DIR/datasets/checkpoints/full/knowledge_bank_snapshot_699.pt"

run_era_series medium "$ABLATIONS_DIR" \
  "no_phys" "$ROOT_DIR/datasets/checkpoints/ablations/no_phys_finetuned.pt" "$ROOT_DIR/datasets/checkpoints/ablations/no_phys_bank.pt" \
  "no_bayesian" "$ROOT_DIR/datasets/checkpoints/ablations/no_bay_finetuned.pt" "$ROOT_DIR/datasets/checkpoints/ablations/no_bay_bank.pt"

run_era_series extreme "$ABLATIONS_DIR" \
  "no_phys" "$ROOT_DIR/datasets/checkpoints/ablations/no_phys_finetuned.pt" "$ROOT_DIR/datasets/checkpoints/ablations/no_phys_bank.pt" \
  "no_bayesian" "$ROOT_DIR/datasets/checkpoints/ablations/no_bay_finetuned.pt" "$ROOT_DIR/datasets/checkpoints/ablations/no_bay_bank.pt"

run_era_series medium "$REDUCED25_DIR" \
  "99" "$ROOT_DIR/datasets/checkpoints/reduced25/agent_finetuned_99.pt" "$ROOT_DIR/datasets/checkpoints/reduced25/knowledge_bank_snapshot_99.pt"

run_era_series extreme "$REDUCED25_DIR" \
  "99" "$ROOT_DIR/datasets/checkpoints/reduced25/agent_finetuned_99.pt" "$ROOT_DIR/datasets/checkpoints/reduced25/knowledge_bank_snapshot_99.pt"

run_era_series medium "$REDUCED50_DIR" \
  "99" "$ROOT_DIR/datasets/checkpoints/reduced50/agent_finetuned_99.pt" "$ROOT_DIR/datasets/checkpoints/reduced50/knowledge_bank_snapshot_99.pt"

run_era_series extreme "$REDUCED50_DIR" \
  "99" "$ROOT_DIR/datasets/checkpoints/reduced50/agent_finetuned_99.pt" "$ROOT_DIR/datasets/checkpoints/reduced50/knowledge_bank_snapshot_99.pt"

run_all_methods_reduced \
  "25" "medium" "$REDUCED25_DATASET_RESOLVED" \
  "$ROOT_DIR/datasets/checkpoints/reduced25/agent_finetuned_99.pt" \
  "$ROOT_DIR/datasets/checkpoints/reduced25/knowledge_bank_snapshot_99.pt" \
  "$REDUCED25_ALL_METHODS_DIR" \
  "$ROOT_DIR/bruce_code/artifacts/bc_policy_25.pt"

run_all_methods_reduced \
  "25" "extreme" "$REDUCED25_DATASET_RESOLVED" \
  "$ROOT_DIR/datasets/checkpoints/reduced25/agent_finetuned_99.pt" \
  "$ROOT_DIR/datasets/checkpoints/reduced25/knowledge_bank_snapshot_99.pt" \
  "$REDUCED25_ALL_METHODS_DIR" \
  "$ROOT_DIR/bruce_code/artifacts/bc_policy_25.pt"

run_all_methods_reduced \
  "50" "medium" "$REDUCED50_DATASET_RESOLVED" \
  "$ROOT_DIR/datasets/checkpoints/reduced50/agent_finetuned_99.pt" \
  "$ROOT_DIR/datasets/checkpoints/reduced50/knowledge_bank_snapshot_99.pt" \
  "$REDUCED50_ALL_METHODS_DIR" \
  "$ROOT_DIR/bruce_code/artifacts/bc_policy_50.pt"

run_all_methods_reduced \
  "50" "extreme" "$REDUCED50_DATASET_RESOLVED" \
  "$ROOT_DIR/datasets/checkpoints/reduced50/agent_finetuned_99.pt" \
  "$ROOT_DIR/datasets/checkpoints/reduced50/knowledge_bank_snapshot_99.pt" \
  "$REDUCED50_ALL_METHODS_DIR" \
  "$ROOT_DIR/bruce_code/artifacts/bc_policy_50.pt"

extract_bc_rows \
  "$REDUCED25_ALL_METHODS_DIR/all_methods_25_medium_seed${SEED}_e${EPISODES}_s${STEPS}.csv" \
  "$REFERENCE_DIR/bc_expert25_medium_e${EPISODES}.csv"

extract_bc_rows \
  "$REDUCED25_ALL_METHODS_DIR/all_methods_25_extreme_seed${SEED}_e${EPISODES}_s${STEPS}.csv" \
  "$REFERENCE_DIR/bc_expert25_extreme_e${EPISODES}.csv"

extract_bc_rows \
  "$REDUCED50_ALL_METHODS_DIR/all_methods_50_medium_seed${SEED}_e${EPISODES}_s${STEPS}.csv" \
  "$REFERENCE_DIR/bc_expert50_medium_e${EPISODES}.csv"

extract_bc_rows \
  "$REDUCED50_ALL_METHODS_DIR/all_methods_50_extreme_seed${SEED}_e${EPISODES}_s${STEPS}.csv" \
  "$REFERENCE_DIR/bc_expert50_extreme_e${EPISODES}.csv"

"$SYSTEM_PYTHON" "$SCRIPT_DIR/build_metrics_table_csv.py" \
  --output-csv "$OUTPUT_DIR/unified_metrics_table.csv"

echo "Finished. Table-ready CSV:"
echo "  $OUTPUT_DIR/unified_metrics_table.csv"
