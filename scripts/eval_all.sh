#!/bin/bash
# Run all evaluations and generate plots/tables
# Usage: ./scripts/eval_all.sh --checkpoint artifacts/models/orchestrator_final

set -e

# Default values
CHECKPOINT=""
ENV_CONFIG="configs/env/webshop_v0.1.0.json"
TASK_MANIFEST="configs/env/webshop_task_manifest_v0.1.0.json"
WORKER_POOL="configs/orchestrator/worker_pool_v0.1.0.json"
OUTPUT_DIR="results/final"
SPLIT="S_final_test"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [ -z "$CHECKPOINT" ]; then
  echo "Error: --checkpoint is required"
  exit 1
fi

echo "=========================================="
echo "Running Full Evaluation"
echo "Checkpoint: $CHECKPOINT"
echo "Split: $SPLIT"
echo "=========================================="

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="eval_${SPLIT}_${TIMESTAMP}"
RUN_DIR="$OUTPUT_DIR/$RUN_ID"

mkdir -p "$RUN_DIR"

# Run evaluation
python -m src.eval.full_evaluation \
  --checkpoint "$CHECKPOINT" \
  --env_config "$ENV_CONFIG" \
  --task_manifest "$TASK_MANIFEST" \
  --worker_pool "$WORKER_POOL" \
  --split "$SPLIT" \
  --output_dir "$RUN_DIR"

# Compute metrics
python -m src.eval.metrics \
  --trajectory_dir "$RUN_DIR/trajectories" \
  --output_dir "$RUN_DIR/metrics"

# Generate plots
python -m scripts.plotting.generate_plots \
  --metrics_dir "$RUN_DIR/metrics" \
  --output_dir "$RUN_DIR/plots"

# Generate tables
python -m scripts.plotting.generate_tables \
  --metrics_dir "$RUN_DIR/metrics" \
  --output_dir "$RUN_DIR/tables"

echo "Evaluation completed: $RUN_DIR"
echo "Results:"
echo "  - Metrics: $RUN_DIR/metrics"
echo "  - Plots: $RUN_DIR/plots"
echo "  - Tables: $RUN_DIR/tables"
