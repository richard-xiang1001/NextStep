#!/bin/bash
# Train orchestrator with GRPO
# Usage: ./scripts/train_orchestrator.sh --rm-checkpoint artifacts/models/rm_ensemble

set -e

# Default values
RM_CHECKPOINT=""
CONFIG="configs/orchestrator/orchestrator_rl_v0.1.0.json"
ENV_CONFIG="configs/env/webshop_v0.1.0.json"
TASK_MANIFEST="configs/env/webshop_task_manifest_v0.1.0.json"
WORKER_POOL="configs/orchestrator/worker_pool_v0.1.0.json"
OUTPUT_DIR="artifacts/models"
SEED=42
MAX_ITERATIONS=100

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --rm-checkpoint)
      RM_CHECKPOINT="$2"
      shift 2
      ;;
    --max-iterations)
      MAX_ITERATIONS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
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

if [ -z "$RM_CHECKPOINT" ]; then
  echo "Error: --rm-checkpoint is required"
  exit 1
fi

echo "=========================================="
echo "Training Orchestrator with GRPO"
echo "RM Checkpoint: $RM_CHECKPOINT"
echo "Max Iterations: $MAX_ITERATIONS"
echo "Seed: $SEED"
echo "=========================================="

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="orchestrator_train_${TIMESTAMP}"
RUN_DIR="$OUTPUT_DIR/$RUN_ID"

mkdir -p "$RUN_DIR"

# Train orchestrator
python -m src.orchestrator.grpo_trainer \
  --config "$CONFIG" \
  --env_config "$ENV_CONFIG" \
  --task_manifest "$TASK_MANIFEST" \
  --worker_pool "$WORKER_POOL" \
  --rm_checkpoint "$RM_CHECKPOINT" \
  --max_iterations "$MAX_ITERATIONS" \
  --seed "$SEED" \
  --output_dir "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/training_log.txt"

echo "Orchestrator training completed: $RUN_DIR"
