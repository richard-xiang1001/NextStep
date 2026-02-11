#!/bin/bash
# Train reward model from offline trajectories
# Usage: ./scripts/train_rm.sh --config configs/rm/rm_training_v0.1.0.json

set -e

# Default values
CONFIG="configs/rm/rm_training_v0.1.0.json"
ENV_CONFIG="configs/env/webshop_v0.1.0.json"
TASK_MANIFEST="configs/env/webshop_task_manifest_v0.1.0.json"
OUTPUT_DIR="artifacts/models"
SEED=42

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
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

echo "=========================================="
echo "Training Reward Model"
echo "Config: $CONFIG"
echo "Seed: $SEED"
echo "=========================================="

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="rm_train_${TIMESTAMP}"
RUN_DIR="$OUTPUT_DIR/$RUN_ID"

mkdir -p "$RUN_DIR"

# Step 1: Generate training data (if not exists)
python -m src.rm.data_generation \
  --config "$CONFIG" \
  --env_config "$ENV_CONFIG" \
  --task_manifest "$TASK_MANIFEST" \
  --output_dir "$RUN_DIR/data"

# Step 2: Train RM ensemble (3 models)
for i in {1..3}; do
  echo "Training RM model $i/3..."
  python -m src.rm.training \
    --config "$CONFIG" \
    --data_dir "$RUN_DIR/data" \
    --model_id "$i" \
    --seed $((SEED + i - 1)) \
    --output_dir "$RUN_DIR/checkpoints/model_$i"
done

# Step 3: Evaluate on dev set
python -m src.rm.evaluate \
  --checkpoint_dir "$RUN_DIR/checkpoints" \
  --split "S_rm_dev" \
  --task_manifest "$TASK_MANIFEST" \
  --output_dir "$RUN_DIR/eval"

echo "RM training completed: $RUN_DIR"
