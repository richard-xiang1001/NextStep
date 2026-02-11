#!/bin/bash
# Run baseline experiments on WebShop
# Usage: ./scripts/run_baselines.sh --method greedy --seed 42

set -euo pipefail

# Default values
METHOD="greedy"
SEED=42
ENV_CONFIG="configs/env/webshop_v0.1.0.json"
TASK_MANIFEST="configs/env/webshop_task_manifest_v0.1.0.json"
OUTPUT_DIR="results/exp_logs"
SPLIT="S_final_test"
NUM_SAMPLES=1
TEMPERATURE=0.7
LLM_PROVIDER="openai"
LLM_MODEL="gpt-4.1-mini"
LLM_TOP_P=""
LLM_STOP=""
LLM_REASONING_EFFORT=""
LLM_TOOLS_ENABLED=0
MAX_TASKS=""
RM_CHECKPOINT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --method)
      METHOD="$2"
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
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --num-samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --llm-provider)
      LLM_PROVIDER="$2"
      shift 2
      ;;
    --llm-model)
      LLM_MODEL="$2"
      shift 2
      ;;
    --llm-top-p)
      LLM_TOP_P="$2"
      shift 2
      ;;
    --llm-stop)
      LLM_STOP="$2"
      shift 2
      ;;
    --llm-reasoning-effort)
      LLM_REASONING_EFFORT="$2"
      shift 2
      ;;
    --llm-tools-enabled)
      LLM_TOOLS_ENABLED=1
      shift 1
      ;;
    --max-tasks)
      MAX_TASKS="$2"
      shift 2
      ;;
    --rm-checkpoint)
      RM_CHECKPOINT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "=========================================="
echo "Running Baseline: $METHOD"
echo "Seed: $SEED"
echo "Split: $SPLIT"
echo "LLM: $LLM_PROVIDER / $LLM_MODEL"
echo "=========================================="

# Create experiment ID
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_ID="baseline_${METHOD}_${TIMESTAMP}"
EXP_DIR="$OUTPUT_DIR/$EXP_ID"

mkdir -p "$EXP_DIR"

PYTHON_BIN="python3"
if [[ -x "venv/bin/python" ]]; then
  PYTHON_BIN="venv/bin/python"
elif ! command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

# Run baseline
CMD=(
  "$PYTHON_BIN" -m src.eval.baselines
  --method "$METHOD"
  --seed "$SEED"
  --env_config "$ENV_CONFIG"
  --task_manifest "$TASK_MANIFEST"
  --split "$SPLIT"
  --num_samples "$NUM_SAMPLES"
  --temperature "$TEMPERATURE"
  --llm_provider "$LLM_PROVIDER"
  --llm_model "$LLM_MODEL"
  --output_dir "$EXP_DIR"
)

if [[ -n "$LLM_TOP_P" ]]; then
  CMD+=(--llm_top_p "$LLM_TOP_P")
fi

if [[ -n "$LLM_STOP" ]]; then
  CMD+=(--llm_stop "$LLM_STOP")
fi

if [[ -n "$LLM_REASONING_EFFORT" ]]; then
  CMD+=(--llm_reasoning_effort "$LLM_REASONING_EFFORT")
fi

if [[ "$LLM_TOOLS_ENABLED" == "1" ]]; then
  CMD+=(--llm_tools_enabled)
fi

if [[ -n "$MAX_TASKS" ]]; then
  CMD+=(--max_tasks "$MAX_TASKS")
fi

if [[ -n "$RM_CHECKPOINT" ]]; then
  CMD+=(--rm_checkpoint "$RM_CHECKPOINT")
fi

"${CMD[@]}" 2>&1 | tee "$EXP_DIR/log.txt"

echo "Experiment completed: $EXP_DIR"
